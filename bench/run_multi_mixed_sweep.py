"""Multi-adapter mixed (chunked-prefill) steps: every listed implementation over a grid of steps.

    python bench/run_multi_mixed_sweep.py bench/configs/multi_mixed_a100_r64.yaml --out results/.../r64.jsonl
    python bench/run_multi_mixed_sweep.py bench/configs/multi_mixed_smoke.yaml --out /tmp/x.jsonl --resume

Steps come from rankfuse.mixed_multi (the four step types of bench/configs/rf2_mixed.yaml, each
request with its own adapter from a Zipf popularity and its own rank). Timing, batch statistics
and the row format are bench/run_multi_sweep.py's (time_graph, batch_stats): GPU time by CUDA-graph
replay, per-step metadata (`prepare`, which also holds each kernel's sort or grouping of tokens)
built outside the graph, an L2 flush inside the graph with its replay time subtracted, every call
adding into a preallocated output. One JSON object per line; rows also carry the step type, decode
and prefill tokens and the decode share.

`grid` is a dict of lists (a cross product) or a list of such dicts, joined, as in run_sweep.py.
Workloads are ordered so one set of adapter weights serves every step that shares it. --resume
skips (workload, impl) pairs already in --out; --impls runs a subset of the config's impls.
B4 (b4_multi_*) runs S-LoRA's bgmv decode kernel on every token, prompts included: label its
prefill-heavy rows "S-LoRA bgmv applied to all tokens" (S-LoRA itself uses a separate prefill
kernel for prompts).
After the sweep, the per-step prepare cost of the same steps (bench/mixed_prepare_cost.py) goes to
`<out stem>-prepare.csv`.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rankfuse.mixed_multi import MixedMultiWorkload, make_batch  # noqa: E402
from rankfuse.multi import MULTI_REGISTRY, load_all, make_weights  # noqa: E402
from run_multi_sweep import batch_stats, git_sha, time_graph  # noqa: E402

FIELDS = list(MixedMultiWorkload.__dataclass_fields__)


def expand(grid) -> list[MixedMultiWorkload]:
    if isinstance(grid, list):
        return [w for g in grid for w in expand(g)]
    keys = list(grid)
    vals = [v if isinstance(v, list) else [v] for v in grid.values()]
    return [MixedMultiWorkload(**dict(zip(keys, combo))) for combo in itertools.product(*vals)]


def weights_key(w: MixedMultiWorkload) -> tuple:
    return (w.seed, w.r_max, w.h_in, w.h_out, w.num_adapters, w.dtype, w.pages)


def row_key(impl: str, w: MixedMultiWorkload) -> tuple:
    return (impl, *(str(getattr(w, f)) for f in FIELDS))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--out", required=True)
    p.add_argument("--impls", default=None)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    load_all(include_slora=True)
    impls = a.impls.split(",") if a.impls else cfg["impls"]
    warmup, reps, replays = cfg.get("warmup", 10), cfg.get("reps", 7), cfg.get("graph_replays", 20)
    flush_l2 = cfg.get("flush_l2", True)
    wl = sorted(expand(cfg["grid"]), key=weights_key)  # stable: grid order within one set of weights
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if a.resume and out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["impl"], *(str(r[f]) for f in FIELDS)))
    gname, sha = torch.cuda.get_device_name(), git_sha()
    print(f"{len(wl)} workloads x {len(impls)} impls on {gname} -> {out_path} (resume: {len(done)} rows done)", flush=True)
    wk, W = None, None
    with out_path.open("a") as f:
        for w in wl:
            todo = [n for n in impls if row_key(n, w) not in done]
            if not todo:
                continue
            if weights_key(w) != wk:
                W = None
                torch.cuda.empty_cache()
                W, wk = make_weights(w.weights_workload(), "cuda"), weights_key(w)
            x, ta, tr = make_batch(w)
            stats = batch_stats(W, ta, tr)
            base = torch.randn(x.shape[0], W.h_out, device="cuda").to(x.dtype)
            out = base.clone()
            for name in todo:
                impl = MULTI_REGISTRY[name]
                reason = impl.unsupported(W)
                if reason:
                    continue
                err, times = None, [float("nan")]
                try:
                    kw = impl.prepare(x, W, ta, tr) if impl.prepare else {}
                    fn = lambda: impl(x, W, ta, tr, out=out, add=True, **kw)  # noqa: E731
                    times = time_graph(fn, warmup, reps, replays, flush_l2, reset=lambda: out.copy_(base))
                except Exception as e:  # report and go on
                    torch.cuda.synchronize()
                    err = f"{type(e).__name__}: {e}"[:300]
                q = statistics.quantiles(times, n=10) if len(times) > 1 else [times[0]] * 9
                row = dict(impl=name, gpu=gname, git_sha=sha, timing="graph", **w.as_row(), **stats,
                           median_us=statistics.median(times), p20_us=q[1], p80_us=q[7], reps=reps, replays=replays,
                           flush_l2=flush_l2, add=True, error=err,
                           b4_note="S-LoRA bgmv applied to all tokens" if name.startswith("b4_") else None,
                           time=datetime.now().isoformat(timespec="seconds"))
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"  {name:<16} r{w.r_max:<4} {str(w.lengths):<18} D={w.num_decodes:<4} N={w.num_adapters:<3} "
                      f"a={w.zipf_alpha:<4} {w.dist:<8} {str(w.prefill_dist):<8} s{w.seed} T={x.shape[0]:<5} "
                      f"{row['median_us']:9.1f} us" + (f"  ERROR {err}" if err else ""), flush=True)
                kw = fn = None
    if not a.impls:  # a subset rerun (--impls) leaves the prepare-cost file alone
        import mixed_prepare_cost

        mixed_prepare_cost.after_sweep("multi", a.config, a.out)


if __name__ == "__main__":
    main()
