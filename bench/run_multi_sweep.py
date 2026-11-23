"""Stage 2 benchmark: multi-adapter implementations over adapters x popularity x ranks x tokens.

    python bench/run_multi_sweep.py bench/configs/multi_decode.yaml --out results/.../raw/decode.jsonl
    python bench/run_multi_sweep.py bench/configs/multi_smoke.yaml
    python bench/run_multi_sweep.py --ncu rf1_ma 4096x4096 16 32 1     # one call for Nsight Compute

Timing is GPU time by CUDA-graph replay, the method of bench/run_sweep.py --timing graph: the
implementation's per-step metadata (`prepare`: tile tables, vLLM's LoRAKernelMeta and pair copies,
S-LoRA's pool and entries, ...) is built outside the graph, the call is captured once and replayed
`graph_replays` times back to back per repetition; an L2 flush inside the graph keeps weights
cold and its own replay time is subtracted. Every call adds into a preallocated output (add=True),
as a LoRA layer adds its delta to the base projection's output; out is reset outside the timing.

Each row (one JSON object per line) records the workload, batch statistics (active adapters,
distinct (adapter, rank) pairs, tokens per active adapter), the median / p20 / p80 time and an
ideal-bytes lower bound: x read + out read and written + each active adapter's weights up to its
group's maximum rank, read once.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rankfuse.multi import MULTI_REGISTRY, MultiWorkload, load_all, make_batch, make_weights  # noqa: E402

L2_FLUSH_BYTES = 256 * 2**20
HBM_BYTES_PER_S = 4.8e12  # H200 spec sheet


def time_graph(fn, warmup: int, reps: int, replays: int, flush_l2: bool, reset=None) -> list[float]:
    """Per-repetition GPU time (us) of fn() by CUDA-graph replay (as bench/run_sweep.py)."""
    flush = torch.empty(L2_FLUSH_BYTES, dtype=torch.uint8, device="cuda") if flush_l2 else None

    def capture(body):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                body()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        return graph

    def per_replay_us(graph) -> float:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1e3 / replays

    def call():
        if flush is not None:
            flush.zero_()
        fn()

    graph = capture(call)
    flush_graph = capture(flush.zero_) if flush is not None else None
    torch.cuda.synchronize()
    flush_us = statistics.median(per_replay_us(flush_graph) for _ in range(5)) if flush_graph else 0.0
    times = [per_replay_us(graph) - flush_us for _ in range(reps)]
    if reset is not None:
        reset()
    del graph
    return times


def batch_stats(W, ta: torch.Tensor, tr: torch.Tensor) -> dict:
    ad, rk = ta.long().cpu(), tr.long().cpu()
    live = (rk > 0) & (ad >= 0)
    act = torch.unique(ad[live])
    lim = {int(a): int(rk[live & (ad == a)].max()) for a in act}
    elem = W.A_pool.element_size()
    T = ad.numel()
    weight_bytes = sum(r * (W.h_in + W.h_out) * elem for r in lim.values())
    act_bytes = T * W.h_in * elem + 2 * T * W.h_out * elem
    pairs = torch.unique(torch.stack([ad[live], rk[live]], 1), dim=0).shape[0] if live.any() else 0
    return dict(
        active_adapters=len(lim), distinct_pairs=int(pairs),
        tokens_per_active_adapter=float(live.sum()) / max(len(lim), 1),
        batch_max_rank=int(rk.max()) if T else 0, mean_rank=float(rk.float().mean()) if T else 0.0,
        ideal_bytes=weight_bytes + act_bytes, ideal_us=(weight_bytes + act_bytes) / HBM_BYTES_PER_S * 1e6,
    )


def workloads(cfg: dict):
    shapes = cfg["shapes"]
    g = cfg["grid"]
    keys = ["num_adapters", "popularity", "dist", "pages"]
    vals = [g[k] if isinstance(g.get(k), list) else [g.get(k, MultiWorkload.__dataclass_fields__[k].default)] for k in keys]
    for (label, (req, tpr)), shape, combo in itertools.product(cfg["tokens"].items(), shapes, itertools.product(*vals)):
        h_in, h_out = shapes[shape]
        p = dict(zip(keys, combo))
        yield label, shape, MultiWorkload(req, tpr, p["num_adapters"], popularity=p["popularity"], dist=p["dist"],
                                          h_in=h_in, h_out=h_out, r_max=g.get("r_max", 64), dtype=g.get("dtype", "fp16"),
                                          seed=g.get("seed", 0), pages=p["pages"])


def stage1_rf1(W, x, tr):
    """Stage 1's rf1_fused on adapter 0 (no page table): the reference for one adapter."""
    from rankfuse.interface import MatryoshkaLoRA
    from rankfuse.kernels.rankfuse_fused import rankfuse_lora

    lora = MatryoshkaLoRA(W.A_dense[0, 0].contiguous(), W.B_dense[0, 0].contiguous())
    r_limit = int(tr.max())

    def call(out):
        return rankfuse_lora(x, lora.A_t, lora.B, tr, out=out, add=True, r_limit=r_limit)

    return call


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, cwd=Path(__file__).parent).strip()
    except Exception:
        return "unknown"


def run_sweep(cfg: dict, out_path: Path, only_impls=None) -> None:
    load_all(include_slora=True)
    impls = only_impls or cfg["impls"]
    warmup, reps, replays = cfg.get("warmup", 10), cfg.get("reps", 7), cfg.get("graph_replays", 20)
    flush_l2 = cfg.get("flush_l2", True)
    gname = torch.cuda.get_device_name()
    sha = git_sha()
    wl = list(workloads(cfg))
    print(f"{len(wl)} workloads x {len(impls)} impls on {gname} -> {out_path}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    weights_key, W = None, None
    with out_path.open("a") as f:
        for label, shape, w in wl:
            key = (w.h_in, w.h_out, w.num_adapters, w.pages, w.dtype, w.seed)
            if key != weights_key:
                W = None
                torch.cuda.empty_cache()
                W, weights_key = make_weights(w, "cuda"), key
            x, ta, tr = make_batch(w)
            stats = batch_stats(W, ta, tr)
            phase = "decode" if w.tokens_per_request == 1 else "prefill"
            base = torch.randn(x.shape[0], W.h_out, device="cuda").to(x.dtype)
            out = base.clone()
            for name in impls:
                if name == "stage1_rf1_fused":
                    if w.num_adapters != 1:
                        continue
                    call = stage1_rf1(W, x, tr)
                    fn = lambda: call(out)  # noqa: E731
                else:
                    impl = MULTI_REGISTRY[name]
                    if impl.unsupported(W):
                        continue
                    # (baselines read W.A_dense / W.B_dense or their own copies; the page layout
                    # only changes the RankFuse kernels)
                    kw = impl.prepare(x, W, ta, tr) if impl.prepare else {}
                    fn = lambda: impl(x, W, ta, tr, out=out, add=True, **kw)  # noqa: E731
                try:
                    times = time_graph(fn, warmup, reps, replays, flush_l2, reset=lambda: out.copy_(base))
                    err = None
                except Exception as e:  # report and go on
                    torch.cuda.synchronize()
                    times, err = [float("nan")], f"{type(e).__name__}: {e}"[:300]
                q = statistics.quantiles(times, n=10) if len(times) > 1 else [times[0]] * 9
                row = dict(impl=name, gpu=gname, git_sha=sha, timing="graph", label=label, phase=phase, shape=shape,
                           **w.as_row(), **stats, median_us=statistics.median(times), p20_us=q[1], p80_us=q[7],
                           reps=reps, replays=replays, flush_l2=flush_l2, add=True, error=err,
                           time=datetime.now().isoformat(timespec="seconds"))
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"  {name:<18} {shape:<10} {label:<12} N={w.num_adapters:<4} {w.popularity:<8} {w.dist:<9} "
                      f"{w.pages:<10} act={stats['active_adapters']:<4} {row['median_us']:9.1f} us"
                      + (f"  ERROR {err}" if err else ""), flush=True)
                kw = fn = None


def run_ncu(name: str, shape: str, num_adapters: int, requests: int, tpr: int, dist: str, pop: str, pages: str) -> None:
    """One prepared call, profiled between cudaProfilerStart/Stop (use ncu --profile-from-start off)."""
    load_all(include_slora=True)
    shapes = {"4096x4096": (4096, 4096), "q_o": (896, 896), "k_v": (896, 128), "gate_up": (896, 4864), "down": (4864, 896)}
    h_in, h_out = shapes[shape]
    w = MultiWorkload(requests, tpr, num_adapters, popularity=pop, dist=dist, h_in=h_in, h_out=h_out, pages=pages)
    W = make_weights(w, "cuda")
    x, ta, tr = make_batch(w)
    impl = MULTI_REGISTRY[name]
    kw = impl.prepare(x, W, ta, tr) if impl.prepare else {}
    out = torch.zeros(x.shape[0], W.h_out, device="cuda", dtype=x.dtype)
    for _ in range(2):
        impl(x, W, ta, tr, out=out, add=True, **kw)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    impl(x, W, ta, tr, out=out, add=True, **kw)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(json.dumps(dict(impl=name, shape=shape, num_adapters=num_adapters, requests=requests,
                          tokens_per_request=tpr, dist=dist, popularity=pop, pages=pages,
                          prepared=list(kw), **batch_stats(W, ta, tr))), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", nargs="?")
    p.add_argument("--out", default=None, help="JSON-lines path (appended to)")
    p.add_argument("--impls", default=None, help="comma-separated subset of the config's impls")
    p.add_argument("--ncu", nargs="+", metavar="ARG",
                   help="impl shape num_adapters requests tokens_per_request [dist popularity pages]")
    a = p.parse_args()
    if a.ncu:
        args = a.ncu + ["uniform", "uniform", "shuffled"][len(a.ncu) - 5:]
        run_ncu(args[0], args[1], int(args[2]), int(args[3]), int(args[4]), *args[5:8])
        return
    cfg = yaml.safe_load(Path(a.config).read_text())
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(a.out or f"results/raw/{cfg['name']}-{stamp}.jsonl")
    run_sweep(cfg, out, a.impls.split(",") if a.impls else None)


if __name__ == "__main__":
    main()
