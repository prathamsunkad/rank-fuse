"""What each RF1 optimization is worth: remove one at a time from the tuned kernel.

    python bench/rf1_ablation.py --out results/raw/rf2/rf1_ablation.csv

Every ablation changes settings the host reads when launching, so no kernel changes. Results are
what you lose by removing that one optimization on a batch; they interact, so do not add them up.
RF2 shares these settings. "tuned" runs first and last, which measures noise between runs.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import statistics
from contextlib import contextmanager
from pathlib import Path

import torch

from run_sweep import gpu_name, git_sha, time_graph  # also puts the repo on sys.path

from rankfuse.interface import REGISTRY  # noqa: E402
from rankfuse.kernels import rankfuse_fused as rf  # noqa: E402
from rankfuse.workloads import Workload, make_batch, make_lora  # noqa: E402

ABLATIONS: dict[str, dict] = {
    "tuned": {},
    "no_split_k": {"MAX_SPLIT_K": 1, "MIN_SPLIT_K_TILED": 1},
    "no_load_pipelining": {"SPLITK_STAGES": 1},
    "short_shrink_step": {"SPLITK_BLOCK_H": 64},
    "one_expand_wave": {"EXPAND_WAVES": 1},
    "no_tile_expand": {"EXPAND_TILED_MIN_TOKENS": 10**9},
    "tile_expand_4_warps": {"EXPAND_TILED_WARPS": 4},
    "no_aligned_hint": {"USE_ALIGNED_HINT": False},
    "fixed_16_token_blocks": {"_block_m": lambda num_tokens, r_pad: 16},
    "separate_add_pass": {},  # the delta is written, then added to the output in a second pass
    "tuned_again": {},
}
BATCHES = [(32, 1), (512, 1), (2048, 1), (8192, 1), (16, 512)]


@contextmanager
def settings(values: dict):
    saved = {k: getattr(rf, k) for k in values}
    for k, v in values.items():
        setattr(rf, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(rf, k, v)


def report(rows: list[dict], path: Path) -> None:
    """Median time(ablation) / time(tuned), so above 1 means the optimization helps."""
    tuned = {(r["gpu"], r["requests"], r["tokens_per_request"], r["r_max"], r["dist"], r["seed"]): r["median_us"]
             for r in rows if r["ablation"] == "tuned"}
    cells: dict[tuple, list[float]] = {}
    for r in rows:
        k = (r["gpu"], r["requests"], r["tokens_per_request"], r["r_max"], r["dist"], r["seed"])
        if r["ablation"] != "tuned" and r["median_us"] and k in tuned:
            cells.setdefault((r["gpu"], r["ablation"], r["requests"] * r["tokens_per_request"], r["r_max"]), []).append(r["median_us"] / tuned[k])
    lines = ["| GPU | removed | tokens | r_max | slower by |", "|---|---|---|---|---|"]
    for (gpu, name, tokens, r_max), v in sorted(cells.items()):
        lines.append(f"| {gpu} | {name} | {tokens} | {r_max} | {statistics.median(v):.2f}x |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gname, sha, rows = gpu_name(), git_sha(), []
    impl = REGISTRY["rf1_fused"]
    with out.open("w", newline="") as f:
        writer = None
        for (n, tpr), r_max, dist, seed in itertools.product(BATCHES, [64, 128], ["all_max", "uniform"], [0, 1]):
            w = Workload(n, tpr, r_max=r_max, dist=dist, seed=seed)
            lora, batch = make_lora(w, "cuda"), make_batch(w, "cuda")
            args = (batch.x, lora, batch.token_ranks)
            kwargs = impl.prepare(*args)
            target = torch.zeros(batch.x.shape[0], w.h_out, dtype=batch.x.dtype, device="cuda")

            def add_in_kernel(x, l, ranks, **kw):
                return impl(x, l, ranks, out=target, add=True, **kw)

            def add_after(x, l, ranks, **kw):
                return target.add_(impl(x, l, ranks, **kw))

            for name, values in ABLATIONS.items():
                fn = add_after if name == "separate_add_pass" else add_in_kernel
                try:
                    with settings(values):
                        times = time_graph(fn, args, kwargs, 10, 30, 20, True)
                    us, error = statistics.median(times), ""
                except Exception as e:  # report it and carry on: other ablations are still useful
                    torch.cuda.synchronize()
                    us, error = "", f"{type(e).__name__}: {e}"[:200]
                row = {"ablation": name, "gpu": gname, "git_sha": sha, "requests": n, "tokens_per_request": tpr, "r_max": r_max,
                       "dist": dist, "seed": seed, "median_us": us, "error": error}
                rows.append(row)
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
                print(row, flush=True)
    report(rows, out.with_suffix(".md"))


if __name__ == "__main__":
    main()
