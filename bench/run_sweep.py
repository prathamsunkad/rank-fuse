"""Benchmark every requested implementation over a grid of workloads; append rows to a CSV.

    python bench/run_sweep.py bench/configs/smoke.yaml
    python bench/run_sweep.py bench/configs/decode.yaml --out results/raw/decode.csv

Timing: CUDA events per repetition (CPU: perf_counter), with an optional L2 flush before each
repetition so adapter weights start cold, as they would in a real model forward pass.
Each row also carries the roofline lower bound for the matching strategy.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rankfuse  # noqa: E402  (registers implementations)
from rankfuse import roofline  # noqa: E402
from rankfuse.interface import REGISTRY  # noqa: E402
from rankfuse.workloads import Workload, make_batch, make_lora  # noqa: E402

# Which roofline strategy each implementation should be compared against.
ROOFLINE_STRATEGY = {
    "b1_padded": "padded",
    "b2_grouped": "grouped",
    "b2_grouped_graphed": "grouped",
    "b3_vllm": "separate_padded",
    "b4_slora": "separate_true",
}
DEFAULT_STRATEGY = "fused_ideal"  # RankFuse kernels

L2_FLUSH_BYTES = 256 * 2**20


def gpu_name() -> str:
    return torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"


def roofline_gpu_key(name: str) -> str | None:
    n = name.upper()
    for key in roofline.GPUS:
        model, *variant = key.upper().split("-")
        if model in n and all(v.replace("GB", "") in n for v in variant):
            return key
    for key in roofline.GPUS:
        if key.split("-")[0].upper() in n:
            return key
    return None


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def time_impl(fn, args, device: str, warmup: int, reps: int, flush_l2: bool) -> list[float]:
    """Return per-repetition times in microseconds."""
    for _ in range(warmup):
        fn(*args)
    times = []
    if device == "cuda":
        flush = torch.empty(L2_FLUSH_BYTES, dtype=torch.uint8, device="cuda") if flush_l2 else None
        torch.cuda.synchronize()
        for _ in range(reps):
            if flush is not None:
                flush.zero_()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn(*args)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1e3)
    else:
        for _ in range(reps):
            t0 = time.perf_counter()
            fn(*args)
            times.append((time.perf_counter() - t0) * 1e6)
    return times


def expand_grid(grid: dict) -> list[Workload]:
    keys = list(grid)
    values = [v if isinstance(v, list) else [v] for v in grid.values()]
    return [Workload(**dict(zip(keys, combo))) for combo in itertools.product(*values)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--out", default=None, help="CSV path (default: results/raw/<name>-<gpu>-<time>.csv)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    impls = sorted(n for n in REGISTRY if n != "b0_reference") if cfg.get("impls", "all") == "all" else cfg["impls"]
    impls = [n for n in impls if not (REGISTRY[n].requires_cuda and a.device != "cuda")]
    workloads = expand_grid(cfg["grid"])

    gname = gpu_name()
    gkey = roofline_gpu_key(gname)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(a.out or f"results/raw/{cfg['name']}-{gname.replace(' ', '_')}-{stamp}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    sha = git_sha()
    print(f"{len(workloads)} workloads x {len(impls)} impls on {gname} (roofline: {gkey}) -> {out}")

    with out.open("w", newline="") as f:
        writer = None
        for w in workloads:
            lora, batch = make_lora(w, a.device), make_batch(w, a.device)
            token_ranks = batch.token_ranks.tolist()
            for name in impls:
                times = time_impl(
                    REGISTRY[name], (batch.x, lora, batch.token_ranks), a.device,
                    cfg.get("warmup", 10), cfg.get("reps", 50), cfg.get("flush_l2", True),
                )
                q = statistics.quantiles(times, n=10)
                strategy = ROOFLINE_STRATEGY.get(name, DEFAULT_STRATEGY)
                c = roofline.cost(strategy, token_ranks, w.h_in, w.h_out, w.r_max, w.dtype)
                row = {
                    "impl": name, "gpu": gname, "git_sha": sha, **w.as_row(),
                    "batch_max_rank": max(token_ranks), "mean_rank": sum(token_ranks) / len(token_ranks),
                    "median_us": statistics.median(times), "p20_us": q[1], "p80_us": q[7],
                    "roofline_strategy": strategy, "roofline_bytes": c.total_bytes, "roofline_flops": c.flops,
                    "roofline_us": c.time_s(gkey) * 1e6 if gkey else "",
                }
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
                print(f"  {name:<20} T={w.num_tokens:<5} h={w.h_in:<5} r_max={w.r_max:<4} {w.dist:<9} "
                      f"{row['median_us']:9.1f} us")


if __name__ == "__main__":
    main()
