"""What ordering a batch by rank costs a scheduler, one step, on the CPU and the GPU.

    python bench/sort_cost.py --out results/raw/rf2/sort_cost.csv

requests_cpu orders whole requests (one rank per request) on the host, which is what rank-ordered
batches on rf2_segment assume a scheduler does; tokens_gpu is a stable token argsort on the GPU,
with the wait for it included. Prints and writes median us.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import torch


def timed(fn, reps: int) -> float:
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(times)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--reps", type=int, default=200)
    a = p.parse_args()
    tiers = [1, 2, 4, 8, 16, 32, 64]
    gen = torch.Generator().manual_seed(0)
    rows = []
    for n in [1, 8, 32, 128, 512, 2048, 8192]:
        ranks = [tiers[i] for i in torch.randint(0, len(tiers), (n,), generator=gen).tolist()]
        row = {"requests": n, "requests_cpu_us": timed(lambda: sorted(range(n), key=ranks.__getitem__), a.reps)}
        if torch.cuda.is_available():
            token_ranks = torch.tensor(ranks, dtype=torch.int32, device="cuda")

            def gpu_sort():
                torch.argsort(token_ranks, stable=True)
                torch.cuda.synchronize()

            row["tokens_gpu_us"] = timed(gpu_sort, a.reps)
            row["gpu_wait_only_us"] = timed(torch.cuda.synchronize, a.reps)
        rows.append(row)
        print(row, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
