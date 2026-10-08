"""Analytic bytes / FLOPs / lower-bound time for each serving strategy.

Run before benchmarking to predict where RankFuse can win:

    python -m rankfuse.roofline --h 4096 --ranks 4 8 16 32 --tokens-per-request 1

Modeling assumptions (keep in sync with docs/PLAN.md):
* y is accumulated (y += delta): output costs a read and a write of T x h_out.
* Unfused strategies write the intermediate v [T, r] to HBM and read it back. The padded
  baseline also pays one extra read+write of v for the mask.
* Separate-adapter strategies (vLLM-, S-LoRA-style) read each distinct adapter's weights once
  per batch. This is optimistic for them: it assumes L2 absorbs repeated per-token gathers.
* Grouped reads the shared prefix once per rank group, and gathers/scatters x and y per group.
* Time lower bound = max(bytes / bandwidth, flops / peak). Real kernels will be slower.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2}

# Nominal spec-sheet numbers: (HBM bytes/s, dense fp16 tensor FLOP/s).
GPUS = {
    "A100-80GB": (2.039e12, 312e12),
    "A100-40GB": (1.555e12, 312e12),
    "H100-SXM": (3.35e12, 989e12),
    "H100-PCIe": (2.0e12, 756e12),
    "V100": (0.9e12, 125e12),
    "L40S": (0.864e12, 362e12),
}

STRATEGIES = ["fused_ideal", "padded", "grouped", "separate_padded", "separate_true"]


@dataclass
class Cost:
    strategy: str
    weight_bytes: int
    activation_bytes: int
    flops: int

    @property
    def total_bytes(self) -> int:
        return self.weight_bytes + self.activation_bytes

    def time_s(self, gpu: str) -> float:
        bw, peak = GPUS[gpu]
        return max(self.total_bytes / bw, self.flops / peak)


def cost(
    strategy: str,
    token_ranks: list[int],
    h_in: int,
    h_out: int,
    r_max: int,
    dtype: str = "fp16",
) -> Cost:
    e = DTYPE_BYTES[dtype]
    T = len(token_ranks)
    r_b = max(token_ranks)  # largest rank in this batch
    distinct = sorted(set(token_ranks))
    h = h_in + h_out
    sum_r = sum(token_ranks)
    xy = T * h_in + 2 * T * h_out  # read x, read + write y

    if strategy == "fused_ideal":  # RankFuse target: shared prefix once, v stays on chip
        w_rows, act, flops = r_b, xy, 2 * sum_r * h
    elif strategy == "padded":  # B1
        w_rows, act, flops = r_b, xy + 4 * T * r_b, 2 * T * r_b * h
    elif strategy == "grouped":  # B2: x gather (r+w) + matmul read; y matmul write + scatter (r+w)
        w_rows = sum(distinct)
        act = 3 * T * h_in + 3 * T * h_out + T * h_out + 2 * sum_r
        flops = 2 * sum_r * h
    elif strategy == "separate_padded":  # B3, vLLM-style: one adapter per rank, padded to r_max
        w_rows, act, flops = len(distinct) * r_max, xy + 2 * T * r_max, 2 * T * r_max * h
    elif strategy == "separate_true":  # B4, S-LoRA-style: one adapter per rank, true ranks
        w_rows, act, flops = sum(distinct), xy + 2 * sum_r, 2 * sum_r * h
    else:
        raise ValueError(f"unknown strategy {strategy}")
    return Cost(strategy, w_rows * h * e, act * e, flops)


def table(token_ranks, h_in, h_out, r_max, dtype="fp16", gpu="A100-80GB") -> list[dict]:
    rows = []
    for s in STRATEGIES:
        c = cost(s, token_ranks, h_in, h_out, r_max, dtype)
        rows.append(
            {
                "strategy": s,
                "weight_MiB": c.weight_bytes / 2**20,
                "activation_MiB": c.activation_bytes / 2**20,
                "total_MiB": c.total_bytes / 2**20,
                "GFLOP": c.flops / 1e9,
                "intensity": c.flops / c.total_bytes,
                "lower_bound_us": c.time_s(gpu) * 1e6,
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h", type=int, default=4096, help="h_in (and h_out unless --h-out)")
    p.add_argument("--h-out", type=int, default=None)
    p.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16, 32], help="one rank per request")
    p.add_argument("--r-max", type=int, default=None, help="adapter r_max (default: max of --ranks)")
    p.add_argument("--tokens-per-request", type=int, default=1, help="1 = decode")
    p.add_argument("--repeat", type=int, default=1, help="repeat the --ranks list this many times")
    p.add_argument("--dtype", default="fp16", choices=list(DTYPE_BYTES))
    p.add_argument("--gpu", default="A100-80GB", choices=list(GPUS))
    a = p.parse_args()

    req_ranks = a.ranks * a.repeat
    token_ranks = [r for r in req_ranks for _ in range(a.tokens_per_request)]
    h_out = a.h_out or a.h
    r_max = a.r_max or max(a.ranks)
    print(f"T={len(token_ranks)} tokens, requests={len(req_ranks)}, h_in={a.h}, h_out={h_out}, "
          f"r_max={r_max}, {a.dtype}, {a.gpu}")
    hdr = f"{'strategy':<16}{'weights MiB':>12}{'acts MiB':>10}{'total MiB':>11}{'GFLOP':>9}{'FLOP/B':>8}{'>= us':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in table(token_ranks, a.h, h_out, r_max, a.dtype, a.gpu):
        print(f"{r['strategy']:<16}{r['weight_MiB']:>12.3f}{r['activation_MiB']:>10.3f}{r['total_MiB']:>11.3f}"
              f"{r['GFLOP']:>9.3f}{r['intensity']:>8.1f}{r['lower_bound_us']:>9.2f}")


if __name__ == "__main__":
    main()
