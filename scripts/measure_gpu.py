"""Measure laptop-GPU FP16 matmul peak and device-copy bandwidth.

The matmul result is a measured fp16 matmul peak, not a spec number. The copy
measurement counts both the device read and device write against theoretical
GDDR6 bandwidth.
"""

from __future__ import annotations

import argparse
import sys

import torch


def best_matmul_flops(size: int, warmup: int, reps: int) -> float:
    a = torch.randn((size, size), device="cuda", dtype=torch.float16)
    b = torch.randn((size, size), device="cuda", dtype=torch.float16)
    out = torch.empty((size, size), device="cuda", dtype=torch.float16)
    for _ in range(warmup):
        torch.mm(a, b, out=out)
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.mm(a, b, out=out)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 1e3)
    return max(2 * size**3 / seconds for seconds in samples)


def best_copy_bandwidth(gib: int, warmup: int, reps: int) -> float:
    source = torch.empty(gib * 2**30, device="cuda", dtype=torch.uint8)
    target = torch.empty_like(source)
    for _ in range(warmup):
        target.copy_(source)
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        target.copy_(source)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 1e3)
    return max(2 * source.numel() / seconds for seconds in samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matmul-size", type=int, default=8192)
    parser.add_argument("--copy-gib", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to measure on CPU")

    print(f"GPU: {torch.cuda.get_device_name()}")
    matmul = best_matmul_flops(args.matmul_size, args.warmup, args.reps)
    bandwidth = best_copy_bandwidth(args.copy_gib, args.warmup, args.reps)
    theoretical = 256e9
    print(f"measured_fp16_matmul_peak_flops: {matmul:.6e}")
    print(f"measured_copy_bandwidth_bytes_per_s: {bandwidth:.6e}")
    print(f"copy_bandwidth_percent_of_256GBps: {100 * bandwidth / theoretical:.2f}%")


if __name__ == "__main__":
    sys.exit(main())
