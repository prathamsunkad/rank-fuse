"""Measure pinned host-to-device (H2D) PCIe copy latency and bandwidth on H200.

Measures:
  1. Single projection (4096 x 4096, fp16):
     - 1 page (rank 16): 256 KB
     - 2 pages (rank 32): 512 KB
     - 4 pages (rank 64): 1024 KB
     - Full padded adapter (rank 64): 1024 KB
  2. Whole-model Qwen2.5-7B sizes:
     - 1 page: 81 MB
     - 2 pages: 162 MB
     - 4 pages: 324 MB
     - Full padded adapter: 323 MB

Uses cudaEvent timing, >= 50 repeats, reporting median and p95 latency and GB/s.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def measure_h2d(
    size_bytes: int,
    warmup: int = 10,
    repeats: int = 50,
) -> tuple[float, float, float, float]:
    """Measure H2D copy latency for size_bytes.

    Returns:
        (median_ms, p95_ms, median_gb_s, p95_gb_s)
    """
    host_buf = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=True)
    gpu_buf = torch.empty(size_bytes, dtype=torch.uint8, device="cuda")

    stream = torch.cuda.current_stream()

    # Warmup
    for _ in range(warmup):
        gpu_buf.copy_(host_buf, non_blocking=True)
    torch.cuda.synchronize()

    times_ms = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(repeats):
        start.record(stream)
        gpu_buf.copy_(host_buf, non_blocking=True)
        end.record(stream)
        end.synchronize()
        times_ms.append(start.elapsed_time(end))

    med_ms = float(np.median(times_ms))
    p95_ms = float(np.percentile(times_ms, 95))

    # Bandwidth in GB/s (10^9 bytes / sec)
    # bytes / (ms * 1e-3) / 1e9 = bytes / (ms * 1e6)
    med_bw = (size_bytes / (med_ms * 1e6)) if med_ms > 0 else 0.0
    p95_bw = (size_bytes / (p95_ms * 1e6)) if p95_ms > 0 else 0.0

    return med_ms, p95_ms, med_bw, p95_bw


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/h200_2026-11-12_multi_adapter/source/h2d_timing.json")
    p.add_argument("--repeats", type=int, default=50)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("h2d_timing requires CUDA (H200)")

    device_name = torch.cuda.get_device_name()
    if "H200" not in device_name:
        raise RuntimeError(f"Expected H200, got: {device_name}")

    PAGE_BYTES_PROJ = 16 * (4096 + 4096) * 2  # 256 KB = 262,144 bytes
    ADAPTER_BYTES_PROJ = 64 * (4096 + 4096) * 2  # 1024 KB = 1,048,576 bytes

    PAGE_BYTES_7B = 81 * 1024 * 1024  # 81 MB
    ADAPTER_BYTES_7B = 323 * 1024 * 1024  # 323 MB

    items = [
        # (scope, label, pages, size_bytes)
        ("Single 4096x4096 projection", "1 page (r=16)", 1, PAGE_BYTES_PROJ),
        ("Single 4096x4096 projection", "2 pages (r=32)", 2, 2 * PAGE_BYTES_PROJ),
        ("Single 4096x4096 projection", "4 pages (r=64)", 4, 4 * PAGE_BYTES_PROJ),
        ("Single 4096x4096 projection", "Full adapter (r=64)", 4, ADAPTER_BYTES_PROJ),

        ("Whole-model Qwen2.5-7B", "1 page", 1, PAGE_BYTES_7B),
        ("Whole-model Qwen2.5-7B", "2 pages", 2, 2 * PAGE_BYTES_7B),
        ("Whole-model Qwen2.5-7B", "4 pages", 4, 4 * PAGE_BYTES_7B),
        ("Whole-model Qwen2.5-7B", "Full adapter (r=64)", 4, ADAPTER_BYTES_7B),
    ]

    print(f"Running H2D copy timing on {device_name} (repeats={args.repeats})...", flush=True)

    results = []
    for scope, label, pages, size_bytes in items:
        med_ms, p95_ms, med_bw, p95_bw = measure_h2d(size_bytes, repeats=args.repeats)
        size_mb = size_bytes / (1024 * 1024)
        rec = {
            "scope": scope,
            "label": label,
            "pages": pages,
            "size_bytes": size_bytes,
            "size_mb": round(size_mb, 3),
            "median_ms": round(med_ms, 3),
            "p95_ms": round(p95_ms, 3),
            "median_gb_s": round(med_bw, 2),
            "p95_gb_s": round(p95_bw, 2),
            "device": device_name,
            "git_commit": get_git_commit(),
        }
        results.append(rec)
        print(f"  {scope} | {label} ({size_mb:.2f} MB): median {med_ms:.3f} ms ({med_bw:.1f} GB/s), p95 {p95_ms:.3f} ms", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save JSON
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote JSON to {out_path}", flush=True)

    # Save CSV alongside JSON
    csv_path = out_path.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote CSV to {csv_path}", flush=True)


if __name__ == "__main__":
    main()
