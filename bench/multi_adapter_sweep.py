"""Multi-adapter sweep benchmark for H200.

Measures:
  - rf1_ma
  - rf1_ma_flat
  - b1_ma_padded
  - b3_ma
  - b4_ma_copies
  - b4_ma_shared

CUDA-graph replay timing (median and p5/p95 over >= 200 replays).
4096 x 4096, r_max 64, fp16.
Clean wall-time limit stop at 3h20m.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rankfuse.baselines.multi import (
    b0_ma,
    b1_ma_padded,
    b3_ma,
    b4_ma_copies,
    b4_ma_shared,
    prepare_b1_ma_padded,
    prepare_b3_ma,
    prepare_b4_ma_copies,
    prepare_b4_ma_shared,
)
from rankfuse.interface import MatryoshkaLoRA
from rankfuse.kernels.rankfuse_ma import rf1_ma, rf1_ma_flat
from rankfuse.paging import AdapterPageManager, PagePool
from rankfuse.workloads import default_tiers


MAX_WALL_TIME_SEC = 3 * 3600 + 20 * 60  # 3h20m
L2_FLUSH_BYTES = 256 * 1024 * 1024


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def generate_ranks(
    tokens: int,
    mix: str,
    r_max: int = 64,
    rng: random.Random | None = None,
) -> torch.Tensor:
    if rng is None:
        rng = random.Random(0)
    tiers = default_tiers(r_max)
    if mix == "all-8":
        return torch.full((tokens,), 8, dtype=torch.int32)
    elif mix == "all-64":
        return torch.full((tokens,), 64, dtype=torch.int32)
    elif mix == "mostly_low":
        # 80% rank 8, 20% rank 64
        ranks = [8 if rng.random() < 0.8 else 64 for _ in range(tokens)]
        return torch.tensor(ranks, dtype=torch.int32)
    elif mix == "uniform":
        ranks = [rng.choice(tiers) for _ in range(tokens)]
        return torch.tensor(ranks, dtype=torch.int32)
    else:
        raise ValueError(f"Unknown rank mix: {mix}")


def generate_adapter_ids(
    tokens: int,
    adapters: int,
    popularity: str,
    tokens_per_group: int,
    rng: random.Random | None = None,
) -> torch.Tensor:
    if rng is None:
        rng = random.Random(0)
    if adapters <= 1:
        return torch.zeros(tokens, dtype=torch.int32)

    if popularity == "zipf":
        weights = [1.0 / (i + 1) ** 1.2 for i in range(adapters)]
    else:
        weights = [1.0] * adapters

    num_groups = math.ceil(tokens / max(1, tokens_per_group))
    group_adapters = rng.choices(range(adapters), weights=weights, k=num_groups)

    ids = []
    for g_idx in range(num_groups):
        g_len = min(tokens_per_group, tokens - len(ids))
        ids.extend([group_adapters[g_idx]] * g_len)
    return torch.tensor(ids, dtype=torch.int32)


def compute_rows_read(
    impl: str,
    adapter_ids: torch.Tensor,
    token_ranks: torch.Tensor,
    num_adapters: int,
    r_max: int = 64,
) -> tuple[int, int]:
    """Return (rows_read, ideal_rows_read)."""
    # Ideal: each active adapter reads only its max active rank once
    unique_adapters = torch.unique(adapter_ids).tolist()
    ideal = 0
    for a in unique_adapters:
        mask = adapter_ids == a
        if mask.any():
            ideal += int(token_ranks[mask].max())

    if impl in {"rf1_ma_flat", "b1_ma_padded"}:
        actual = ideal
    elif impl == "rf1_ma":
        # Paged in 16-row pages
        actual = 0
        for a in unique_adapters:
            mask = adapter_ids == a
            if mask.any():
                r = int(token_ranks[mask].max())
                actual += math.ceil(r / 16) * 16
    elif impl == "b3_ma":
        # Reads r_max for every distinct (adapter, rank) pair present
        active_pairs = set()
        for i in range(adapter_ids.numel()):
            r = int(token_ranks[i])
            if r > 0:
                active_pairs.add((int(adapter_ids[i]), r))
        actual = len(active_pairs) * r_max
    elif impl in {"b4_ma_copies", "b4_ma_shared"}:
        # S-LoRA BGMV: each token gathers its rank rows
        actual = int(token_ranks.sum().item())
    else:
        actual = ideal
    return actual, ideal


def time_cuda_graph(
    fn,
    warmup: int = 10,
    replays: int = 200,
    flush_l2: bool = True,
) -> tuple[float, float, float]:
    """Capture fn() in a CUDA graph, replay >= 200 times, return (median_us, p5_us, p95_us)."""
    flush = torch.empty(L2_FLUSH_BYTES, dtype=torch.uint8, device="cuda") if flush_l2 else None

    # Warmup on side stream
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)

    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(replays):
        if flush is not None:
            flush.zero_()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)

    med = float(np.median(times))
    p5 = float(np.percentile(times, 5))
    p95 = float(np.percentile(times, 95))
    return med, p5, p95


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--phase", choices=["decode", "prefill", "all"], default="all", help="Phase to run (decode, prefill, or all)")
    parser.add_argument("--n-adapters", default="1,4,16,64,256", help="Comma-separated adapter counts (e.g. '1,4' or '64,256')")
    parser.add_argument("--time-budget-min", type=float, default=200.0, help="Clean wall-time budget in minutes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replays", type=int, default=200)
    args = parser.parse_args()

    max_wall_time_sec = args.time_budget_min * 60.0
    adapter_counts = [int(x.strip()) for x in args.n_adapters.split(",") if x.strip()]

    if not torch.cuda.is_available():
        raise RuntimeError("Multi-adapter sweep requires CUDA (H200)")

    device_name = torch.cuda.get_device_name()
    if "H200" not in device_name:
        raise RuntimeError(f"Expected H200, got: {device_name}")

    start_time = time.time()
    rng = random.Random(args.seed)
    torch_gen = torch.Generator(device="cpu").manual_seed(args.seed)

    H_IN = 4096
    H_OUT = 4096
    R_MAX = 64
    DTYPE = torch.float16

    vllm_ver = "none"
    try:
        import vllm
        vllm_ver = vllm.__version__
    except Exception:
        pass

    git_commit = get_git_commit()

    # Pre-generate adapters for max adapter count (256)
    # Scaled like real adapters (A ~ N(0, 1/h_in), B ~ N(0, 1/r_max)) so outputs are O(1)
    MAX_ADAPTERS = 256
    print(f"Allocating {MAX_ADAPTERS} adapters of shape {R_MAX}x{H_IN} and {R_MAX}x{H_OUT} (fp16, scaled)...", flush=True)
    all_adapters = []
    std_a = 1.0 / math.sqrt(H_IN)
    std_b = 1.0 / math.sqrt(R_MAX)
    for _ in range(MAX_ADAPTERS):
        a_t = (torch.randn(R_MAX, H_IN, dtype=DTYPE, generator=torch_gen) * std_a).cuda()
        b = (torch.randn(R_MAX, H_OUT, dtype=DTYPE, generator=torch_gen) * std_b).cuda()
        all_adapters.append(MatryoshkaLoRA(a_t, b))

    a_pool_all = torch.stack([item.A_t for item in all_adapters]).unsqueeze(0)
    b_pool_all = torch.stack([item.B for item in all_adapters]).unsqueeze(0)
    scales = torch.ones(R_MAX + 1, dtype=torch.float32, device="cuda")

    # Set up PagePool and AdapterPageManager for rf1_ma
    # 256 adapters * 4 pages = 1024 frames
    num_frames = MAX_ADAPTERS * 4
    page_pool = PagePool(num_frames=num_frames, h_in=H_IN, h_out=H_OUT, dtype=DTYPE, device="cuda")
    page_mgr = AdapterPageManager(page_pool, page_rank=16, max_adapters=MAX_ADAPTERS, r_max=R_MAX)
    for idx, item in enumerate(all_adapters):
        page_mgr.register(idx, item.A_t.cpu(), item.B.cpu())
    page_mgr.ensure_resident(list(range(MAX_ADAPTERS)), R_MAX)
    page_mgr.wait_for_copies()

    # Sweep grid
    decode_tokens = [1, 8, 32, 128]
    prefill_tokens = [1024, 8192]
    rank_mixes = ["all-8", "all-64", "mostly_low", "uniform"]
    popularities = ["uniform", "zipf"]

    configs = []
    # 1. Decode configs
    if args.phase in {"decode", "all"}:
        for t in decode_tokens:
            for n in adapter_counts:
                for mix in rank_mixes:
                    for pop in popularities:
                        configs.append({
                            "phase": "decode",
                            "tokens": t,
                            "adapters": n,
                            "tpr": 1,
                            "mix": mix,
                            "pop": pop,
                        })

    # 2. Prefill configs
    if args.phase in {"prefill", "all"}:
        for t in prefill_tokens:
            for n in adapter_counts:
                for mix in rank_mixes:
                    for pop in popularities:
                        # Tokens per group: large groups e.g. t // min(n, 16)
                        tpr = max(1, t // min(n, 16))
                        configs.append({
                            "phase": "prefill",
                            "tokens": t,
                            "adapters": n,
                            "tpr": tpr,
                            "mix": mix,
                            "pop": pop,
                        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "config_idx", "phase", "tokens", "adapters", "tpr", "rank_mix", "popularity",
        "impl", "status", "error", "correctness", "max_abs_diff", "rel_max_diff", "rel_l2_diff",
        "median_us", "p5_us", "p95_us", "rows_read", "ideal_rows_read",
        "gpu", "torch_version", "triton_version", "vllm_version", "git_commit"
    ]

    # Write header if file does not exist or is empty
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    csv_file = out_path.open("a", newline="")
    if write_header:
        csv_file.write(f"# args: phase={args.phase}, n_adapters={args.n_adapters}, time_budget_min={args.time_budget_min}, replays={args.replays}, seed={args.seed}\n")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        csv_file.flush()

    impl_names = ["rf1_ma", "rf1_ma_flat", "b1_ma_padded", "b3_ma", "b4_ma_copies", "b4_ma_shared"]
    print(f"Starting sweep of {len(configs)} configs x {len(impl_names)} implementations on {device_name} (phase={args.phase}, n_adapters={args.n_adapters}, budget={args.time_budget_min}m)...", flush=True)

    for cfg_idx, cfg in enumerate(configs):
        elapsed = time.time() - start_time
        if elapsed >= max_wall_time_sec:
            print(f"Wall-time budget reached ({elapsed:.1f}s >= {max_wall_time_sec:.1f}s). Stopping cleanly.", flush=True)
            break

        tokens = cfg["tokens"]
        adapters_n = cfg["adapters"]
        tpr = cfg["tpr"]
        mix = cfg["mix"]
        pop = cfg["pop"]
        phase = cfg["phase"]

        # Create batch inputs
        x = torch.randn(tokens, H_IN, dtype=DTYPE, device="cuda", generator=torch.Generator(device="cuda").manual_seed(cfg_idx))
        token_ranks = generate_ranks(tokens, mix, R_MAX, rng=rng).cuda()
        adapter_ids = generate_adapter_ids(tokens, adapters_n, pop, tpr, rng=rng).cuda()

        # Sort tokens by adapter_ids so groups are contiguous
        sort_idx = torch.argsort(adapter_ids)
        x = x[sort_idx].contiguous()
        adapter_ids = adapter_ids[sort_idx].contiguous()
        token_ranks = token_ranks[sort_idx].contiguous()

        adapters_subset = all_adapters[:adapters_n]
        a_pool_sub = a_pool_all[:, :adapters_n].contiguous()
        b_pool_sub = b_pool_all[:, :adapters_n].contiguous()

        # Compute reference b0_ma output in FP32
        adapters_subset_fp32 = [MatryoshkaLoRA(item.A_t.float(), item.B.float()) for item in adapters_subset]
        ref_out = b0_ma(x.float(), adapters_subset_fp32, adapter_ids, token_ranks)
        ref_max = float(ref_out.abs().max().item())
        ref_l2 = float(torch.linalg.norm(ref_out).item())

        # Noise floor (FP16 rounding error of FP32 reference)
        noise_diff = ref_out.half().float() - ref_out
        noise_max = float(noise_diff.abs().max().item()) / max(ref_max, 1e-7)
        noise_l2 = float(torch.linalg.norm(noise_diff).item()) / max(ref_l2, 1e-7)
        if cfg_idx == 0:
            print(f"\n[Noise Floor (FP16 round of FP32 ref, cfg 0)] rel_max={noise_max:.6f}, rel_l2={noise_l2:.6f}", flush=True)

        # Pre-allocate output buffer
        out_buf = torch.zeros(tokens, H_OUT, dtype=DTYPE, device="cuda")

        # Prepare metadata for each implementation
        prep_b1 = prepare_b1_ma_padded(x, adapters_subset, adapter_ids, token_ranks)
        prep_b3 = prepare_b3_ma(x, adapters_subset, adapter_ids, token_ranks)
        prep_b4_c = prepare_b4_ma_copies(x, adapters_subset, adapter_ids, token_ranks)
        prep_b4_s = prepare_b4_ma_shared(x, adapters_subset, token_ranks)

        for impl in impl_names:
            status = "OK"
            err_msg = ""
            corr_status = "PASS"
            max_abs = 0.0
            rel_max = 0.0
            rel_l2 = 0.0
            med_us, p5_us, p95_us = 0.0, 0.0, 0.0

            rows_read, ideal_rows = compute_rows_read(impl, adapter_ids, token_ranks, adapters_n, R_MAX)

            try:
                # 1. Define call closure
                if impl == "rf1_ma":
                    fn = lambda: rf1_ma(
                        x, page_pool.frames, page_mgr.page_table, adapter_ids, token_ranks,
                        h_in=H_IN, h_out=H_OUT, r_max=R_MAX, scales=scales, out=out_buf
                    )
                elif impl == "rf1_ma_flat":
                    fn = lambda: rf1_ma_flat(
                        x, a_pool_sub, b_pool_sub, adapter_ids, token_ranks,
                        scales=scales, out=out_buf
                    )
                elif impl == "b1_ma_padded":
                    fn = lambda: b1_ma_padded(
                        x, adapters_subset, adapter_ids, token_ranks,
                        out=out_buf, prepared=prep_b1
                    )
                elif impl == "b3_ma":
                    fn = lambda: b3_ma(
                        x, adapters_subset, adapter_ids, token_ranks,
                        out=out_buf, prepared=prep_b3
                    )
                elif impl == "b4_ma_copies":
                    fn = lambda: b4_ma_copies(
                        x, adapters_subset, adapter_ids, token_ranks,
                        out=out_buf, prepared=prep_b4_c
                    )
                elif impl == "b4_ma_shared":
                    fn = lambda: b4_ma_shared(
                        x, adapters_subset, adapter_ids, token_ranks,
                        out=out_buf, prepared=prep_b4_s
                    )
                else:
                    raise ValueError(f"Unknown impl: {impl}")

                # 2. Correctness check vs fp32 b0_ma
                test_out = fn()
                diff = test_out.float() - ref_out
                max_abs = float(diff.abs().max().item())
                rel_max = max_abs / max(ref_max, 1e-7)
                l2_diff = float(torch.linalg.norm(diff).item())
                rel_l2 = l2_diff / max(ref_l2, 1e-7)

                if cfg_idx == 0:
                    ratio_max = rel_max / max(noise_max, 1e-7)
                    ratio_l2 = rel_l2 / max(noise_l2, 1e-7)
                    print(f"[cfg 0 correctness] {impl:14s}: rel_max={rel_max:.6f} ({ratio_max:.2f}x noise), rel_l2={rel_l2:.6f} ({ratio_l2:.2f}x noise)", flush=True)

                # Pass if max|test-ref| <= 1e-2 * max|ref| AND ||test-ref||_2 / ||ref||_2 <= 5e-3
                if rel_max <= 1e-2 and rel_l2 <= 5e-3:
                    # 3. Timing via CUDA graph replay
                    med_us, p5_us, p95_us = time_cuda_graph(fn, replays=args.replays)
                else:
                    corr_status = "FAIL"
                    status = "BROKEN"
                    err_msg = f"Relative error vs b0_ma exceeded: rel_max={rel_max:.5f} (> 1e-2) or rel_l2={rel_l2:.5f} (> 5e-3)"

            except Exception as exc:
                status = "ERROR"
                err_msg = str(exc)

            row = {
                "config_idx": cfg_idx,
                "phase": phase,
                "tokens": tokens,
                "adapters": adapters_n,
                "tpr": tpr,
                "rank_mix": mix,
                "popularity": pop,
                "impl": impl,
                "status": status,
                "error": err_msg,
                "correctness": corr_status,
                "max_abs_diff": round(max_abs, 6),
                "rel_max_diff": round(rel_max, 6),
                "rel_l2_diff": round(rel_l2, 6),
                "median_us": round(med_us, 2),
                "p5_us": round(p5_us, 2),
                "p95_us": round(p95_us, 2),
                "rows_read": rows_read,
                "ideal_rows_read": ideal_rows,
                "gpu": device_name,
                "torch_version": torch.__version__,
                "triton_version": __import__("triton").__version__,
                "vllm_version": vllm_ver,
                "git_commit": git_commit,
            }
            writer.writerow(row)
            csv_file.flush()

        if (cfg_idx + 1) % 10 == 0 or cfg_idx == 0:
            print(f"[{cfg_idx + 1}/{len(configs)}] {phase} T={tokens} N={adapters_n} {mix} {pop} (elapsed: {time.time() - start_time:.1f}s)", flush=True)

    csv_file.close()
    print(f"Sweep complete in {time.time() - start_time:.1f}s. Results written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
