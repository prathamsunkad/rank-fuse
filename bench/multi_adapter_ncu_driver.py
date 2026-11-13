"""Nsight Compute profiling driver for multi-adapter implementations.

Profiles exactly one final call of the requested implementation with all per-step
metadata precomputed outside the profiler start/stop range.

Supports:
  - rf1_ma
  - rf1_ma_flat
  - b1_ma_padded
  - b3_ma
  - b4_ma_copies
  - b4_ma_shared
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

from rankfuse.baselines.multi import (
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


def generate_ranks(tokens: int, mix: str, r_max: int, rng: random.Random) -> torch.Tensor:
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


def generate_adapter_ids(tokens: int, adapters: int, popularity: str, rng: random.Random) -> torch.Tensor:
    if adapters <= 1:
        return torch.zeros(tokens, dtype=torch.int32)
    if popularity == "zipf":
        weights = [1.0 / (i + 1) ** 1.2 for i in range(adapters)]
    else:
        weights = [1.0] * adapters
    ids = rng.choices(range(adapters), weights=weights, k=tokens)
    return torch.tensor(ids, dtype=torch.int32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("impl", choices=["rf1_ma", "rf1_ma_flat", "b1_ma_padded", "b3_ma", "b4_ma_copies", "b4_ma_shared"])
    p.add_argument("tokens", type=int, nargs="?", default=32)
    p.add_argument("adapters", type=int, nargs="?", default=16)
    p.add_argument("h_in", type=int, nargs="?", default=4096)
    p.add_argument("h_out", type=int, nargs="?", default=4096)
    p.add_argument("r_max", type=int, nargs="?", default=64)
    p.add_argument("mix", nargs="?", default="mostly_low")
    p.add_argument("pop", nargs="?", default="uniform")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("multi_adapter_ncu_driver requires CUDA")

    device_name = torch.cuda.get_device_name()
    if "H200" not in device_name:
        raise RuntimeError(f"Expected H200, got: {device_name}")

    tokens = args.tokens
    adapters_n = args.adapters
    h_in = args.h_in
    h_out = args.h_out
    r_max = args.r_max
    dtype = torch.float16

    rng = random.Random(42)
    torch_gen = torch.Generator(device="cpu").manual_seed(42)

    # Generate adapters (scaled so outputs are O(1))
    adapters = []
    std_a = 1.0 / (h_in ** 0.5)
    std_b = 1.0 / (r_max ** 0.5)
    for _ in range(adapters_n):
        a_t = (torch.randn(r_max, h_in, dtype=dtype, generator=torch_gen) * std_a).cuda()
        b = (torch.randn(r_max, h_out, dtype=dtype, generator=torch_gen) * std_b).cuda()
        adapters.append(MatryoshkaLoRA(a_t, b))

    a_pool = torch.stack([item.A_t for item in adapters]).unsqueeze(0)
    b_pool = torch.stack([item.B for item in adapters]).unsqueeze(0)
    scales = torch.ones(r_max + 1, dtype=torch.float32, device="cuda")

    # Generate batch
    x = torch.randn(tokens, h_in, dtype=dtype, device="cuda", generator=torch.Generator(device="cuda").manual_seed(0))
    token_ranks = generate_ranks(tokens, args.mix, r_max, rng).cuda()
    adapter_ids = generate_adapter_ids(tokens, adapters_n, args.pop, rng).cuda()

    # Sort tokens by adapter_ids
    sort_idx = torch.argsort(adapter_ids)
    x = x[sort_idx].contiguous()
    adapter_ids = adapter_ids[sort_idx].contiguous()
    token_ranks = token_ranks[sort_idx].contiguous()

    out_buf = torch.zeros(tokens, h_out, dtype=dtype, device="cuda")

    # Pre-build per-step metadata
    if args.impl == "rf1_ma":
        num_frames = adapters_n * 4
        page_pool = PagePool(num_frames=num_frames, h_in=h_in, h_out=h_out, dtype=dtype, device="cuda")
        page_mgr = AdapterPageManager(page_pool, page_rank=16, max_adapters=adapters_n, r_max=r_max)
        for idx, item in enumerate(adapters):
            page_mgr.register(idx, item.A_t.cpu(), item.B.cpu())
        page_mgr.ensure_resident(list(range(adapters_n)), r_max)
        page_mgr.wait_for_copies()
        fn = lambda: rf1_ma(
            x, page_pool.frames, page_mgr.page_table, adapter_ids, token_ranks,
            h_in=h_in, h_out=h_out, r_max=r_max, scales=scales, out=out_buf
        )
    elif args.impl == "rf1_ma_flat":
        fn = lambda: rf1_ma_flat(
            x, a_pool, b_pool, adapter_ids, token_ranks,
            scales=scales, out=out_buf
        )
    elif args.impl == "b1_ma_padded":
        prep_b1 = prepare_b1_ma_padded(x, adapters, adapter_ids, token_ranks)
        fn = lambda: b1_ma_padded(
            x, adapters, adapter_ids, token_ranks,
            out=out_buf, prepared=prep_b1
        )
    elif args.impl == "b3_ma":
        prep_b3 = prepare_b3_ma(x, adapters, adapter_ids, token_ranks)
        fn = lambda: b3_ma(
            x, adapters, adapter_ids, token_ranks,
            out=out_buf, prepared=prep_b3
        )
    elif args.impl == "b4_ma_copies":
        prep_b4_c = prepare_b4_ma_copies(x, adapters, adapter_ids, token_ranks)
        fn = lambda: b4_ma_copies(
            x, adapters, adapter_ids, token_ranks,
            out=out_buf, prepared=prep_b4_c
        )
    elif args.impl == "b4_ma_shared":
        prep_b4_s = prepare_b4_ma_shared(x, adapters, token_ranks)
        fn = lambda: b4_ma_shared(
            x, adapters, adapter_ids, token_ranks,
            out=out_buf, prepared=prep_b4_s
        )
    else:
        raise ValueError(f"Unknown impl: {args.impl}")

    # 1. Warm up 2 calls off-profile (compiles Triton / PyTorch kernels)
    for _ in range(2):
        fn()
    torch.cuda.synchronize()

    # 2. Profile only the 3rd (final) call
    torch.cuda.cudart().cudaProfilerStart()
    fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    print(f"Profiled {args.impl}: {tokens} tokens, {adapters_n} adapters, {h_in}x{h_out}, r_max {r_max}, {args.mix}, {args.pop}", flush=True)


if __name__ == "__main__":
    main()
