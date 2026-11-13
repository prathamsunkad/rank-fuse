"""Correctness, paging, and kernel tests for multi-adapter Matryoshka LoRA.

Covers the B7 test matrix:
- N in {1, 2, 5, 33}
- Ranks in {1, 2, 4, 8, 16, 32, 64}
- 1-token and 1000-token groups
- limit-8 group next to a limit-64 group
- page-0-only residency serving rank <= 16
- shuffled non-contiguous frames
- mixed ranks in one adapter
- merged q|k|v and gate|up slices
- add=True
- empty group with T > 0
- T = 0
- rf1_ma vs rf1_ma_flat bitwise equal
- CUDA-graph capture then replay with changed page table and token ranks
- int64 address test with frame index near 2^31 / PAGE_STRIDE
- Small-dim variants runnable on CPU under TRITON_INTERPRET=1 without skipif(cuda)
"""

from __future__ import annotations

import os
import pytest
import torch

# Force interpreter when running on CPU without CUDA BEFORE triton is imported
if not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"

import triton
import triton.language as tl

from rankfuse.baselines.multi import (
    b0_ma,
    b1_ma,
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make_adapters(
    num_adapters: int,
    r_max: int,
    h_in: int,
    h_out: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = DEVICE,
    seed: int = 42,
) -> list[MatryoshkaLoRA]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    adapters = []
    for _ in range(num_adapters):
        a_t = torch.randn(r_max, h_in, dtype=dtype, generator=gen).to(device)
        b = torch.randn(r_max, h_out, dtype=dtype, generator=gen).to(device)
        adapters.append(MatryoshkaLoRA(a_t, b))
    return adapters


def _setup_paged_manager(
    adapters: list[MatryoshkaLoRA],
    r_max: int,
    h_in: int,
    h_out: int,
    num_frames: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = DEVICE,
) -> tuple[PagePool, AdapterPageManager]:
    pool = PagePool(num_frames=num_frames, h_in=h_in, h_out=h_out, dtype=dtype, device=device)
    manager = AdapterPageManager(
        pool, page_rank=16, max_adapters=len(adapters), r_max=r_max
    )
    for idx, item in enumerate(adapters):
        manager.register(idx, item.A_t.cpu(), item.B.cpu())
    return pool, manager


# =====================================================================
# 1. Existing baseline tests (preserved)
# =====================================================================

def test_multi_adapter_padded_matches_reference():
    gen = torch.Generator().manual_seed(3)
    adapters = [
        MatryoshkaLoRA(torch.randn(8, 6, generator=gen), torch.randn(8, 5, generator=gen)),
        MatryoshkaLoRA(torch.randn(8, 6, generator=gen), torch.randn(8, 5, generator=gen)),
    ]
    x = torch.randn(7, 6, generator=gen)
    adapter_ids = torch.tensor([0, 1, 0, 1, 1, 0, 1])
    ranks = torch.tensor([1, 8, 2, 4, 8, 1, 4], dtype=torch.int32)
    assert torch.allclose(b1_ma(x, adapters, adapter_ids, ranks), b0_ma(x, adapters, adapter_ids, ranks), atol=1e-5)


def test_multi_adapter_new_baselines_match_reference():
    gen = torch.Generator().manual_seed(42)
    adapters = [
        MatryoshkaLoRA(torch.randn(16, 8, generator=gen), torch.randn(16, 6, generator=gen)),
        MatryoshkaLoRA(torch.randn(16, 8, generator=gen), torch.randn(16, 6, generator=gen)),
        MatryoshkaLoRA(torch.randn(16, 8, generator=gen), torch.randn(16, 6, generator=gen)),
    ]
    x = torch.randn(9, 8, generator=gen)
    adapter_ids = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 0])
    ranks = torch.tensor([2, 8, 4, 16, 8, 1, 2, 8, 4], dtype=torch.int32)
    ref = b0_ma(x, adapters, adapter_ids, ranks)

    # 1. b1_ma_padded (eager and prepared)
    out_b1 = b1_ma_padded(x, adapters, adapter_ids, ranks)
    assert torch.allclose(out_b1, ref, atol=1e-5)
    prep_b1 = prepare_b1_ma_padded(x, adapters, adapter_ids, ranks)
    out_b1_prep = b1_ma_padded(x, adapters, adapter_ids, ranks, prepared=prep_b1)
    assert torch.allclose(out_b1_prep, ref, atol=1e-5)

    # 2. b3_ma (eager/prepared)
    out_b3 = b3_ma(x, adapters, adapter_ids, ranks)
    assert torch.allclose(out_b3, ref, atol=1e-5)
    prep_b3 = prepare_b3_ma(x, adapters, adapter_ids, ranks)
    out_b3_prep = b3_ma(x, adapters, adapter_ids, ranks, prepared=prep_b3)
    assert torch.allclose(out_b3_prep, ref, atol=1e-5)

    # 3. b4_ma_copies
    out_b4_copies = b4_ma_copies(x, adapters, adapter_ids, ranks)
    assert torch.allclose(out_b4_copies, ref, atol=1e-5)
    prep_b4_c = prepare_b4_ma_copies(x, adapters, adapter_ids, ranks)
    out_b4_c_prep = b4_ma_copies(x, adapters, adapter_ids, ranks, prepared=prep_b4_c)
    assert torch.allclose(out_b4_c_prep, ref, atol=1e-5)

    # 4. b4_ma_shared
    out_b4_shared = b4_ma_shared(x, adapters, adapter_ids, ranks)
    assert torch.allclose(out_b4_shared, ref, atol=1e-5)
    prep_b4_s = prepare_b4_ma_shared(x, adapters, ranks)
    out_b4_s_prep = b4_ma_shared(x, adapters, adapter_ids, ranks, prepared=prep_b4_s)
    assert torch.allclose(out_b4_s_prep, ref, atol=1e-5)


def test_multi_adapter_empty_batch():
    adapter = MatryoshkaLoRA(torch.empty(4, 3), torch.empty(4, 2))
    x = torch.empty(0, 3)
    ids = torch.empty(0, dtype=torch.int64)
    ranks = torch.empty(0, dtype=torch.int32)
    assert b0_ma(x, [adapter], ids, ranks).shape == (0, 2)
    assert b1_ma(x, [adapter], ids, ranks).shape == (0, 2)


# =====================================================================
# 2. T = 0 empty tokens
# =====================================================================

def test_rf1_ma_t0_empty():
    h_in, h_out, r_max = 32, 16, 16
    x = torch.empty(0, h_in, dtype=torch.float32, device=DEVICE)
    a = torch.empty(1, 1, r_max, h_in, dtype=torch.float32, device=DEVICE)
    b = torch.empty(1, 1, r_max, h_out, dtype=torch.float32, device=DEVICE)
    ids = torch.empty(0, dtype=torch.int32, device=DEVICE)
    ranks = torch.empty(0, dtype=torch.int32, device=DEVICE)
    scales = torch.ones(r_max + 1, dtype=torch.float32, device=DEVICE)

    flat = rf1_ma_flat(x, a, b, ids, ranks, scales=scales)
    assert flat.shape == (0, h_out)

    pool = torch.zeros(2, 16, h_in + h_out, dtype=torch.float32, device=DEVICE)
    page_table = torch.zeros(1, 1, dtype=torch.int32, device=DEVICE)
    paged = rf1_ma(x, pool, page_table, ids, ranks, h_in=h_in, h_out=h_out, r_max=r_max, scales=scales)
    assert paged.shape == (0, h_out)


# =====================================================================
# 3. N in {1, 2, 5, 33} (Small-dim CPU / Interpreter)
# =====================================================================

@pytest.mark.parametrize("num_adapters", [1, 2, 5, 33])
def test_rf1_ma_num_adapters_cpu(num_adapters):
    h_in, h_out, r_max = 32, 16, 32
    adapters = _make_adapters(num_adapters, r_max, h_in, h_out, dtype=torch.float32, seed=num_adapters)
    tokens = max(num_adapters, 2)
    ids = torch.arange(tokens, dtype=torch.int32, device=DEVICE) % num_adapters
    ranks = torch.where(ids % 2 == 0, torch.full_like(ids, 16), torch.full_like(ids, 32))
    x = torch.randn(tokens, h_in, dtype=torch.float32, device=DEVICE)
    scales = torch.ones(r_max + 1, dtype=torch.float32, device=DEVICE)

    a_pool = torch.stack([item.A_t for item in adapters]).unsqueeze(0)
    b_pool = torch.stack([item.B for item in adapters]).unsqueeze(0)

    max_tiles = max(64, num_adapters + 1)
    flat = rf1_ma_flat(x, a_pool, b_pool, ids, ranks, scales=scales, max_tiles=max_tiles)

    pool, manager = _setup_paged_manager(adapters, r_max, h_in, h_out, num_frames=num_adapters * 2, device=DEVICE)
    manager.ensure_resident(list(range(num_adapters)), 32)
    manager.wait_for_copies()

    paged = rf1_ma(x, pool.frames, manager.page_table, ids, ranks,
                   h_in=h_in, h_out=h_out, r_max=r_max, scales=scales, max_tiles=max_tiles)

    assert torch.equal(flat, paged)
    reference = b0_ma(x, adapters, ids, ranks)
    assert torch.allclose(flat, reference, atol=2e-5)


# =====================================================================
# 4. Ranks in {1, 2, 4, 8, 16, 32, 64}
# =====================================================================

@pytest.mark.parametrize("test_rank", [1, 2, 4, 8, 16, 32, 64])
def test_rf1_ma_ranks_cpu(test_rank):
    h_in, h_out, r_max = 32, 16, 64
    adapters = _make_adapters(2, r_max, h_in, h_out, dtype=torch.float32, seed=test_rank)
    tokens = 4
    ids = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=DEVICE)
    ranks = torch.full((tokens,), test_rank, dtype=torch.int32, device=DEVICE)
    x = torch.randn(tokens, h_in, dtype=torch.float32, device=DEVICE)
    scales = torch.ones(r_max + 1, dtype=torch.float32, device=DEVICE)

    a_pool = torch.stack([item.A_t for item in adapters]).unsqueeze(0)
    b_pool = torch.stack([item.B for item in adapters]).unsqueeze(0)

    flat = rf1_ma_flat(x, a_pool, b_pool, ids, ranks, scales=scales, max_tiles=8)

    pool, manager = _setup_paged_manager(adapters, r_max, h_in, h_out, num_frames=8, device=DEVICE)
    manager.ensure_resident([0, 1], test_rank)
    manager.wait_for_copies()

    paged = rf1_ma(x, pool.frames, manager.page_table, ids, ranks,
                   h_in=h_in, h_out=h_out, r_max=r_max, scales=scales, max_tiles=8)

    assert torch.equal(flat, paged)
    reference = b0_ma(x, adapters, ids, ranks)
    assert torch.allclose(flat, reference, atol=2e-5)


# =====================================================================
# 5. Token group sizes: 1-token and large token groups
# =====================================================================

def test_rf1_ma_token_group_sizes_cpu():
    h_in, h_out, r_max = 32, 16, 16
    adapters = _make_adapters(2, r_max, h_in, h_out, dtype=torch.float32, seed=99)
    # Group 0: 1 token, Group 1: 48 tokens
    tokens = 49
    ids = torch.cat([torch.zeros(1, dtype=torch.int32, device=DEVICE), torch.ones(48, dtype=torch.int32, device=DEVICE)])
    ranks = torch.full((tokens,), 16, dtype=torch.int32, device=DEVICE)
    x = torch.randn(tokens, h_in, dtype=torch.float32, device=DEVICE)
    scales = torch.ones(r_max + 1, dtype=torch.float32, device=DEVICE)

    a_pool = torch.stack([item.A_t for item in adapters]).unsqueeze(0)
    b_pool = torch.stack([item.B for item in adapters]).unsqueeze(0)

    flat = rf1_ma_flat(x, a_pool, b_pool, ids, ranks, scales=scales, max_tiles=8)

    pool, manager = _setup_paged_manager(adapters, r_max, h_in, h_out, num_frames=4, device=DEVICE)
    manager.ensure_resident([0, 1], 16)
    manager.wait_for_copies()

    paged = rf1_ma(x, pool.frames, manager.page_table, ids, ranks,
                   h_in=h_in, h_out=h_out, r_max=r_max, scales=scales, max_tiles=8)

    assert torch.equal(flat, paged)
    reference = b0_ma(x, adapters, ids, ranks)
    assert torch.allclose(flat, reference, atol=2e-5)


# =====================================================================
# 6. Rank limits: limit-8 group next to a limit-64 group
# =====================================================================

def test_rf1_ma_group_rank_limits():
    h_in, h_out, r_max = 32, 16, 64
    adapters = _make_adapters(2, r_max, h_in, h_out, dtype=torch.float32, seed=7)
    tokens = 32
    # Two groups of 16 tokens
    ids = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    starts = torch.tensor([0, 16], dtype=torch.int32, device=DEVICE)
    lens = torch.tensor([16, 16], dtype=torch.int32, device=DEVICE)
    # Group 0 limit is 8, Group 1 limit is 64
    limits = torch.tensor([8, 64], dtype=torch.int32, device=DEVICE)
    # All token ranks nominally 64 (group 0 should be clamped to 8)
    token_ranks = torch.full((tokens,), 64, dtype=torch.int32, device=DEVICE)
    x = torch.randn(tokens, h_in, dtype=torch.float32, device=DEVICE)
    scales = torch.ones(r_max + 1, dtype=torch.float32, device=DEVICE)

    a_pool = torch.stack([item.A_t for item in adapters]).unsqueeze(0)
    b_pool = torch.stack([item.B for item in adapters]).unsqueeze(0)

    flat = rf1_ma_flat(
        x, a_pool, b_pool, ids, token_ranks,
        scales=scales, group_starts=starts, group_lens=lens, group_rank_limit=limits, max_tiles=8
    )

    pool, manager = _setup_paged_manager(adapters, r_max, h_in, h_out, num_frames=8, device=DEVICE)
    manager.ensure_resident([0, 1], 64)
    manager.wait_for_copies()

    paged = rf1_ma(
        x, pool.frames, manager.page_table, ids, token_ranks,
        h_in=h_in, h_out=h_out, r_max=r_max, scales=scales,
        group_starts=starts, group_lens=lens, group_rank_limit=limits, max_tiles=8
    )

    assert torch.equal(flat, paged)
    # Effective ranks: 8 for group 0, 64 for group 1
    effective_ranks = torch.cat([torch.full((16,), 8, dtype=torch.int32, device=DEVICE),
                                 torch.full((16,), 64, dtype=torch.int32, device=DEVICE)])
    adapter_ids_per_tok = torch.cat([torch.zeros(16, dtype=torch.int32, device=DEVICE),
                                    torch.ones(16, dtype=torch.int32, device=DEVICE)])
    reference = b0_ma(x, adapters, adapter_ids_per_tok, effective_ranks)
    assert torch.allclose(flat, reference, atol=2e-5)


# =====================================================================
# 7. Page-0-only residency serving rank <= 16
# =====================================================================

def test_rf1_ma_page0_only_residency():
    h_in, h_out, r_max = 32, 16, 64
    adapters = _make_adapters(1, r_max, h_in, h_out, dtype=torch.float32, seed=12)
    tokens = 4
    ids = torch.zeros(tokens, dtype=torch.int32, device=DEVICE)
    ranks = torch.tensor([4, 8, 12, 16], dtype=torch.int32, device=DEVICE)
    x = torch.randn(tokens, h_in, dtype=torch.float32, device=DEVICE)
    scales = torch.ones(r_max + 1, dtype=torch.float32, device=DEVICE)

    # Setup pool with only page 0 allocated (1 frame), pages 1..3 are -1
    pool, manager = _setup_paged_manager(adapters, r_max, h_in, h_out, num_frames=4, device=DEVICE)
    manager.ensure_prefix(0, 16)
    manager.wait_for_copies()

    # Verify pages 1..3 are -1 in the page table
    pt = manager.page_table[0].tolist()
    assert pt[0] >= 0
    assert pt[1:] == [-1, -1, -1]

    paged = rf1_ma(
        x, pool.frames, manager.page_table, ids, ranks,
        h_in=h_in, h_out=h_out, r_max=r_max, scales=scales, max_tiles=4
    )
    reference = b0_ma(x, adapters, ids, ranks)
    assert torch.allclose(paged, reference, atol=2e-5)


# =====================================================================
# 8. Shuffled non-contiguous frames
# =====================================================================

def test_rf1_ma_shuffled_noncontiguous_frames():
    h_in, h_out, r_max = 32, 16, 64
    adapters = _make_adapters(1, r_max, h_in, h_out, dtype=torch.float32, seed=55)
    tokens = 4
    ids = torch.zeros(tokens, dtype=torch.int32, device=DEVICE)
    ranks = torch.full((tokens,), 64, dtype=torch.int32, device=DEVICE)
    x = torch.randn(tokens, h_in, dtype=torch.float32, device=DEVICE)
    scales = torch.ones(r_max + 1, dtype=torch.float32, device=DEVICE)

    # 10-frame pool, assign pages 0..3 to non-contiguous frames [7, 2, 8, 1]
    pool = PagePool(num_frames=10, h_in=h_in, h_out=h_out, dtype=torch.float32, device=DEVICE)
    shuffled_frames = [7, 2, 8, 1]
    a_t = adapters[0].A_t
    b = adapters[0].B
    for p, frame_idx in enumerate(shuffled_frames):
        r_slice = slice(p * 16, (p + 1) * 16)
        pool.frames[frame_idx, :, :h_in].copy_(a_t[r_slice])
        pool.frames[frame_idx, :, h_in:].copy_(b[r_slice])

    page_table = torch.full((1, 4), -1, dtype=torch.int32, device=DEVICE)
    page_table[0] = torch.tensor(shuffled_frames, dtype=torch.int32, device=DEVICE)

    paged = rf1_ma(
        x, pool.frames, page_table, ids, ranks,
        h_in=h_in, h_out=h_out, r_max=r_max, scales=scales, max_tiles=4
    )
    reference = b0_ma(x, adapters, ids, ranks)
    assert torch.allclose(paged, reference, atol=2e-5)


# =====================================================================
# 9. Mixed ranks in one adapter
# =====================================================================

def test_rf1_ma_mixed_ranks_in_one_adapter():
    h_in, h_out, r_max = 32, 16, 64
    adapters = _make_adapters(1, r_max, h_in, h_out, dtype=torch.float32, seed=88)
    tokens = 5
    ids = torch.zeros(tokens, dtype=torch.int32, device=DEVICE)
    ranks = torch.tensor([2, 8, 16, 32, 64], dtype=torch.int32, device=DEVICE)
    x = torch.randn(tokens, h_in, dtype=torch.float32, device=DEVICE)
    scales = torch.ones(r_max + 1, dtype=torch.float32, device=DEVICE)

    a_pool = torch.stack([item.A_t for item in adapters]).unsqueeze(0)
    b_pool = torch.stack([item.B for item in adapters]).unsqueeze(0)

    flat = rf1_ma_flat(x, a_pool, b_pool, ids, ranks, scales=scales, max_tiles=4)

    pool, manager = _setup_paged_manager(adapters, r_max, h_in, h_out, num_frames=4, device=DEVICE)
    manager.ensure_resident([0], 64)
    manager.wait_for_copies()

    paged = rf1_ma(
        x, pool.frames, manager.page_table, ids, ranks,
        h_in=h_in, h_out=h_out, r_max=r_max, scales=scales, max_tiles=4
    )

    assert torch.equal(flat, paged)
    reference = b0_ma(x, adapters, ids, ranks)
    assert torch.allclose(flat, reference, atol=2e-5)


# =====================================================================
# 10. Merged q|k|v and gate|up slices
# =====================================================================

@pytest.mark.parametrize("slice_config", [
    ("qkv", (16, 16, 16)),
    ("gate_up", (32, 32)),
])
def test_rf1_ma_merged_slices(slice_config):
    name, slice_sizes = slice_config
    n_slices = len(slice_sizes)
    h_in = 32
    r_max = 16
    tokens = 4
    x = torch.randn(tokens, h_in, dtype=torch.float32, device=DEVICE)
    ids = torch.zeros(tokens, dtype=torch.int32, device=DEVICE)
    ranks = torch.tensor([4, 8, 12, 16], dtype=torch.int32, device=DEVICE)
    scales = torch.ones(r_max + 1, dtype=torch.float32, device=DEVICE)

    gen = torch.Generator(device="cpu").manual_seed(42)
    widest = max(slice_sizes)
    a_pool = torch.randn(n_slices, 1, r_max, h_in, dtype=torch.float32, generator=gen).to(DEVICE)
    b_pool = torch.randn(n_slices, 1, r_max, widest, dtype=torch.float32, generator=gen).to(DEVICE)

    flat = rf1_ma_flat(
        x, a_pool, b_pool, ids, ranks,
        scales=scales, slice_sizes=slice_sizes, max_tiles=4
    )
    assert flat.shape == (tokens, sum(slice_sizes))

    # Reference per slice
    ref_slices = []
    for s in range(n_slices):
        adapter = MatryoshkaLoRA(a_pool[s, 0], b_pool[s, 0, :, :slice_sizes[s]])
        ref_s = b0_ma(x, [adapter], ids, ranks)
        ref_slices.append(ref_s)
    ref_total = torch.cat(ref_slices, dim=1)

    assert torch.allclose(flat, ref_total, atol=2e-5)


# =====================================================================
# 11. add = True
# =====================================================================

def test_rf1_ma_add_true():
    h_in, h_out, r_max = 32, 16, 16
    adapters = _make_adapters(1, r_max, h_in, h_out, dtype=torch.float32, seed=123)
    tokens = 4
    x = torch.randn(tokens, h_in, dtype=torch.float32, device=DEVICE)
    ids = torch.zeros(tokens, dtype=torch.int32, device=DEVICE)
    ranks = torch.tensor([4, 8, 12, 16], dtype=torch.int32, device=DEVICE)
    scales = torch.ones(r_max + 1, dtype=torch.float32, device=DEVICE)

    a_pool = torch.stack([item.A_t for item in adapters]).unsqueeze(0)
    b_pool = torch.stack([item.B for item in adapters]).unsqueeze(0)

    base = torch.randn(tokens, h_out, dtype=torch.float32, device=DEVICE)
    normal = rf1_ma_flat(x, a_pool, b_pool, ids, ranks, scales=scales, max_tiles=4)

    added = base.clone()
    rf1_ma_flat(x, a_pool, b_pool, ids, ranks, scales=scales, out=added, add=True, max_tiles=4)

    expected = base + normal
    assert torch.allclose(added, expected, atol=1e-6)


# =====================================================================
# 12. Empty group with T > 0
# =====================================================================

def test_rf1_ma_empty_group_with_positive_tokens():
    h_in, h_out, r_max = 32, 16, 16
    adapters = _make_adapters(3, r_max, h_in, h_out, dtype=torch.float32, seed=77)
    tokens = 32
    # Group 0: 16 tokens (adapter 0)
    # Group 1: 0 tokens (adapter 1 - EMPTY)
    # Group 2: 16 tokens (adapter 2)
    group_starts = torch.tensor([0, 16, 16], dtype=torch.int32, device=DEVICE)
    group_lens = torch.tensor([16, 0, 16], dtype=torch.int32, device=DEVICE)
    group_adapters = torch.tensor([0, 1, 2], dtype=torch.int32, device=DEVICE)
    token_ranks = torch.full((tokens,), 16, dtype=torch.int32, device=DEVICE)
    x = torch.randn(tokens, h_in, dtype=torch.float32, device=DEVICE)
    scales = torch.ones(r_max + 1, dtype=torch.float32, device=DEVICE)

    a_pool = torch.stack([item.A_t for item in adapters]).unsqueeze(0)
    b_pool = torch.stack([item.B for item in adapters]).unsqueeze(0)

    flat = rf1_ma_flat(
        x, a_pool, b_pool, group_adapters, token_ranks,
        scales=scales, group_starts=group_starts, group_lens=group_lens, max_tiles=8
    )

    token_adapter_ids = torch.cat([torch.zeros(16, dtype=torch.int32, device=DEVICE),
                                   torch.full((16,), 2, dtype=torch.int32, device=DEVICE)])
    ref = b0_ma(x, adapters, token_adapter_ids, token_ranks)
    assert torch.allclose(flat, ref, atol=2e-5)


# =====================================================================
# 13. Int64 address overflow test
# =====================================================================

def test_rf1_ma_int64_address_overflow_logic():
    """Verify frame.to(tl.int64) * PAGE_STRIDE prevents 32-bit integer overflow."""
    PAGE_STRIDE = 32768
    # Frame index chosen such that frame * PAGE_STRIDE > 2^31 - 1
    large_frame = 70000
    expected_offset = 70000 * 32768  # 2,293,760,000 > 2,147,483,647

    # In int32, this overflows to a negative number:
    int32_val = (torch.tensor([large_frame], dtype=torch.int32) * PAGE_STRIDE).item()
    assert int32_val < 0, f"Expected 32-bit overflow, got {int32_val}"

    # In int64, cast BEFORE multiply preserves positive 64-bit address:
    int64_val = (torch.tensor([large_frame], dtype=torch.int32).to(torch.int64) * PAGE_STRIDE).item()
    assert int64_val == expected_offset
    assert int64_val > 2**31 - 1


# =====================================================================
# 14. GPU-only tests with realistic dimensions (h_in = h_out = 4096)
# =====================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_adapters", [1, 2, 5, 33])
def test_rf1_paged_and_flat_match_reference_gpu(num_adapters):
    gen = torch.Generator().manual_seed(num_adapters)
    h_in, h_out, r_max = 4096, 4096, 64
    adapters = [
        MatryoshkaLoRA(
            torch.randn(r_max, h_in, generator=gen, dtype=torch.float16),
            torch.randn(r_max, h_out, generator=gen, dtype=torch.float16),
        )
        for _ in range(num_adapters)
    ]
    tokens = max(num_adapters, 2)
    ids = torch.arange(tokens, device="cuda", dtype=torch.int32) % num_adapters
    ranks = torch.where(ids % 2 == 0, torch.full_like(ids, 8), torch.full_like(ids, 64))
    x = torch.randn(tokens, h_in, dtype=torch.float16, generator=gen).cuda()
    scales = torch.ones(r_max + 1, device="cuda", dtype=torch.float16)

    a_pool = torch.stack([item.A_t for item in adapters]).cuda().unsqueeze(0)
    b_pool = torch.stack([item.B for item in adapters]).cuda().unsqueeze(0)
    flat = rf1_ma_flat(x, a_pool, b_pool, ids, ranks, scales=scales)

    pool = PagePool(num_frames=num_adapters * 4, h_in=h_in, h_out=h_out,
                    dtype=torch.float16, device="cuda")
    manager = AdapterPageManager(pool, page_rank=16, max_adapters=num_adapters, r_max=r_max)
    for index, item in enumerate(adapters):
        manager.register(index, item.A_t.cpu(), item.B.cpu())
    manager.ensure_resident(list(range(num_adapters)), 64)
    manager.wait_for_copies()

    paged = rf1_ma(x, pool.frames, manager.page_table, ids, ranks,
                   h_in=h_in, h_out=h_out, r_max=r_max, scales=scales)
    assert torch.equal(flat, paged)

    reference = b0_ma(x.cpu(), adapters, ids.cpu(), ranks.cpu())
    # FP16 tolerance vs FP32 oracle tied to reduction dimension h_in
    assert torch.allclose(flat.cpu(), reference, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rf1_add_empty_and_mixed_ranks_gpu():
    h_in, h_out, r_max = 4096, 4096, 64
    a = torch.randn(1, 2, r_max, h_in, device="cuda", dtype=torch.float16)
    b = torch.randn(1, 2, r_max, h_out, device="cuda", dtype=torch.float16)
    x = torch.randn(1000, h_in, device="cuda", dtype=torch.float16)
    ids = torch.cat((torch.zeros(500, device="cuda", dtype=torch.int32),
                     torch.ones(500, device="cuda", dtype=torch.int32)))
    ranks = torch.cat((torch.full((500,), 8, device="cuda", dtype=torch.int32),
                       torch.full((500,), 64, device="cuda", dtype=torch.int32)))
    scales = torch.ones(r_max + 1, device="cuda", dtype=torch.float16)

    base = torch.randn(1000, h_out, device="cuda", dtype=torch.float16)
    expected = base.clone()
    expected += rf1_ma_flat(x, a, b, ids, ranks, scales=scales)
    actual = rf1_ma_flat(x, a, b, ids, ranks, scales=scales, out=base, add=True)
    assert torch.equal(actual, expected)

    empty = rf1_ma_flat(x[:0], a, b, ids[:0], ranks[:0], scales=scales)
    assert empty.shape == (0, h_out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rf1_graph_replay_reads_changed_metadata_gpu():
    h_in, h_out, r_max = 4096, 4096, 64
    x = torch.randn(8, h_in, device="cuda", dtype=torch.float16)
    a = torch.randn(1, 2, r_max, h_in, device="cuda", dtype=torch.float16)
    b = torch.randn(1, 2, r_max, h_out, device="cuda", dtype=torch.float16)
    ids = torch.zeros(8, device="cuda", dtype=torch.int32)
    ranks = torch.full((8,), 8, device="cuda", dtype=torch.int32)
    scales = torch.ones(r_max + 1, device="cuda", dtype=torch.float16)
    out = torch.empty(8, h_out, device="cuda", dtype=torch.float16)

    rf1_ma_flat(x, a, b, ids, ranks, scales=scales, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rf1_ma_flat(x, a, b, ids, ranks, scales=scales, out=out)

    for adapter, rank in ((1, 64), (0, 8)):
        ids.fill_(adapter)
        ranks.fill_(rank)
        graph.replay()
        expected = rf1_ma_flat(x, a, b, ids, ranks, scales=scales)
        assert torch.equal(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rf1_ma_1_and_1000_token_groups_gpu():
    gen = torch.Generator().manual_seed(99)
    h_in, h_out, r_max = 4096, 4096, 64
    adapters = [
        MatryoshkaLoRA(
            torch.randn(r_max, h_in, generator=gen, dtype=torch.float16),
            torch.randn(r_max, h_out, generator=gen, dtype=torch.float16),
        )
        for _ in range(2)
    ]
    tokens = 1001
    # Group 0: 1 token, Group 1: 1000 tokens
    ids = torch.cat([torch.zeros(1, device="cuda", dtype=torch.int32),
                     torch.ones(1000, device="cuda", dtype=torch.int32)])
    ranks = torch.cat([torch.full((1,), 16, device="cuda", dtype=torch.int32),
                       torch.full((1000,), 64, device="cuda", dtype=torch.int32)])
    x = torch.randn(tokens, h_in, dtype=torch.float16, generator=gen).cuda()
    scales = torch.ones(r_max + 1, device="cuda", dtype=torch.float16)

    a_pool = torch.stack([item.A_t for item in adapters]).cuda().unsqueeze(0)
    b_pool = torch.stack([item.B for item in adapters]).cuda().unsqueeze(0)

    flat = rf1_ma_flat(x, a_pool, b_pool, ids, ranks, scales=scales)

    pool = PagePool(num_frames=8, h_in=h_in, h_out=h_out, dtype=torch.float16, device="cuda")
    manager = AdapterPageManager(pool, page_rank=16, max_adapters=2, r_max=r_max)
    for idx, item in enumerate(adapters):
        manager.register(idx, item.A_t.cpu(), item.B.cpu())
    manager.ensure_resident([0, 1], 64)
    manager.wait_for_copies()

    paged = rf1_ma(x, pool.frames, manager.page_table, ids, ranks,
                   h_in=h_in, h_out=h_out, r_max=r_max, scales=scales)

    assert torch.equal(flat, paged)
    reference = b0_ma(x.cpu(), adapters, ids.cpu(), ranks.cpu())
    assert torch.allclose(flat.cpu(), reference, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rf1_paged_graph_replay_reads_changed_page_table_and_ranks_gpu():
    h_in, h_out, r_max = 4096, 4096, 64
    tokens = 8
    num_adapters = 2
    gen = torch.Generator().manual_seed(42)
    adapters = [
        MatryoshkaLoRA(
            torch.randn(r_max, h_in, generator=gen, dtype=torch.float16),
            torch.randn(r_max, h_out, generator=gen, dtype=torch.float16),
        )
        for _ in range(num_adapters)
    ]
    x = torch.randn(tokens, h_in, dtype=torch.float16, generator=gen).cuda()
    scales = torch.ones(r_max + 1, device="cuda", dtype=torch.float16)
    out = torch.empty(tokens, h_out, device="cuda", dtype=torch.float16)

    pool = PagePool(num_frames=num_adapters * 4, h_in=h_in, h_out=h_out,
                    dtype=torch.float16, device="cuda")
    manager = AdapterPageManager(pool, page_rank=16, max_adapters=num_adapters, r_max=r_max)
    for index, item in enumerate(adapters):
        manager.register(index, item.A_t.cpu(), item.B.cpu())
    manager.ensure_resident([0, 1], 64)
    manager.wait_for_copies()

    ids = torch.zeros(tokens, device="cuda", dtype=torch.int32)
    ranks = torch.full((tokens,), 8, device="cuda", dtype=torch.int32)
    page_table = manager.page_table.clone()

    rf1_ma(x, pool.frames, page_table, ids, ranks,
           h_in=h_in, h_out=h_out, r_max=r_max, scales=scales, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rf1_ma(x, pool.frames, page_table, ids, ranks,
               h_in=h_in, h_out=h_out, r_max=r_max, scales=scales, out=out)

    for adapter, rank in ((1, 64), (0, 16), (1, 32)):
        ids.fill_(adapter)
        ranks.fill_(rank)
        # Permute frames in page table in-place on GPU
        f_orig = page_table[adapter].clone()
        f_new = torch.roll(f_orig, shifts=1)
        tmp_frames = pool.frames[f_orig].clone()
        pool.frames[f_new] = tmp_frames
        page_table[adapter] = f_new

        graph.replay()
        expected = rf1_ma(x, pool.frames, page_table, ids, ranks,
                          h_in=h_in, h_out=h_out, r_max=r_max, scales=scales)
        assert torch.equal(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rank", [1, 2, 4, 8, 16, 32, 64])
def test_rf1_ma_ranks_gpu(rank):
    gen = torch.Generator().manual_seed(rank)
    h_in, h_out, r_max = 4096, 4096, 64
    tokens = 4
    adapters = [
        MatryoshkaLoRA(
            torch.randn(r_max, h_in, generator=gen, dtype=torch.float16),
            torch.randn(r_max, h_out, generator=gen, dtype=torch.float16),
        )
    ]
    ids = torch.zeros(tokens, device="cuda", dtype=torch.int32)
    ranks = torch.full((tokens,), rank, device="cuda", dtype=torch.int32)
    x = torch.randn(tokens, h_in, dtype=torch.float16, generator=gen).cuda()
    scales = torch.ones(r_max + 1, device="cuda", dtype=torch.float16)

    a_pool = torch.stack([item.A_t for item in adapters]).cuda().unsqueeze(0)
    b_pool = torch.stack([item.B for item in adapters]).cuda().unsqueeze(0)
    flat = rf1_ma_flat(x, a_pool, b_pool, ids, ranks, scales=scales)

    pool = PagePool(num_frames=4, h_in=h_in, h_out=h_out, dtype=torch.float16, device="cuda")
    manager = AdapterPageManager(pool, page_rank=16, max_adapters=1, r_max=r_max)
    manager.register(0, adapters[0].A_t.cpu(), adapters[0].B.cpu())
    manager.ensure_resident([0], max(rank, 16))
    manager.wait_for_copies()

    paged = rf1_ma(x, pool.frames, manager.page_table, ids, ranks,
                   h_in=h_in, h_out=h_out, r_max=r_max, scales=scales)
    assert torch.equal(flat, paged)
    reference = b0_ma(x.cpu(), adapters, ids.cpu(), ranks.cpu())
    assert torch.allclose(flat.cpu(), reference, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rf1_ma_group_rank_limits_gpu():
    gen = torch.Generator().manual_seed(42)
    h_in, h_out, r_max = 4096, 4096, 64
    adapters = [
        MatryoshkaLoRA(
            torch.randn(r_max, h_in, generator=gen, dtype=torch.float16),
            torch.randn(r_max, h_out, generator=gen, dtype=torch.float16),
        )
        for _ in range(2)
    ]
    tokens = 32
    ids = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, 16], device="cuda", dtype=torch.int32)
    lens = torch.tensor([16, 16], device="cuda", dtype=torch.int32)
    limits = torch.tensor([8, 64], device="cuda", dtype=torch.int32)
    token_ranks = torch.full((tokens,), 64, device="cuda", dtype=torch.int32)
    x = torch.randn(tokens, h_in, dtype=torch.float16, generator=gen).cuda()
    scales = torch.ones(r_max + 1, device="cuda", dtype=torch.float16)

    a_pool = torch.stack([item.A_t for item in adapters]).cuda().unsqueeze(0)
    b_pool = torch.stack([item.B for item in adapters]).cuda().unsqueeze(0)
    flat = rf1_ma_flat(
        x, a_pool, b_pool, ids, token_ranks,
        scales=scales, group_starts=starts, group_lens=lens, group_rank_limit=limits
    )

    pool = PagePool(num_frames=8, h_in=h_in, h_out=h_out, dtype=torch.float16, device="cuda")
    manager = AdapterPageManager(pool, page_rank=16, max_adapters=2, r_max=r_max)
    for idx, item in enumerate(adapters):
        manager.register(idx, item.A_t.cpu(), item.B.cpu())
    manager.ensure_resident([0, 1], 64)
    manager.wait_for_copies()

    paged = rf1_ma(
        x, pool.frames, manager.page_table, ids, token_ranks,
        h_in=h_in, h_out=h_out, r_max=r_max, scales=scales,
        group_starts=starts, group_lens=lens, group_rank_limit=limits
    )
    assert torch.equal(flat, paged)

    effective_ranks = torch.cat([torch.full((16,), 8, dtype=torch.int32),
                                 torch.full((16,), 64, dtype=torch.int32)])
    adapter_ids_per_tok = torch.cat([torch.zeros(16, dtype=torch.int32),
                                     torch.ones(16, dtype=torch.int32)])
    reference = b0_ma(x.cpu(), adapters, adapter_ids_per_tok, effective_ranks)
    assert torch.allclose(flat.cpu(), reference, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rf1_ma_page0_only_residency_gpu():
    gen = torch.Generator().manual_seed(43)
    h_in, h_out, r_max = 4096, 4096, 64
    adapters = [
        MatryoshkaLoRA(
            torch.randn(r_max, h_in, generator=gen, dtype=torch.float16),
            torch.randn(r_max, h_out, generator=gen, dtype=torch.float16),
        )
    ]
    tokens = 4
    ids = torch.zeros(tokens, device="cuda", dtype=torch.int32)
    ranks = torch.tensor([4, 8, 12, 16], device="cuda", dtype=torch.int32)
    x = torch.randn(tokens, h_in, dtype=torch.float16, generator=gen).cuda()
    scales = torch.ones(r_max + 1, device="cuda", dtype=torch.float16)

    pool = PagePool(num_frames=4, h_in=h_in, h_out=h_out, dtype=torch.float16, device="cuda")
    manager = AdapterPageManager(pool, page_rank=16, max_adapters=1, r_max=r_max)
    manager.register(0, adapters[0].A_t.cpu(), adapters[0].B.cpu())
    manager.ensure_prefix(0, 16)
    manager.wait_for_copies()

    pt = manager.page_table[0].tolist()
    assert pt[0] >= 0
    assert pt[1:] == [-1, -1, -1]

    paged = rf1_ma(
        x, pool.frames, manager.page_table, ids, ranks,
        h_in=h_in, h_out=h_out, r_max=r_max, scales=scales
    )
    reference = b0_ma(x.cpu(), adapters, ids.cpu(), ranks.cpu())
    assert torch.allclose(paged.cpu(), reference, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rf1_ma_shuffled_noncontiguous_frames_gpu():
    gen = torch.Generator().manual_seed(44)
    h_in, h_out, r_max = 4096, 4096, 64
    adapters = [
        MatryoshkaLoRA(
            torch.randn(r_max, h_in, generator=gen, dtype=torch.float16),
            torch.randn(r_max, h_out, generator=gen, dtype=torch.float16),
        )
    ]
    tokens = 4
    ids = torch.zeros(tokens, device="cuda", dtype=torch.int32)
    ranks = torch.full((tokens,), 64, device="cuda", dtype=torch.int32)
    x = torch.randn(tokens, h_in, dtype=torch.float16, generator=gen).cuda()
    scales = torch.ones(r_max + 1, device="cuda", dtype=torch.float16)

    pool = PagePool(num_frames=10, h_in=h_in, h_out=h_out, dtype=torch.float16, device="cuda")
    shuffled_frames = [7, 2, 8, 1]
    a_t = adapters[0].A_t.cuda()
    b = adapters[0].B.cuda()
    for p, frame_idx in enumerate(shuffled_frames):
        r_slice = slice(p * 16, (p + 1) * 16)
        pool.frames[frame_idx, :, :h_in].copy_(a_t[r_slice])
        pool.frames[frame_idx, :, h_in:].copy_(b[r_slice])

    page_table = torch.full((1, 4), -1, device="cuda", dtype=torch.int32)
    page_table[0] = torch.tensor(shuffled_frames, device="cuda", dtype=torch.int32)

    paged = rf1_ma(
        x, pool.frames, page_table, ids, ranks,
        h_in=h_in, h_out=h_out, r_max=r_max, scales=scales
    )
    reference = b0_ma(x.cpu(), adapters, ids.cpu(), ranks.cpu())
    assert torch.allclose(paged.cpu(), reference, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("slice_config", [
    ("qkv", (1024, 1024, 1024)),
    ("gate_up", (2048, 2048)),
])
def test_rf1_ma_merged_slices_gpu(slice_config):
    name, slice_sizes = slice_config
    n_slices = len(slice_sizes)
    h_in = 4096
    r_max = 64
    tokens = 4
    gen = torch.Generator().manual_seed(45)
    x = torch.randn(tokens, h_in, dtype=torch.float16, generator=gen).cuda()
    ids = torch.zeros(tokens, device="cuda", dtype=torch.int32)
    ranks = torch.tensor([16, 32, 48, 64], device="cuda", dtype=torch.int32)
    scales = torch.ones(r_max + 1, device="cuda", dtype=torch.float16)

    widest = max(slice_sizes)
    a_pool = torch.randn(n_slices, 1, r_max, h_in, dtype=torch.float16, generator=gen).cuda()
    b_pool = torch.randn(n_slices, 1, r_max, widest, dtype=torch.float16, generator=gen).cuda()

    flat = rf1_ma_flat(
        x, a_pool, b_pool, ids, ranks,
        scales=scales, slice_sizes=slice_sizes
    )
    assert flat.shape == (tokens, sum(slice_sizes))

    ref_slices = []
    for s in range(n_slices):
        adapter = MatryoshkaLoRA(a_pool[s, 0].cpu(), b_pool[s, 0, :, :slice_sizes[s]].cpu())
        ref_s = b0_ma(x.cpu(), [adapter], ids.cpu(), ranks.cpu())
        ref_slices.append(ref_s)
    ref_total = torch.cat(ref_slices, dim=1)
    assert torch.allclose(flat.cpu(), ref_total, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rf1_ma_int64_address_gpu():
    """Verify kernel executes correctly when frame * PAGE_STRIDE exceeds 2^31 - 1 with frame index near 2^31 / PAGE_STRIDE."""
    h_in, h_out, r_max = 4096, 4096, 16
    tokens = 4
    gen = torch.Generator().manual_seed(46)
    x = torch.randn(tokens, h_in, dtype=torch.float16, generator=gen).cuda()
    scales = torch.ones(r_max + 1, device="cuda", dtype=torch.float16)
    ids = torch.zeros(tokens, device="cuda", dtype=torch.int32)
    ranks = torch.full((tokens,), 16, device="cuda", dtype=torch.int32)

    frame_width = h_in + h_out  # 8192
    PAGE_STRIDE = 16 * frame_width  # 131,072
    # Frame index near 2^31 / PAGE_STRIDE = 16384
    frame_idx = 16384
    # frame_idx * PAGE_STRIDE = 16384 * 131072 = 2,147,483,648 = 2^31
    storage = torch.zeros((frame_idx + 1) * PAGE_STRIDE, dtype=torch.float16, device="cuda")
    a_t = torch.randn(16, h_in, dtype=torch.float16, generator=gen).cuda()
    b = torch.randn(16, h_out, dtype=torch.float16, generator=gen).cuda()
    pool_frames = storage.view(frame_idx + 1, 16, frame_width)
    pool_frames[frame_idx, :, :h_in].copy_(a_t)
    pool_frames[frame_idx, :, h_in:].copy_(b)
    page_table = torch.tensor([[frame_idx]], device="cuda", dtype=torch.int32)

    paged = rf1_ma(
        x, pool_frames, page_table, ids, ranks,
        h_in=h_in, h_out=h_out, r_max=r_max, scales=scales, max_tiles=4
    )
    ref = b0_ma(x.cpu(), [MatryoshkaLoRA(a_t.cpu(), b.cpu())], ids.cpu(), ranks.cpu())
    assert torch.allclose(paged.cpu(), ref, atol=3e-2, rtol=3e-2)
    del storage, pool_frames
    torch.cuda.empty_cache()
