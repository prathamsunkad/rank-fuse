"""Multi-adapter RF1 kernels with split-K shrink and expand.

This module implements two kernels:
  - `rf1_ma_flat`: unpaged multi-adapter Matryoshka LoRA
  - `rf1_ma`: prefix-paged multi-adapter Matryoshka LoRA with frame addressing
    via a fixed-address page table.

Both kernels follow the two-pass split-K shrink + expand architecture of
`rankfuse_fused.py`, using Tensor Cores via `tl.dot`, per-token rank masking,
preallocated scale tables, and a graph-safe tile table that bounds empty
groups and assigns each token tile (BM >= 16) to exactly one adapter group.
"""

from __future__ import annotations

import functools
import os
from typing import Any

import torch
import triton
import triton.language as tl

from rankfuse.paging.page_pool import PagePool

DEFAULT_MAX_TILES = 1024
DEFAULT_BM: int = 16
MAX_R_PAD = 256


# =====================================================================
# 1. Tile table construction kernel (GPU-side, graph-safe)
# =====================================================================

@triton.jit
def _build_tile_table_kernel(
    group_starts_ptr,
    group_lens_ptr,
    group_adapters_ptr,
    group_rank_limits_ptr,
    tile_offsets_ptr,
    tile_adapter_ptr,
    tile_tok_start_ptr,
    tile_tok_len_ptr,
    tile_r_limit_ptr,
    G,
    MAX_TILES: tl.constexpr,
    BM: tl.constexpr,
):
    gid = tl.program_id(0)
    if gid >= G:
        return
    g_start = tl.load(group_starts_ptr + gid)
    g_len = tl.load(group_lens_ptr + gid)
    adapter = tl.load(group_adapters_ptr + gid)
    r_limit = tl.load(group_rank_limits_ptr + gid)
    out_base = tl.load(tile_offsets_ptr + gid)

    # Number of tiles for this group
    n_tiles = (g_len + BM - 1) // BM
    for t in range(0, n_tiles):
        tile_idx = out_base + t
        if tile_idx < MAX_TILES:
            t_start = g_start + t * BM
            rem = g_len - t * BM
            t_len = tl.minimum(BM, rem)
            tl.store(tile_adapter_ptr + tile_idx, adapter)
            tl.store(tile_tok_start_ptr + tile_idx, t_start)
            tl.store(tile_tok_len_ptr + tile_idx, t_len)
            tl.store(tile_r_limit_ptr + tile_idx, r_limit)


# =====================================================================
# 2. Split-K Shrink Kernels
# =====================================================================

@triton.jit
def _rf1_ma_flat_shrink_splitk_kernel(
    x_ptr,
    a_ptr,
    ws_ptr,
    tile_adapter_ptr,
    tile_tok_start_ptr,
    tile_tok_len_ptr,
    tile_r_limit_ptr,
    T,
    H_IN,
    K_PER,
    R_MAX,
    A_SLICE,
    WS_SLICE,
    T_PAD,
    BM: tl.constexpr,
    BH: tl.constexpr,
    R_PAD: tl.constexpr,
):
    pid_m = tl.program_id(0)
    tile_len = tl.load(tile_tok_len_ptr + pid_m)
    if tile_len <= 0:
        return

    pid_k = tl.program_id(1)
    pid_s = tl.program_id(2)

    tok_start = tl.load(tile_tok_start_ptr + pid_m)
    adapter = tl.load(tile_adapter_ptr + pid_m)
    r_limit = tl.load(tile_r_limit_ptr + pid_m)

    offs_m = tl.arange(0, BM)
    row_ok = offs_m < tile_len
    rows = tok_start + offs_m

    k = tl.arange(0, R_PAD)
    k_ok = k < r_limit

    # base address in int64
    a_base = (
        pid_s.to(tl.int64) * A_SLICE
        + adapter.to(tl.int64) * (R_MAX * H_IN)
    )

    h_lo = pid_k * K_PER
    v = tl.zeros((BM, R_PAD), dtype=tl.float32)
    for h_start in range(0, K_PER, BH):
        h = h_lo + h_start + tl.arange(0, BH)
        h_ok = h < H_IN
        x = tl.load(
            x_ptr + rows[:, None].to(tl.int64) * H_IN + h[None, :],
            mask=row_ok[:, None] & h_ok[None, :],
            other=0.0,
        )
        a = tl.load(
            a_ptr + a_base + k[:, None].to(tl.int64) * H_IN + h[None, :],
            mask=k_ok[:, None] & h_ok[None, :],
            other=0.0,
        )
        v += tl.dot(x, tl.trans(a), input_precision="ieee")

    ws = (
        ws_ptr
        + pid_s.to(tl.int64) * WS_SLICE
        + pid_k.to(tl.int64) * (T_PAD * R_PAD)
        + rows[:, None].to(tl.int64) * R_PAD
        + k[None, :]
    )
    tl.store(ws, v, mask=row_ok[:, None])


@triton.jit
def _rf1_ma_paged_shrink_splitk_kernel(
    x_ptr,
    pool_ptr,
    page_table_ptr,
    ws_ptr,
    tile_adapter_ptr,
    tile_tok_start_ptr,
    tile_tok_len_ptr,
    tile_r_limit_ptr,
    T,
    H_IN,
    K_PER,
    PAGES,
    FRAME_WIDTH,
    PAGE_STRIDE,
    WS_SLICE,
    T_PAD,
    BM: tl.constexpr,
    BH: tl.constexpr,
    R_PAD: tl.constexpr,
):
    pid_m = tl.program_id(0)
    tile_len = tl.load(tile_tok_len_ptr + pid_m)
    if tile_len <= 0:
        return

    pid_k = tl.program_id(1)
    pid_s = tl.program_id(2)

    tok_start = tl.load(tile_tok_start_ptr + pid_m)
    adapter = tl.load(tile_adapter_ptr + pid_m)
    r_limit = tl.load(tile_r_limit_ptr + pid_m)

    offs_m = tl.arange(0, BM)
    row_ok = offs_m < tile_len
    rows = tok_start + offs_m

    k = tl.arange(0, R_PAD)
    page = k // 16
    page_ok = page < PAGES
    pt_offset = adapter.to(tl.int64) * PAGES + page.to(tl.int64)
    frame = tl.load(page_table_ptr + pt_offset, mask=page_ok, other=-1)
    k_ok = (frame >= 0) & (k < r_limit)

    # Cast to int64 BEFORE multiplying by PAGE_STRIDE (avoids 32-bit overflow)
    frame_safe = tl.where(frame >= 0, frame, 0)
    k_offset = frame_safe.to(tl.int64) * PAGE_STRIDE + (k % 16).to(tl.int64) * FRAME_WIDTH

    h_lo = pid_k * K_PER
    v = tl.zeros((BM, R_PAD), dtype=tl.float32)
    for h_start in range(0, K_PER, BH):
        h = h_lo + h_start + tl.arange(0, BH)
        h_ok = h < H_IN
        x = tl.load(
            x_ptr + rows[:, None].to(tl.int64) * H_IN + h[None, :],
            mask=row_ok[:, None] & h_ok[None, :],
            other=0.0,
        )
        a = tl.load(
            pool_ptr + k_offset[:, None] + h[None, :],
            mask=k_ok[:, None] & h_ok[None, :],
            other=0.0,
        )
        v += tl.dot(x, tl.trans(a), input_precision="ieee")

    ws = (
        ws_ptr
        + pid_s.to(tl.int64) * WS_SLICE
        + pid_k.to(tl.int64) * (T_PAD * R_PAD)
        + rows[:, None].to(tl.int64) * R_PAD
        + k[None, :]
    )
    tl.store(ws, v, mask=row_ok[:, None])


# =====================================================================
# 3. Split-K Expand Kernels
# =====================================================================

@triton.jit
def _rf1_ma_flat_expand_splitk_kernel(
    ws_ptr,
    ranks_ptr,
    scale_tab_ptr,
    b_ptr,
    out_ptr,
    tile_adapter_ptr,
    tile_tok_start_ptr,
    tile_tok_len_ptr,
    tile_r_limit_ptr,
    slice_off_ptr,
    slice_size_ptr,
    T,
    N_MAX,
    OUT_STRIDE,
    cols_per_prog,
    SPLIT_K,
    R_MAX,
    B_SLICE,
    WS_SLICE,
    T_PAD,
    BM: tl.constexpr,
    BN: tl.constexpr,
    R_PAD: tl.constexpr,
    ADD_OUT: tl.constexpr,
    ALIGNED: tl.constexpr,
    FP16_EXPAND: tl.constexpr = False,
):
    pid_m = tl.program_id(0)
    tile_len = tl.load(tile_tok_len_ptr + pid_m)
    if tile_len <= 0:
        return

    pid_s = tl.program_id(2)
    out_col0 = tl.load(slice_off_ptr + pid_s)
    n_cols = tl.load(slice_size_ptr + pid_s)
    if ALIGNED:
        out_col0 = tl.multiple_of(out_col0, 16)
        n_cols = tl.multiple_of(n_cols, 16)

    col_base = tl.program_id(1) * cols_per_prog
    if col_base >= n_cols:
        return

    tok_start = tl.load(tile_tok_start_ptr + pid_m)
    adapter = tl.load(tile_adapter_ptr + pid_m)
    r_limit = tl.load(tile_r_limit_ptr + pid_m)

    offs_m = tl.arange(0, BM)
    row_ok = offs_m < tile_len
    rows = tok_start + offs_m

    ranks = tl.load(ranks_ptr + rows, mask=row_ok, other=0)
    k = tl.arange(0, R_PAD)
    k_ok = k < r_limit

    ws_s = ws_ptr + pid_s.to(tl.int64) * WS_SLICE
    v = tl.zeros((BM, R_PAD), dtype=tl.float32)
    for split in range(0, SPLIT_K):
        v += tl.load(
            ws_s
            + tl.cast(split, tl.int64) * (T_PAD * R_PAD)
            + rows[:, None].to(tl.int64) * R_PAD
            + k[None, :],
            mask=row_ok[:, None],
            other=0.0,
        )

    # Per-token rank mask
    v = tl.where(k[None, :] < ranks[:, None], v, 0.0)

    # Per-token scale from preallocated scale table
    scale = tl.load(scale_tab_ptr + ranks, mask=row_ok, other=0.0)
    v = v * scale[:, None]

    b_base = (
        pid_s.to(tl.int64) * B_SLICE
        + adapter.to(tl.int64) * (R_MAX * N_MAX)
    )

    for col_start in range(0, cols_per_prog, BN):
        cols = col_base + col_start + tl.arange(0, BN)
        col_ok = cols < n_cols
        b = tl.load(
            b_ptr + b_base + k[:, None].to(tl.int64) * N_MAX + cols[None, :],
            mask=k_ok[:, None] & col_ok[None, :],
            other=0.0,
        )
        if FP16_EXPAND:
            acc = tl.dot(v.to(b.dtype), b, input_precision="ieee")
        else:
            acc = tl.dot(v, b.to(tl.float32), input_precision="ieee")
        ptrs = out_ptr + rows[:, None].to(tl.int64) * OUT_STRIDE + (out_col0 + cols[None, :])
        mask = row_ok[:, None] & col_ok[None, :]
        val = acc.to(out_ptr.dtype.element_ty)
        if ADD_OUT:
            val += tl.load(ptrs, mask=mask, other=0.0)
        tl.store(ptrs, val, mask=mask)


@triton.jit
def _rf1_ma_paged_expand_splitk_kernel(
    ws_ptr,
    ranks_ptr,
    scale_tab_ptr,
    pool_ptr,
    page_table_ptr,
    out_ptr,
    tile_adapter_ptr,
    tile_tok_start_ptr,
    tile_tok_len_ptr,
    tile_r_limit_ptr,
    slice_off_ptr,
    slice_size_ptr,
    T,
    N_MAX,
    OUT_STRIDE,
    cols_per_prog,
    SPLIT_K,
    PAGES,
    FRAME_WIDTH,
    PAGE_STRIDE,
    B_START,
    WS_SLICE,
    T_PAD,
    BM: tl.constexpr,
    BN: tl.constexpr,
    R_PAD: tl.constexpr,
    ADD_OUT: tl.constexpr,
    ALIGNED: tl.constexpr,
    FP16_EXPAND: tl.constexpr = False,
):
    pid_m = tl.program_id(0)
    tile_len = tl.load(tile_tok_len_ptr + pid_m)
    if tile_len <= 0:
        return

    pid_s = tl.program_id(2)
    out_col0 = tl.load(slice_off_ptr + pid_s)
    n_cols = tl.load(slice_size_ptr + pid_s)
    if ALIGNED:
        out_col0 = tl.multiple_of(out_col0, 16)
        n_cols = tl.multiple_of(n_cols, 16)

    col_base = tl.program_id(1) * cols_per_prog
    if col_base >= n_cols:
        return

    tok_start = tl.load(tile_tok_start_ptr + pid_m)
    adapter = tl.load(tile_adapter_ptr + pid_m)
    r_limit = tl.load(tile_r_limit_ptr + pid_m)

    offs_m = tl.arange(0, BM)
    row_ok = offs_m < tile_len
    rows = tok_start + offs_m

    ranks = tl.load(ranks_ptr + rows, mask=row_ok, other=0)
    k = tl.arange(0, R_PAD)

    page = k // 16
    page_ok = page < PAGES
    pt_offset = adapter.to(tl.int64) * PAGES + page.to(tl.int64)
    frame = tl.load(page_table_ptr + pt_offset, mask=page_ok, other=-1)
    k_ok = (frame >= 0) & (k < r_limit)

    ws_s = ws_ptr + pid_s.to(tl.int64) * WS_SLICE
    v = tl.zeros((BM, R_PAD), dtype=tl.float32)
    for split in range(0, SPLIT_K):
        v += tl.load(
            ws_s
            + tl.cast(split, tl.int64) * (T_PAD * R_PAD)
            + rows[:, None].to(tl.int64) * R_PAD
            + k[None, :],
            mask=row_ok[:, None],
            other=0.0,
        )

    # Per-token rank mask
    v = tl.where(k[None, :] < ranks[:, None], v, 0.0)

    # Per-token scale from preallocated scale table
    scale = tl.load(scale_tab_ptr + ranks, mask=row_ok, other=0.0)
    v = v * scale[:, None]

    # Frame B offset cast to int64 before multiply
    frame_safe = tl.where(frame >= 0, frame, 0)
    k_b_offset = (
        frame_safe.to(tl.int64) * PAGE_STRIDE
        + (k % 16).to(tl.int64) * FRAME_WIDTH
        + B_START
        + out_col0
    )

    for col_start in range(0, cols_per_prog, BN):
        cols = col_base + col_start + tl.arange(0, BN)
        col_ok = cols < n_cols
        b = tl.load(
            pool_ptr + k_b_offset[:, None] + cols[None, :],
            mask=k_ok[:, None] & col_ok[None, :],
            other=0.0,
        )
        if FP16_EXPAND:
            acc = tl.dot(v.to(b.dtype), b, input_precision="ieee")
        else:
            acc = tl.dot(v, b.to(tl.float32), input_precision="ieee")
        ptrs = out_ptr + rows[:, None].to(tl.int64) * OUT_STRIDE + (out_col0 + cols[None, :])
        mask = row_ok[:, None] & col_ok[None, :]
        val = acc.to(out_ptr.dtype.element_ty)
        if ADD_OUT:
            val += tl.load(ptrs, mask=mask, other=0.0)
        tl.store(ptrs, val, mask=mask)


# =====================================================================
# 4. Host helpers & caching
# =====================================================================

@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    if not torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1":
        return 1
    return torch.cuda.get_device_properties(device_index).multi_processor_count


_SLICE_TABLES: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _slice_tables(sizes: tuple[int, ...], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (sizes, device)
    if key not in _SLICE_TABLES:
        offsets = [sum(sizes[:s]) for s in range(len(sizes))]
        _SLICE_TABLES[key] = (
            torch.tensor(offsets, dtype=torch.int32, device=device),
            torch.tensor(sizes, dtype=torch.int32, device=device),
        )
    return _SLICE_TABLES[key]


_TILE_BUFFERS: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _get_tile_buffers(max_tiles: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (max_tiles, device)
    if key not in _TILE_BUFFERS:
        _TILE_BUFFERS[key] = (
            torch.zeros(max_tiles, dtype=torch.int32, device=device),
            torch.zeros(max_tiles, dtype=torch.int32, device=device),
            torch.zeros(max_tiles, dtype=torch.int32, device=device),
            torch.zeros(max_tiles, dtype=torch.int32, device=device),
        )
    return _TILE_BUFFERS[key]


_WS_BUFFERS: dict[tuple, torch.Tensor] = {}


def _get_workspace(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    key = (shape, device)
    buf = _WS_BUFFERS.get(key)
    if buf is None or buf.shape != shape or buf.device != device:
        buf = torch.empty(shape, dtype=torch.float32, device=device)
        _WS_BUFFERS[key] = buf
    return buf


def _output(x: torch.Tensor, h_out: int, out: torch.Tensor | None, add: bool) -> torch.Tensor:
    if out is not None:
        if tuple(out.shape) != (x.shape[0], h_out):
            raise ValueError(f"out has incompatible shape {out.shape}, expected {(x.shape[0], h_out)}")
        return out
    if add:
        raise ValueError("add=True requires an existing out tensor to add into")
    return torch.empty((x.shape[0], h_out), device=x.device, dtype=x.dtype)


def _prepare_groups(
    adapter_ids: torch.Tensor,
    token_ranks: torch.Tensor,
    r_max: int,
    num_tokens: int,
    group_starts: torch.Tensor | None,
    group_lens: torch.Tensor | None,
    group_rank_limit: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if num_tokens == 0:
        empty = torch.empty(0, dtype=torch.int32, device=device)
        return empty, empty, empty, empty

    if group_starts is not None and group_lens is not None:
        starts = group_starts.to(device=device, dtype=torch.int32)
        lens = group_lens.to(device=device, dtype=torch.int32)
        G = starts.numel()
        if G == 0:
            adapters = torch.empty(0, dtype=torch.int32, device=device)
        elif adapter_ids.numel() == G:
            adapters = adapter_ids.to(device=device, dtype=torch.int32)
        else:
            adapters = adapter_ids[starts.long()].to(device=device, dtype=torch.int32)
        if group_rank_limit is not None:
            limits = group_rank_limit.to(device=device, dtype=torch.int32)
        else:
            limits = torch.full((G,), r_max, dtype=torch.int32, device=device)
        return starts, lens, adapters, limits

    # Automatically derive contiguous groups from adapter_ids
    if adapter_ids.numel() != num_tokens:
        raise ValueError("adapter_ids must have length T when group_starts/group_lens are omitted")
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        # Inside CUDA graph capture: dynamic nonzero shape allocation is illegal.
        # Treat as a single contiguous group covering all tokens.
        zero = torch.zeros(1, dtype=torch.int32, device=device)
        t_end = torch.full((1,), num_tokens, dtype=torch.int32, device=device)
        starts = zero
        lens = t_end
        adapters = adapter_ids[:1]
        limits = torch.full((1,), r_max, dtype=torch.int32, device=device)
        return starts, lens, adapters, limits

    diff = adapter_ids[1:] != adapter_ids[:-1]
    change_pts = torch.nonzero(diff).squeeze(-1) + 1
    zero = torch.zeros(1, dtype=torch.int32, device=device)
    t_end = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    starts = torch.cat([zero, change_pts.to(dtype=torch.int32, device=device)])
    ends = torch.cat([change_pts.to(dtype=torch.int32, device=device), t_end])
    lens = ends - starts
    adapters = adapter_ids[starts.long()].to(device=device, dtype=torch.int32)
    limits = torch.full_like(starts, r_max, dtype=torch.int32)
    return starts, lens, adapters, limits


# =====================================================================
# 5. Launcher: rf1_ma_flat
# =====================================================================

def rf1_ma_flat(
    x: torch.Tensor,
    a_pool: torch.Tensor,
    b_pool: torch.Tensor,
    adapter_ids: torch.Tensor,
    token_ranks: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    add: bool = False,
    scales: torch.Tensor,
    group_starts: torch.Tensor | None = None,
    group_lens: torch.Tensor | None = None,
    group_rank_limit: torch.Tensor | None = None,
    slice_sizes: list[int] | tuple[int, ...] | None = None,
    split_k: int | None = None,
    max_tiles: int = DEFAULT_MAX_TILES,
    fp16_expand: bool | None = None,
) -> torch.Tensor:
    """Flat multi-adapter RF1 kernel with split-K shrink and expand."""
    if not (x.is_cuda or os.environ.get("TRITON_INTERPRET") == "1"):
        raise RuntimeError("rf1_ma_flat requires CUDA tensors (or TRITON_INTERPRET=1)")
    if scales is None or not isinstance(scales, torch.Tensor):
        raise ValueError("scales (scale_table) must be provided as a preallocated tensor")

    if a_pool.dim() == 3:
        # [N, r_max, h_in] -> [1, N, r_max, h_in]
        a_pool = a_pool.unsqueeze(0)
    if b_pool.dim() == 3:
        b_pool = b_pool.unsqueeze(0)

    n_slices, n_adapters, r_max, h_in = a_pool.shape
    if b_pool.shape[0] != n_slices or b_pool.shape[1] != n_adapters or b_pool.shape[2] != r_max:
        raise ValueError("a_pool and b_pool dimensions mismatch")
    n_max = b_pool.shape[3]

    sizes = tuple(int(s) for s in slice_sizes) if slice_sizes is not None else (n_max,) * n_slices
    if len(sizes) != n_slices or not all(0 < s <= n_max for s in sizes):
        raise ValueError(f"slice_sizes must give {n_slices} widths in [1, {n_max}], got {sizes}")
    total_out_cols = sum(sizes)

    num_tokens = x.shape[0]
    result = _output(x, total_out_cols, out, add)
    if num_tokens == 0:
        return result

    if fp16_expand is None:
        fp16_expand = os.environ.get("RF_FP16_EXPAND", "0") == "1"

    # Dimensions and tile configuration
    BM = DEFAULT_BM
    BH = 16 if h_in <= 16 else (32 if h_in <= 32 else (64 if h_in <= 64 else 128))
    widest = max(sizes)
    BN = 16 if widest <= 16 else (32 if widest <= 32 else (64 if widest <= 64 else 128))
    r_pad = max(16, triton.next_power_of_2(max(r_max, 1)))

    # Build group metadata and tile table
    starts, lens, adapters, limits = _prepare_groups(
        adapter_ids, token_ranks, r_max, num_tokens, group_starts, group_lens, group_rank_limit, x.device
    )
    G = starts.numel()

    tile_adapter, tile_tok_start, tile_tok_len, tile_r_limit = _get_tile_buffers(max_tiles, x.device)
    tile_tok_len.zero_()

    if G > 0:
        tiles_per_group = torch.div(lens + (BM - 1), BM, rounding_mode="floor")
        tile_offsets = torch.zeros_like(tiles_per_group)
        tile_offsets[1:] = torch.cumsum(tiles_per_group[:-1], dim=0)
        _build_tile_table_kernel[(G,)](
            starts, lens, adapters, limits, tile_offsets,
            tile_adapter, tile_tok_start, tile_tok_len, tile_r_limit,
            G, MAX_TILES=max_tiles, BM=BM,
        )

    # Determine split-K and active grid
    act_tiles = min(max_tiles, max(1, num_tokens))
    device_idx = x.device.index or 0
    num_sms = _num_sms(device_idx)
    if split_k is None:
        if not x.is_cuda:
            split_k = 1
        else:
            split_k = max(1, min(16, num_sms // (act_tiles * n_slices)))
    k_per = triton.cdiv(triton.cdiv(h_in, split_k), BH) * BH
    split_k = triton.cdiv(h_in, k_per)

    # Workspace: [n_slices, split_k, t_pad, r_pad]
    t_pad = max_tiles * BM
    ws_shape = (n_slices, split_k, t_pad, r_pad)
    ws = _get_workspace(ws_shape, x.device)
    ws_slice = split_k * t_pad * r_pad

    a_slice = n_adapters * r_max * h_in
    b_slice = n_adapters * r_max * n_max

    # 1. Launch Split-K Shrink Kernel
    grid_shrink = (act_tiles, split_k, n_slices)
    _rf1_ma_flat_shrink_splitk_kernel[grid_shrink](
        x, a_pool, ws, tile_adapter, tile_tok_start, tile_tok_len, tile_r_limit,
        num_tokens, h_in, k_per, r_max, a_slice, ws_slice, t_pad,
        BM=BM, BH=BH, R_PAD=r_pad,
    )

    # 2. Launch Split-K Expand Kernel
    offsets, widths = _slice_tables(sizes, x.device)
    aligned = all(s % 16 == 0 for s in sizes)
    n_tiles = triton.cdiv(widest, BN)
    n_split = min(n_tiles, max(1, num_sms // (act_tiles * n_slices)))
    cols_per_prog = triton.cdiv(n_tiles, n_split) * BN

    grid_expand = (act_tiles, triton.cdiv(widest, cols_per_prog), n_slices)
    _rf1_ma_flat_expand_splitk_kernel[grid_expand](
        ws, token_ranks, scales, b_pool, result,
        tile_adapter, tile_tok_start, tile_tok_len, tile_r_limit,
        offsets, widths,
        num_tokens, n_max, result.stride(0), cols_per_prog, split_k,
        r_max, b_slice, ws_slice, t_pad,
        BM=BM, BN=BN, R_PAD=r_pad, ADD_OUT=add, ALIGNED=aligned, FP16_EXPAND=fp16_expand,
    )
    return result


# =====================================================================
# 6. Launcher: rf1_ma (Paged)
# =====================================================================

def rf1_ma(
    x: torch.Tensor,
    pool: torch.Tensor | PagePool,
    page_table: torch.Tensor,
    adapter_ids: torch.Tensor,
    token_ranks: torch.Tensor,
    *,
    h_in: int,
    h_out: int,
    r_max: int,
    out: torch.Tensor | None = None,
    add: bool = False,
    scales: torch.Tensor,
    group_starts: torch.Tensor | None = None,
    group_lens: torch.Tensor | None = None,
    group_rank_limit: torch.Tensor | None = None,
    slice_sizes: list[int] | tuple[int, ...] | None = None,
    split_k: int | None = None,
    max_tiles: int = DEFAULT_MAX_TILES,
    fp16_expand: bool | None = None,
) -> torch.Tensor:
    """Prefix-paged multi-adapter RF1 kernel with frame addressing."""
    if hasattr(pool, "frames"):
        pool = pool.frames
    if not (x.is_cuda or os.environ.get("TRITON_INTERPRET") == "1"):
        raise RuntimeError("rf1_ma requires CUDA tensors (or TRITON_INTERPRET=1)")
    if scales is None or not isinstance(scales, torch.Tensor):
        raise ValueError("scales (scale_table) must be provided as a preallocated tensor")

    if pool.ndim != 3 or pool.shape[1] != 16:
        raise ValueError(f"pool must have shape [frames, 16, frame_width], got {pool.shape}")

    frame_width = pool.shape[2]
    page_stride = pool.stride(0)

    n_slices = len(slice_sizes) if slice_sizes is not None else 1
    sizes = tuple(int(s) for s in slice_sizes) if slice_sizes is not None else (h_out,)
    total_out_cols = sum(sizes)
    widest = max(sizes)
    n_max = widest

    num_tokens = x.shape[0]
    result = _output(x, total_out_cols, out, add)
    if num_tokens == 0:
        return result

    if fp16_expand is None:
        fp16_expand = os.environ.get("RF_FP16_EXPAND", "0") == "1"

    # Dimensions and tile configuration
    BM = DEFAULT_BM
    BH = 16 if h_in <= 16 else (32 if h_in <= 32 else (64 if h_in <= 64 else 128))
    BN = 16 if widest <= 16 else (32 if widest <= 32 else (64 if widest <= 64 else 128))
    r_pad = max(16, triton.next_power_of_2(max(r_max, 1)))

    # Frame B column start (h_in if combined frame, else 0)
    b_start = h_in if frame_width >= h_in + h_out else 0

    # Build group metadata and tile table
    starts, lens, adapters, limits = _prepare_groups(
        adapter_ids, token_ranks, r_max, num_tokens, group_starts, group_lens, group_rank_limit, x.device
    )
    G = starts.numel()

    tile_adapter, tile_tok_start, tile_tok_len, tile_r_limit = _get_tile_buffers(max_tiles, x.device)
    tile_tok_len.zero_()

    if G > 0:
        tiles_per_group = torch.div(lens + (BM - 1), BM, rounding_mode="floor")
        tile_offsets = torch.zeros_like(tiles_per_group)
        tile_offsets[1:] = torch.cumsum(tiles_per_group[:-1], dim=0)
        _build_tile_table_kernel[(G,)](
            starts, lens, adapters, limits, tile_offsets,
            tile_adapter, tile_tok_start, tile_tok_len, tile_r_limit,
            G, MAX_TILES=max_tiles, BM=BM,
        )

    # Determine split-K and active grid
    act_tiles = min(max_tiles, max(1, num_tokens))
    device_idx = x.device.index or 0
    num_sms = _num_sms(device_idx)
    if split_k is None:
        if not x.is_cuda:
            split_k = 1
        else:
            split_k = max(1, min(16, num_sms // (act_tiles * n_slices)))
    k_per = triton.cdiv(triton.cdiv(h_in, split_k), BH) * BH
    split_k = triton.cdiv(h_in, k_per)

    # Workspace: [n_slices, split_k, t_pad, r_pad]
    t_pad = max_tiles * BM
    ws_shape = (n_slices, split_k, t_pad, r_pad)
    ws = _get_workspace(ws_shape, x.device)
    ws_slice = split_k * t_pad * r_pad

    pages = page_table.shape[1]

    # 1. Launch Split-K Paged Shrink Kernel
    grid_shrink = (act_tiles, split_k, n_slices)
    _rf1_ma_paged_shrink_splitk_kernel[grid_shrink](
        x, pool, page_table, ws,
        tile_adapter, tile_tok_start, tile_tok_len, tile_r_limit,
        num_tokens, h_in, k_per, pages, frame_width, page_stride,
        ws_slice, t_pad,
        BM=BM, BH=BH, R_PAD=r_pad,
    )

    # 2. Launch Split-K Paged Expand Kernel
    offsets, widths = _slice_tables(sizes, x.device)
    aligned = all(s % 16 == 0 for s in sizes)
    n_tiles = triton.cdiv(widest, BN)
    n_split = min(n_tiles, max(1, num_sms // (act_tiles * n_slices)))
    cols_per_prog = triton.cdiv(n_tiles, n_split) * BN

    grid_expand = (act_tiles, triton.cdiv(widest, cols_per_prog), n_slices)
    _rf1_ma_paged_expand_splitk_kernel[grid_expand](
        ws, token_ranks, scales, pool, page_table, result,
        tile_adapter, tile_tok_start, tile_tok_len, tile_r_limit,
        offsets, widths,
        num_tokens, n_max, result.stride(0), cols_per_prog, split_k,
        pages, frame_width, page_stride, b_start, ws_slice, t_pad,
        BM=BM, BN=BN, R_PAD=r_pad, ADD_OUT=add, ALIGNED=aligned, FP16_EXPAND=fp16_expand,
    )
    return result
