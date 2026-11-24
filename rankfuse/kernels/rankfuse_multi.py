"""Stage 2 kernels: many Matryoshka adapters, mixed ranks, weights behind a page table.

For every token t with adapter a_t and rank r_t (rankfuse/multi.py):

    y[t, slice s] = scale(a_t, r_t) * x[t] @ A[a_t, s, :r_t].T @ B[a_t, s, :r_t]

rf1_ma, rf2_ma_splitk(_fw), rf2_ma_widths, rf2_ma_segment and rf2_ma_bucketed(_seq) run on the
shared split-K kernels themselves (rankfuse_fused.py, MULTI mode, section 9; multi_lora_splitk and
multi_lora_bucketed_splitk below). The *_legacy implementations use this module's own kernel pair, rf1_fused's split-K design (rankfuse_fused.py, section 8) run over a tile table
instead of fixed token blocks:

    shrink  grid (capacity, S_k, n_slices): program (i, k, s) takes tile i = (adapter, first
            position, length, rank limit) from the per-step metadata, gathers its tokens through
            `order`, and reduces h_in slice k over the limit's rank rows of that adapter's A;
            partial v [BM, R] in fp32 -> ws[s, k, position, :R]
    expand  the same tile: sum the S_k partials, mask v[t, j] = 0 for j >= r_t, scale by
            scale(a, r_t), cast, multiply by the adapter's B rows and write (or add into) each
            token's own output row. Up to 128 tokens: a column strip per program (EXPAND_WAVES
            programs per SM); above: one BM x BN tile per program. A tile whose limit is 0 (the
            no-LoRA group) only writes zeros (nothing with add=True); empty tiles (rows past the
            step's tile count in a fixed-capacity table) exit at once.

Weights are read through the page table: rank row j of adapter a, slice s, lives at
frame(a, j // 16) * frame_stride + slice_off[s] + (j % 16) * ld. Each program loads the frame of
every rank row it may read once, before its h_in (shrink) or column (expand) loop, from an L1-hot
table of r_max / 16 int32 per adapter, so the page table adds R small loads per program and
nothing per weight element.

Rank limits and the constexpr rank width
-----------------------------------------
tl.dot needs a compile-time tile shape, so the rank width R of the register tile v [BM, R] is
constexpr. Two ways are used:

    rf1_ma          (mode "group") R = R_PAD, the batch's (or the caller's) maximum rank bucket, for
                    every tile; the tile's runtime limit (its adapter group's maximum rank) masks
                    the loads of A and B rows j >= limit. A masked load issues no memory request,
                    so a group reads exactly limit rank rows per matrix: bytes follow each group's
                    limit, the MMA always runs R_PAD wide (cheap: these kernels are bound by
                    memory and latency, and one compiled variant serves every rank mix).
    rf2_ma_splitk   (mode "rank_sorted") tokens sorted by (adapter, rank); each tile's limit is its
                    own maximum rank and the tile branches at runtime to the variant compiled for
                    the smallest bucket 16/32/64/128/256 covering it (BUCKETED): fewer registers
                    and MMA work for low-rank tiles, as rf2_fused does in Stage 1.
    rf2_ma_segment  (mode "segment") no sort: tiles are runs of consecutive tokens on one adapter
                    (cut to BM), each at its own bucket. Contiguous loads, but in decode batches,
                    where neighbours rarely share an adapter, almost every tile holds one token.
    rf2_ma_bucketed a CPU plan per step: tokens split by rank bucket, then grouped by adapter; one
                    kernel pair compiled for exactly R rows per bucket (no runtime branch), the
                    buckets' pairs on concurrent streams (fork/join, capturable). The plan fixes the
                    number of launches and their grids, so a captured graph is valid only for steps
                    with the same bucket set and tile counts (pass per-bucket capacities to reuse it).

A runtime loop over 16-row chunks of v was not used: Triton cannot index a register tile with a
runtime chunk number, so it would need one accumulator per chunk or a re-read of x per chunk.

CUDA graphs: the metadata (order, tile table, page table, scale table, slice table) lives in
device tensors built outside the call; the call reads no tensor value on the host and allocates
only ws (from the graph's pool during capture) and, without `out`, the output. The grid uses the
tile table's capacity, so with `build_meta(..., capacity=tile_capacity(...), r_pad=r_max)` one
captured graph replays any later step of the same T after `meta.load_(new_meta)`.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from rankfuse.kernels import rankfuse_fused as rf
from rankfuse.multi import (
    PAGE,
    MultiMeta,
    PagedWeights,
    build_meta,
    default_block_m,
    mark_consecutive,
    output_buffer,
    register_multi,
)

assert PAGE == 16  # the kernels compute pages as k // 16


@triton.jit
def _frame_rows(pt_ptr, adapter, PT_LD, k, k_ok, FRAME, off, LD, ALIGNED: tl.constexpr, OFF64: tl.constexpr):
    # Element offset of rank row k of this adapter's matrix: frame(k // 16) * FRAME + off + (k % 16) * LD.
    frame = tl.load(pt_ptr + adapter * PT_LD + k // 16, mask=k_ok, other=0)
    if OFF64:
        rows = frame.to(tl.int64) * FRAME + off + (k % 16) * LD
    else:  # every offset fits in int32 (pools below 2**31 elements): cheaper address arithmetic
        rows = frame * FRAME + off.to(tl.int32) + (k % 16) * LD.to(tl.int32)
    if ALIGNED:
        # All of FRAME, off and LD are multiples of 16 (checked on the host): lets the loads of a
        # row be vectorized; the compiler cannot infer it from values loaded at run time.
        rows = tl.multiple_of(rows, 16)
    return rows


@triton.jit
def _tile_tokens(tiles_ptr, tile, order_ptr, pos, row_ok, BM: tl.constexpr):
    # Token indices of a tile. Tiles whose tokens are consecutive in the batch (a prompt, a
    # one-adapter batch) carry their first token and skip the dependent load of `order`.
    first = tl.load(tiles_ptr + tile * 5 + 4)
    if first >= 0:
        tok = first + tl.arange(0, BM)
    else:
        tok = tl.load(order_ptr + pos, mask=row_ok, other=0)
    return tok


@triton.jit
def _ma_shrink_part(x_ptr, a_pool, pt_ptr, ws_k, tok, pos, row_ok, adapter, limit, H_IN, PT_LD, FRAME_A, a_off,
                    h_lo, K_PER, WS_LD, BM: tl.constexpr, BH: tl.constexpr, R: tl.constexpr, ALIGNED: tl.constexpr, OFF64: tl.constexpr):
    k = tl.arange(0, R)
    k_ok = k < limit
    a_rows = _frame_rows(pt_ptr, adapter, PT_LD, k, k_ok, FRAME_A, a_off, H_IN, ALIGNED, OFF64)
    v = tl.zeros((BM, R), dtype=tl.float32)
    for h_start in range(0, K_PER, BH):
        h = h_lo + h_start + tl.arange(0, BH)
        h_ok = h < H_IN
        x = tl.load(x_ptr + tok[:, None] * H_IN + h[None, :], mask=row_ok[:, None] & h_ok[None, :], other=0.0)
        a = tl.load(a_pool + a_rows[:, None] + h[None, :], mask=k_ok[:, None] & h_ok[None, :], other=0.0)
        v += tl.dot(x, tl.trans(a), input_precision="ieee")
    tl.store(ws_k + pos[:, None] * WS_LD + k[None, :], v, mask=row_ok[:, None])


@triton.jit
def _ma_shrink_kernel(x_ptr, a_pool, pt_ptr, tiles_ptr, order_ptr, slice_ptr, ws_ptr, T, H_IN, PT_LD, FRAME_A,
                      K_PER, WS_LD, WS_SLICE, BM: tl.constexpr, BH: tl.constexpr, R_PAD: tl.constexpr,
                      BUCKETED: tl.constexpr, ALIGNED: tl.constexpr, OFF64: tl.constexpr):
    tile = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_s = tl.program_id(2)
    adapter = tl.load(tiles_ptr + tile * 5)
    start = tl.load(tiles_ptr + tile * 5 + 1)
    length = tl.load(tiles_ptr + tile * 5 + 2)
    limit = tl.load(tiles_ptr + tile * 5 + 3)
    if limit > 0:  # empty tiles and the no-LoRA group have limit 0
        pos = start + tl.arange(0, BM)
        row_ok = tl.arange(0, BM) < length
        tok = _tile_tokens(tiles_ptr, tile, order_ptr, pos, row_ok, BM=BM)
        a_off = tl.load(slice_ptr + pid_s * 5 + 2)
        h_lo = pid_k * K_PER
        ws_k = ws_ptr + pid_s * WS_SLICE + pid_k * T * WS_LD
        if BUCKETED:
            if limit <= 16:
                _ma_shrink_part(x_ptr, a_pool, pt_ptr, ws_k, tok, pos, row_ok, adapter, limit, H_IN, PT_LD, FRAME_A,
                                a_off, h_lo, K_PER, WS_LD, BM=BM, BH=BH, R=16, ALIGNED=ALIGNED, OFF64=OFF64)
            elif limit <= 32:
                if R_PAD >= 32:
                    _ma_shrink_part(x_ptr, a_pool, pt_ptr, ws_k, tok, pos, row_ok, adapter, limit, H_IN, PT_LD,
                                    FRAME_A, a_off, h_lo, K_PER, WS_LD, BM=BM, BH=BH, R=32, ALIGNED=ALIGNED, OFF64=OFF64)
            elif limit <= 64:
                if R_PAD >= 64:
                    _ma_shrink_part(x_ptr, a_pool, pt_ptr, ws_k, tok, pos, row_ok, adapter, limit, H_IN, PT_LD,
                                    FRAME_A, a_off, h_lo, K_PER, WS_LD, BM=BM, BH=BH, R=64, ALIGNED=ALIGNED, OFF64=OFF64)
            elif limit <= 128:
                if R_PAD >= 128:
                    _ma_shrink_part(x_ptr, a_pool, pt_ptr, ws_k, tok, pos, row_ok, adapter, limit, H_IN, PT_LD,
                                    FRAME_A, a_off, h_lo, K_PER, WS_LD, BM=BM, BH=BH, R=128, ALIGNED=ALIGNED, OFF64=OFF64)
            else:
                if R_PAD >= 256:
                    _ma_shrink_part(x_ptr, a_pool, pt_ptr, ws_k, tok, pos, row_ok, adapter, limit, H_IN, PT_LD,
                                    FRAME_A, a_off, h_lo, K_PER, WS_LD, BM=BM, BH=BH, R=256, ALIGNED=ALIGNED, OFF64=OFF64)
        else:
            _ma_shrink_part(x_ptr, a_pool, pt_ptr, ws_k, tok, pos, row_ok, adapter, limit, H_IN, PT_LD, FRAME_A,
                            a_off, h_lo, K_PER, WS_LD, BM=BM, BH=BH, R=R_PAD, ALIGNED=ALIGNED, OFF64=OFF64)


@triton.jit
def _ma_store_tile(v_in, b_pool, b_rows, k_ok, out_ptr, tok, row_ok, c0, n_cols, out_col0, OUT_STRIDE,
                   BN: tl.constexpr, ADD_OUT: tl.constexpr):
    cols = c0 + tl.arange(0, BN)
    col_ok = cols < n_cols
    b = tl.load(b_pool + b_rows[:, None] + cols[None, :], mask=k_ok[:, None] & col_ok[None, :], other=0.0)
    acc = tl.dot(v_in, b, input_precision="ieee")
    ptrs = out_ptr + tok[:, None] * OUT_STRIDE + out_col0 + cols[None, :]
    mask = row_ok[:, None] & col_ok[None, :]
    if ADD_OUT:
        acc += tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    tl.store(ptrs, acc, mask=mask)


@triton.jit
def _ma_expand_part(ws_s, scale_ptr, b_pool, pt_ptr, out_ptr, tok, pos, row_ok, ranks, adapter, limit, T,
                    OUT_STRIDE, PT_LD, FRAME_B, b_off, b_ld, out_col0, n_cols, SPLIT_K, WS_LD, SCALE_LD, col_base,
                    cols_per_prog, BM: tl.constexpr, BN: tl.constexpr, R: tl.constexpr, HAS_SCALE: tl.constexpr,
                    ADD_OUT: tl.constexpr, LOOP: tl.constexpr, ALIGNED: tl.constexpr,
                    OFF64: tl.constexpr):
    k = tl.arange(0, R)
    k_ok = k < limit
    v = tl.zeros((BM, R), dtype=tl.float32)
    for s in range(0, SPLIT_K):
        v += tl.load(ws_s + s * T * WS_LD + pos[:, None] * WS_LD + k[None, :], mask=row_ok[:, None], other=0.0)
    v = tl.where(k[None, :] < ranks[:, None], v, 0.0)
    if HAS_SCALE:
        v = v * tl.load(scale_ptr + adapter * SCALE_LD + ranks, mask=row_ok, other=0.0)[:, None]
    v_in = v.to(out_ptr.dtype.element_ty)
    b_rows = _frame_rows(pt_ptr, adapter, PT_LD, k, k_ok, FRAME_B, b_off, b_ld, ALIGNED, OFF64)
    if LOOP:
        for col_start in range(0, cols_per_prog, BN):
            _ma_store_tile(v_in, b_pool, b_rows, k_ok, out_ptr, tok, row_ok, col_base + col_start, n_cols, out_col0,
                           OUT_STRIDE, BN=BN, ADD_OUT=ADD_OUT)
    else:
        _ma_store_tile(v_in, b_pool, b_rows, k_ok, out_ptr, tok, row_ok, col_base, n_cols, out_col0, OUT_STRIDE,
                       BN=BN, ADD_OUT=ADD_OUT)


@triton.jit
def _ma_zero_part(out_ptr, tok, row_ok, col_base, cols_per_prog, n_cols, out_col0, OUT_STRIDE,
                  BM: tl.constexpr, BN: tl.constexpr, LOOP: tl.constexpr):
    zeros = tl.zeros((BM, BN), dtype=out_ptr.dtype.element_ty)
    if LOOP:
        for col_start in range(0, cols_per_prog, BN):
            cols = col_base + col_start + tl.arange(0, BN)
            tl.store(out_ptr + tok[:, None] * OUT_STRIDE + out_col0 + cols[None, :], zeros,
                     mask=row_ok[:, None] & (cols[None, :] < n_cols))
    else:
        cols = col_base + tl.arange(0, BN)
        tl.store(out_ptr + tok[:, None] * OUT_STRIDE + out_col0 + cols[None, :], zeros,
                 mask=row_ok[:, None] & (cols[None, :] < n_cols))


@triton.jit
def _ma_expand_kernel(ws_ptr, ranks_ptr, scale_ptr, b_pool, pt_ptr, tiles_ptr, order_ptr, slice_ptr, out_ptr, T,
                      OUT_STRIDE, PT_LD, FRAME_B, SPLIT_K, WS_LD, WS_SLICE, SCALE_LD, cols_per_prog,
                      BM: tl.constexpr, BN: tl.constexpr, R_PAD: tl.constexpr, BUCKETED: tl.constexpr,
                      HAS_SCALE: tl.constexpr, ADD_OUT: tl.constexpr, LOOP: tl.constexpr, ALIGNED: tl.constexpr,
                      OFF64: tl.constexpr):
    tile = tl.program_id(0)
    pid_s = tl.program_id(2)
    length = tl.load(tiles_ptr + tile * 5 + 2)
    out_col0 = tl.load(slice_ptr + pid_s * 5)
    n_cols = tl.load(slice_ptr + pid_s * 5 + 1)
    if ALIGNED:
        out_col0 = tl.multiple_of(out_col0, 16)
        n_cols = tl.multiple_of(n_cols, 16)
    col_base = tl.program_id(1) * cols_per_prog
    if length > 0:
        if col_base < n_cols:
            adapter = tl.load(tiles_ptr + tile * 5)
            start = tl.load(tiles_ptr + tile * 5 + 1)
            limit = tl.load(tiles_ptr + tile * 5 + 3)
            pos = start + tl.arange(0, BM)
            row_ok = tl.arange(0, BM) < length
            tok = _tile_tokens(tiles_ptr, tile, order_ptr, pos, row_ok, BM=BM)
            if limit == 0:
                if not ADD_OUT:
                    _ma_zero_part(out_ptr, tok, row_ok, col_base, cols_per_prog, n_cols, out_col0, OUT_STRIDE,
                                  BM=BM, BN=BN, LOOP=LOOP)
            else:
                ranks = tl.load(ranks_ptr + tok, mask=row_ok, other=0)
                b_off = tl.load(slice_ptr + pid_s * 5 + 3)
                b_ld = tl.load(slice_ptr + pid_s * 5 + 4)
                ws_s = ws_ptr + pid_s * WS_SLICE
                if BUCKETED:
                    if limit <= 16:
                        _ma_expand_part(ws_s, scale_ptr, b_pool, pt_ptr, out_ptr, tok, pos, row_ok, ranks, adapter,
                                        limit, T, OUT_STRIDE, PT_LD, FRAME_B, b_off, b_ld, out_col0, n_cols, SPLIT_K,
                                        WS_LD, SCALE_LD, col_base, cols_per_prog, BM=BM, BN=BN, R=16,
                                        HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT, LOOP=LOOP, ALIGNED=ALIGNED, OFF64=OFF64)
                    elif limit <= 32:
                        if R_PAD >= 32:
                            _ma_expand_part(ws_s, scale_ptr, b_pool, pt_ptr, out_ptr, tok, pos, row_ok, ranks,
                                            adapter, limit, T, OUT_STRIDE, PT_LD, FRAME_B, b_off, b_ld, out_col0,
                                            n_cols, SPLIT_K, WS_LD, SCALE_LD, col_base, cols_per_prog, BM=BM, BN=BN,
                                            R=32, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT, LOOP=LOOP, ALIGNED=ALIGNED, OFF64=OFF64)
                    elif limit <= 64:
                        if R_PAD >= 64:
                            _ma_expand_part(ws_s, scale_ptr, b_pool, pt_ptr, out_ptr, tok, pos, row_ok, ranks,
                                            adapter, limit, T, OUT_STRIDE, PT_LD, FRAME_B, b_off, b_ld, out_col0,
                                            n_cols, SPLIT_K, WS_LD, SCALE_LD, col_base, cols_per_prog, BM=BM, BN=BN,
                                            R=64, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT, LOOP=LOOP, ALIGNED=ALIGNED, OFF64=OFF64)
                    elif limit <= 128:
                        if R_PAD >= 128:
                            _ma_expand_part(ws_s, scale_ptr, b_pool, pt_ptr, out_ptr, tok, pos, row_ok, ranks,
                                            adapter, limit, T, OUT_STRIDE, PT_LD, FRAME_B, b_off, b_ld, out_col0,
                                            n_cols, SPLIT_K, WS_LD, SCALE_LD, col_base, cols_per_prog, BM=BM, BN=BN,
                                            R=128, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT, LOOP=LOOP, ALIGNED=ALIGNED, OFF64=OFF64)
                    else:
                        if R_PAD >= 256:
                            _ma_expand_part(ws_s, scale_ptr, b_pool, pt_ptr, out_ptr, tok, pos, row_ok, ranks,
                                            adapter, limit, T, OUT_STRIDE, PT_LD, FRAME_B, b_off, b_ld, out_col0,
                                            n_cols, SPLIT_K, WS_LD, SCALE_LD, col_base, cols_per_prog, BM=BM, BN=BN,
                                            R=256, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT, LOOP=LOOP, ALIGNED=ALIGNED, OFF64=OFF64)
                else:
                    _ma_expand_part(ws_s, scale_ptr, b_pool, pt_ptr, out_ptr, tok, pos, row_ok, ranks, adapter, limit,
                                    T, OUT_STRIDE, PT_LD, FRAME_B, b_off, b_ld, out_col0, n_cols, SPLIT_K, WS_LD,
                                    SCALE_LD, col_base, cols_per_prog, BM=BM, BN=BN, R=R_PAD, HAS_SCALE=HAS_SCALE,
                                    ADD_OUT=ADD_OUT, LOOP=LOOP, ALIGNED=ALIGNED, OFF64=OFF64)


# ------------------------------------------------------------------------------------------------
# Host side
# ------------------------------------------------------------------------------------------------


def _check(x: torch.Tensor, W: PagedWeights, token_ranks: torch.Tensor) -> None:
    """Layout checks only (never reads tensor values, so no host sync)."""
    if not x.is_cuda or x.ndim != 2 or x.shape[1] != W.h_in:
        raise ValueError(f"x must be a CUDA tensor of shape [T, {W.h_in}]")
    if x.dtype not in rf.SUPPORTED_DTYPES or W.dtype != x.dtype or W.B_pool.dtype != x.dtype:
        raise ValueError("x and the weight pools must share a supported floating dtype")
    if token_ranks.ndim != 1 or token_ranks.numel() != x.shape[0] or token_ranks.dtype not in (torch.int32, torch.int64):
        raise ValueError("token_ranks must be [T] int32 or int64")
    if not (x.is_contiguous() and W.A_pool.is_contiguous() and W.B_pool.is_contiguous() and token_ranks.is_contiguous()):
        raise ValueError("x, the pools and token_ranks must be contiguous")
    if W.page_table.dtype != torch.int32 or not W.page_table.is_contiguous():
        raise ValueError("page_table must be contiguous int32")


# Shrink slices for long reductions (h_in >= MULTI_SPLITK_MIN_H) in decode-sized batches (at most
# MULTI_SPLITK_MAX_T tokens): aim at MULTI_SPLITK_WAVES programs per SM, at most MULTI_MAX_SPLIT_K
# slices. With many adapter groups (tens to hundreds of tiles)
# rf1_fused's rule (one program per SM) leaves one long serial h_in loop per program; on an H200,
# 4 waves capped at 8 slices took decode with 64-256 adapters at h_in 4096-4864 from 25-38 us to
# 19-28 us. Below 2048 inputs the old rule was as fast or faster, and on prompt batches (tiles of
# up to 64 tokens) the extra slices cost ~8% at 2048-8192 tokens. Fitted on one H200 experiment
# (~/scratch/rankfuse/checks/multi/tune_splitk.py, results README).
MULTI_SPLITK_MIN_H = 2048
MULTI_SPLITK_MAX_T = 256
MULTI_SPLITK_WAVES = 4
MULTI_MAX_SPLIT_K = 8


def _split_k(h_in: int, bh: int, programs: int, tiled: bool, num_sms: int, split_k: int | None, T: int):
    if split_k is None:
        # rf1_fused's rule; with few tiles (one adapter) it already gives more slices than below
        split_k = max(rf.MIN_SPLIT_K_TILED if tiled else 1,
                      min(rf.MAX_SPLIT_K, rf.SPLITK_WAVES * num_sms // max(programs, 1)))
        if h_in >= MULTI_SPLITK_MIN_H and T <= MULTI_SPLITK_MAX_T:
            split_k = max(split_k, min(MULTI_MAX_SPLIT_K, MULTI_SPLITK_WAVES * num_sms // max(programs, 1)))
    k_per = triton.cdiv(triton.cdiv(h_in, split_k), bh) * bh  # whole BH chunks per slice
    return triton.cdiv(h_in, k_per), k_per


def _expand_geometry(T, widest, R, bm, programs, elem, dev, num_sms):
    """(BN, cols_per_prog, warps, LOOP), as rf1_fused picks them (section 8)."""
    if T > rf.EXPAND_TILED_MIN_TOKENS:
        bn = rf.EXPAND_TILED_BLOCK_N if bm <= 64 else 128
        while bn > 64 and R * bn * elem > 0.35 * rf._max_shared_mem(dev):
            bn //= 2
        return bn, bn, rf.EXPAND_TILED_WARPS, False
    bn = rf.EXPAND_BLOCK_N
    n_tiles = triton.cdiv(widest, bn)
    n_split = min(n_tiles, max(1, rf.EXPAND_WAVES * num_sms // max(programs, 1)))
    return bn, triton.cdiv(n_tiles, n_split) * bn, rf.EXPAND_WARPS, True


def _launch_pair(x, W, token_ranks, out, add, tiles, cap, order, ws, ws_ld, split_k, k_per, bm, bh, stages, R,
                 bucketed, n_slices, num_sms, dev):
    T, h_in = x.shape
    slice_tab = W.slice_table()
    aligned = W.aligned()
    off64 = max(W.A_pool.numel(), W.B_pool.numel()) >= 2**31
    ws_slice = split_k * T * ws_ld
    pt_ld = W.page_table.shape[1]
    _ma_shrink_kernel[(cap, split_k, n_slices)](
        x, W.A_pool, W.page_table, tiles, order, slice_tab, ws, T, h_in, pt_ld, W.frame_stride_a, k_per, ws_ld,
        ws_slice, BM=bm, BH=bh, R_PAD=R, BUCKETED=bucketed, ALIGNED=aligned, OFF64=off64, num_warps=4, num_stages=stages,
    )
    bn, cols, warps, loop = _expand_geometry(T, max(W.slice_sizes), R, bm, cap * n_slices, x.element_size(), dev, num_sms)
    scales = out if W.scale_table is None else W.scale_table
    _ma_expand_kernel[(cap, triton.cdiv(max(W.slice_sizes), cols), n_slices)](
        ws, token_ranks, scales, W.B_pool, W.page_table, tiles, order, slice_tab, out, T, out.stride(0), pt_ld,
        W.frame_stride_b, split_k, ws_ld, ws_slice, W.r_max + 1, cols,
        BM=bm, BN=bn, R_PAD=R, BUCKETED=bucketed, HAS_SCALE=W.scale_table is not None, ADD_OUT=add, LOOP=loop,
        ALIGNED=aligned, OFF64=off64, num_warps=warps, num_stages=2,
    )


def multi_lora(x: torch.Tensor, W: PagedWeights, token_ranks: torch.Tensor, meta: MultiMeta, *,
               out: torch.Tensor | None = None, add: bool = False, bucketed: bool | None = None,
               split_k: int | None = None) -> torch.Tensor:
    """out (+)= the multi-adapter delta over meta's tile table (modes: see the module docstring).

    bucketed: per-tile rank buckets (default: True for modes "rank_sorted" and "segment").
    """
    _check(x, W, token_ranks)
    out = output_buffer(x, W, out, add)
    T, h_in = x.shape
    if T == 0:
        return out
    if meta.T != T:
        raise ValueError(f"metadata built for {meta.T} tokens, x has {T}")
    if bucketed is None:
        bucketed = meta.mode != "group"
    dev = x.device.index or 0
    num_sms = rf._num_sms(dev)
    S, cap, bm, R = W.n_slices, meta.capacity, meta.bm, meta.r_pad
    bh, stages = rf._splitk_shrink_tiles(bm, R, x.element_size(), dev)
    split_k, k_per = _split_k(h_in, bh, cap * S, T > rf.EXPAND_TILED_MIN_TOKENS, num_sms, split_k, T)
    ws = torch.empty((S, split_k, T, R), dtype=torch.float32, device=x.device)
    _launch_pair(x, W, token_ranks, out, add, meta.tiles, cap, meta.order, ws, R, split_k, k_per, bm, bh, stages, R,
                 bucketed, S, num_sms, dev)
    return out


# ------------------------------------------------------------------------------------------------
# RF1-MA on the shared split-K kernels (rankfuse_fused.py, MULTI mode, section 9)
# ------------------------------------------------------------------------------------------------

# Apply the many-adapter split-K rule above (MULTI_SPLITK_*) on top of rankfuse_fused._split_count.
# Fitted on the H200 for the legacy kernels; kept until the A100 sweep shows where it helps.
MULTI_SPLIT_RULE = True


def multi_tiles(W: PagedWeights, tiles: torch.Tensor, order: torch.Tensor) -> rf.MultiTiles:
    """A tile table, its token order and the weight layout, as splitk_pass's multi mode takes them."""
    return rf.MultiTiles(
        tiles=tiles, order=order, page_table=W.page_table, slice_table=W.slice_table(),
        pt_ld=W.page_table.shape[1], frame_a=W.frame_stride_a, frame_b=W.frame_stride_b, scale_ld=W.r_max + 1,
        n_slices=W.n_slices, aligned=W.aligned(), off64=max(W.A_pool.numel(), W.B_pool.numel()) >= 2**31,
    )


def multi_split_k(h_in: int, T: int, tiled: bool, programs: int, num_sms: int) -> int | None:
    """The slice count the many-adapter rule asks for, or None to leave it to _split_count."""
    if not (MULTI_SPLIT_RULE and h_in >= MULTI_SPLITK_MIN_H and T <= MULTI_SPLITK_MAX_T):
        return None
    base = max(rf.MIN_SPLIT_K_TILED if tiled else 1, min(rf.MAX_SPLIT_K, rf.SPLITK_WAVES * num_sms // max(programs, 1)))
    return max(base, min(MULTI_MAX_SPLIT_K, MULTI_SPLITK_WAVES * num_sms // max(programs, 1)))


def multi_lora_splitk(x: torch.Tensor, W: PagedWeights, token_ranks: torch.Tensor, meta: MultiMeta, *,
                      out: torch.Tensor | None = None, add: bool = False, rank_skip: bool = False,
                      full_width: bool = False, full_rows: bool = False, split_k: int | None = None,
                      table_width: bool = False) -> torch.Tensor:
    """out (+)= the multi-adapter delta over meta's tile table, on rankfuse_fused.splitk_pass (MULTI).

    Launch settings: _config at meta.r_pad (warps, BH, stages, tiled-expand BN / warps, column-loop
    expand); the token block is meta.bm, since the tile table holds at most bm tokens per tile.
    Every tile reads its tokens' ranks from token_ranks when it runs (also at CUDA-graph replay) and
    loads only rank rows below their maximum (fewer bytes). rank_skip: each tile also runs at that
    maximum's width bucket (RF2-MA: less MMA work). full_width: every tile at meta.r_pad through the
    same binary (no MMA saving); full_rows: every tile loads all W.r_max rank rows (no byte saving;
    needs every page resident). full_width + full_rows is the no-skipping control, so
        bytes saved = t(full_width + full_rows) - t(full_width);  MMA saved = t(full_width) - t(rank_skip).
    table_width: take each tile's width from the tile table's limit column instead of its tokens' ranks
    (one dependent load fewer per program; a captured graph then follows rank changes only through
    MultiMeta.load_, which a serving engine calls every step anyway).
    """
    _check(x, W, token_ranks)
    out = output_buffer(x, W, out, add)
    T, h_in = x.shape
    if T == 0:
        return out
    if meta.T != T:
        raise ValueError(f"metadata built for {meta.T} tokens, x has {T}")
    if (full_width or full_rows) and not rank_skip:
        raise ValueError("full_width / full_rows are controls for rank_skip; they need rank_skip=True")
    dev = x.device.index or 0
    num_sms = rf._num_sms(dev)
    S, cap, bm, R = W.n_slices, meta.capacity, meta.bm, meta.r_pad
    cfg = rf._config(R, T, dev, bm=bm)
    bh, stages = rf._splitk_shrink_tiles(bm, R, x.element_size(), dev, cfg["bh"], cfg["stages"])
    tiled = cfg["tiled"]
    if split_k is None:
        split_k = multi_split_k(h_in, T, tiled, cap * S, num_sms)
    split_k, k_per = rf._split_count(h_in, tiled, cap, S, num_sms, bh, split_k)
    ws = torch.empty((S, split_k, T, R), dtype=torch.float32, device=x.device)
    return rf.splitk_pass(x, W.A_pool, W.B_pool, token_ranks, out, add, W.slice_sizes, W.scale_table, ws, R, R, split_k,
                          k_per, bm, bh, stages, rank_skip=rank_skip, force_rank=R if full_width else 0,
                          block_n=cfg["bn"], shrink_warps=cfg["shrink_warps"], expand_warps=cfg["expand_warps"],
                          tiled=tiled, multi=multi_tiles(W, meta.tiles, meta.order), mask_floor=W.r_max if full_rows else 0,
                          token_ranks_width=not table_width)


def multi_lora_bucketed_splitk(x, W: PagedWeights, token_ranks, plan: "BucketPlan", *, out=None, add=False,
                               concurrent=True, table_width: bool = False):
    """RF2-MA, per-width group launches (rf2_bucketed_tuned's strategy) on the shared kernels (MULTI).

    One splitk_pass per rank bucket R present, at exactly R rank rows (no runtime branch), over that
    bucket's tiles (tokens grouped by adapter inside the bucket), the buckets' pairs on concurrent
    streams (fork/join, capturable). Each bucket's launch settings are _config at its own width,
    with the plan's token block (BucketPlan.bm) and rf2_bucketed_tuned's tiled-expand BN; one split
    count for all buckets (they share the workspace), as rf2_bucketed does."""
    _check(x, W, token_ranks)
    out = output_buffer(x, W, out, add)
    T, h_in = x.shape
    if T == 0:
        return out
    if plan.zero_idx.numel() and not add:
        out.index_fill_(0, plan.zero_idx, 0)
    if not plan.buckets:
        return out
    dev = x.device.index or 0
    num_sms = rf._num_sms(dev)
    S, elem = W.n_slices, x.element_size()
    groups = []
    for b in plan.buckets:  # heaviest first
        cfg = rf._config(b["R"], T, dev, bm=b["bm"], bn=256 if b["bm"] <= 64 else 128)
        bh, stages = rf._splitk_shrink_tiles(b["bm"], b["R"], elem, dev, cfg["bh"], cfg["stages"])
        groups.append((b, cfg, bh, stages))
    grid_tiles = sum(b["cap"] for b in plan.buckets)
    bh_min = min(g[2] for g in groups)
    tiled = any(g[1]["tiled"] for g in groups)
    split_k = multi_split_k(h_in, T, tiled, grid_tiles * S, num_sms)
    split_k, k_per = rf._split_count(h_in, tiled, grid_tiles, S, num_sms, bh_min, split_k)  # k_per: a multiple of bh_min
    ws = torch.empty((S, split_k, T, plan.r_pad), dtype=torch.float32, device=x.device)

    def launch(b, cfg, bh, stages):
        # tile positions are global (one order for every bucket), so every bucket uses ws_row0 = 0
        rf.splitk_pass(x, W.A_pool, W.B_pool, token_ranks, out, add, W.slice_sizes, W.scale_table, ws, plan.r_pad,
                       plan.r_pad, split_k, k_per, b["bm"], bh if k_per % bh == 0 else bh_min, stages, grid_tiles=grid_tiles,
                       rank=b["R"], block_n=cfg["bn"], shrink_warps=cfg["shrink_warps"], expand_warps=cfg["expand_warps"],
                       tiled=cfg["tiled"], multi=multi_tiles(W, b["tiles"], plan.order), token_ranks_width=not table_width)

    if concurrent and len(groups) > 1:
        main = torch.cuda.current_stream()
        streams = []
        for g in groups:
            st = _stream(x.device, g[0]["R"])
            st.wait_stream(main)
            with torch.cuda.stream(st):
                launch(*g)
            streams.append(st)
        for st in streams:
            main.wait_stream(st)
    else:
        for g in groups:
            launch(*g)
    return out


# ------------------------------------------------------------------------------------------------
# rf2_ma_bucketed: a per-step CPU plan, one kernel pair per rank bucket
# ------------------------------------------------------------------------------------------------


def bucket_of(r: int) -> int:
    return 0 if r <= 0 else max(16, triton.next_power_of_2(int(r)))


class BucketPlan:
    """Tokens split by rank bucket (16/32/64/...), each bucket's tokens grouped by adapter into
    tiles. One global order (bucket, then adapter) so every bucket owns a disjoint range of ws
    positions. Heaviest bucket (tokens x R) first. capacities: optional {R: tile capacity} so a
    captured graph's grids stay fixed (every later step must then have the same bucket set)."""

    def __init__(self, token_adapter, token_ranks, num_adapters: int, device="cuda", capacities=None):
        ad = torch.as_tensor(token_adapter).to("cpu", torch.int64).flatten()
        rk = torch.as_tensor(token_ranks).to("cpu", torch.int64).flatten()
        self.T = rk.numel()
        none = (rk == 0) | (ad < 0)
        rk = torch.where(none, torch.zeros_like(rk), rk)
        pow2 = torch.ceil(torch.log2(rk.clamp(min=1).double())).long()
        buckets = torch.where(rk > 0, torch.clamp(1 << pow2, min=16), 0)
        self.r_pad = int(buckets.max()) if self.T and int(buckets.max()) > 0 else 16
        live = buckets > 0
        # rank rows the bucketed kernels multiply relative to rf1_ma (every tile at the batch's bucket)
        self.work_ratio = float(buckets[live].sum()) / (int(live.sum()) * self.r_pad) if live.any() else 1.0
        self.zero_idx = torch.nonzero(none).flatten().to(device)
        order_parts, self.buckets, base = [], [], 0
        present = sorted(torch.unique(buckets[buckets > 0]).tolist())
        for R in present:
            idx = torch.nonzero(buckets == R).flatten()
            groups = int(torch.unique(ad[idx]).numel())
            bm = default_block_m(self.T, groups, R)
            if self.T > 128 and R <= 32 and idx.numel() / max(groups, 1) > 64:
                bm = 128
            cap = (capacities or {}).get(R)
            m = build_meta(ad[idx], rk[idx], num_adapters=num_adapters, mode="group", device="cpu", bm=bm, r_pad=R,
                           capacity=cap)
            tiles = m.tiles.clone()
            tiles[:, 1] += base
            tiles[:, 4] = -1  # set below from the global order
            order_parts.append(idx[m.order.long()])
            self.buckets.append(dict(R=R, bm=bm, tiles=tiles, cap=m.capacity, count=idx.numel()))
            base += idx.numel()
        self.buckets.sort(key=lambda b: -b["count"] * b["R"])  # heaviest first
        rest = torch.nonzero(buckets == 0).flatten()
        order = torch.cat(order_parts + [rest]) if order_parts else rest
        for b in self.buckets:
            b["tiles"] = mark_consecutive(b["tiles"], order).to(device)
        self.order = order.to(torch.int32).to(device)


_STREAMS: dict[tuple, torch.cuda.Stream] = {}


def _stream(device, R):
    key = (device, R)
    if key not in _STREAMS:
        _STREAMS[key] = torch.cuda.Stream(device=device)
    return _STREAMS[key]


def multi_lora_bucketed(x, W: PagedWeights, token_ranks, plan: BucketPlan, *, out=None, add=False, concurrent=True):
    _check(x, W, token_ranks)
    out = output_buffer(x, W, out, add)
    T, h_in = x.shape
    if T == 0:
        return out
    if plan.zero_idx.numel() and not add:
        out.index_fill_(0, plan.zero_idx, 0)
    if not plan.buckets:
        return out
    dev = x.device.index or 0
    num_sms = rf._num_sms(dev)
    S, elem = W.n_slices, x.element_size()
    tiled = T > rf.EXPAND_TILED_MIN_TOKENS
    programs = sum(b["cap"] for b in plan.buckets) * S
    shapes = {b["R"]: rf._splitk_shrink_tiles(b["bm"], b["R"], elem, dev) for b in plan.buckets}
    bh_min = min(bh for bh, _ in shapes.values())
    split_k, k_per = _split_k(h_in, bh_min, programs, tiled, num_sms, None, T)  # k_per: a multiple of every BH used
    ws_ld = plan.r_pad
    ws = torch.empty((S, split_k, T, ws_ld), dtype=torch.float32, device=x.device)

    def launch(b):
        bh, stages = shapes[b["R"]]
        if k_per % bh:
            bh = bh_min
        _launch_pair(x, W, token_ranks, out, add, b["tiles"], b["cap"], plan.order, ws, ws_ld, split_k, k_per,
                     b["bm"], bh, stages, b["R"], False, S, num_sms, dev)

    if concurrent and len(plan.buckets) > 1:
        main = torch.cuda.current_stream()
        streams = []
        for b in plan.buckets:
            s = _stream(x.device, b["R"])
            s.wait_stream(main)
            with torch.cuda.stream(s):
                launch(b)
            streams.append(s)
        for s in streams:
            main.wait_stream(s)
    else:
        for b in plan.buckets:
            launch(b)
    return out


# ------------------------------------------------------------------------------------------------
# Registry entries
# ------------------------------------------------------------------------------------------------


def _meta_prepare(mode: str):
    def prepare(x, W, token_adapter, token_ranks) -> dict:
        return dict(meta=build_meta(token_adapter, token_ranks, num_adapters=W.num_adapters, mode=mode, device=x.device))

    return prepare


def _meta_impl(mode: str, doc: str):
    prepare = _meta_prepare(mode)

    def fn(x, W, token_adapter, token_ranks, meta: MultiMeta | None = None, out=None, add=False):
        if meta is None:  # one host round trip; a serving engine builds meta once per step
            meta = prepare(x, W, token_adapter, token_ranks)["meta"]
        return multi_lora(x, W, token_ranks, meta, out=out, add=add)

    fn.__doc__ = doc
    return fn, prepare


@register_multi("rf1_ma_table", prepare=_meta_prepare("group"))
def rf1_ma_table(x, W, token_adapter, token_ranks, meta: MultiMeta | None = None, out=None, add=False):
    """rf1_ma with each tile's rank rows masked at the tile table's limit (its group's maximum rank) instead
    of its tokens' ranks read at run time: one dependent load fewer per program (2c30e56's behaviour)."""
    if meta is None:
        meta = _meta_prepare("group")(x, W, token_adapter, token_ranks)["meta"]
    return multi_lora_splitk(x, W, token_ranks, meta, out=out, add=add, table_width=True)


@register_multi("rf1_ma", prepare=_meta_prepare("group"))
def rf1_ma(x, W, token_adapter, token_ranks, meta: MultiMeta | None = None, out=None, add=False):
    """RF1-MA on the shared split-K kernels (rankfuse_fused.splitk_pass, MULTI): tiles inside adapter
    groups, every tile at its group's rank limit (masked loads), launch settings from _config."""
    if meta is None:  # one host round trip; a serving engine builds meta once per step
        meta = _meta_prepare("group")(x, W, token_adapter, token_ranks)["meta"]
    return multi_lora_splitk(x, W, token_ranks, meta, out=out, add=add)


def _skip_impl(mode: str, full_width: bool, doc: str, full_rows: bool = False, table_width: bool = False):
    prepare = _meta_prepare(mode)

    def fn(x, W, token_adapter, token_ranks, meta: MultiMeta | None = None, out=None, add=False):
        if meta is None:
            meta = prepare(x, W, token_adapter, token_ranks)["meta"]
        return multi_lora_splitk(x, W, token_ranks, meta, out=out, add=add, rank_skip=True, full_width=full_width,
                                 full_rows=full_rows, table_width=table_width)

    fn.__doc__ = doc
    return fn, prepare


# RF2-MA strategy (i): RANK_SKIP inside multi-adapter tiles, on the shared kernels.
for _name, _mode, _fw, _doc in (
    ("rf2_ma_splitk", "rank_sorted", False, "RF2-MA: sort by (adapter, rank); each tile at its own width bucket (RANK_SKIP)."),
    ("rf2_ma_splitk_fw", "rank_sorted", True,
     "rf2_ma_splitk's binary with every tile at the batch width: rank rows still masked per tile (bytes saved, no MMA saved)."),
    ("rf2_ma_widths", "adapter_bucket", False,
     "RF2-MA: groups of (adapter, width bucket), so no tile spans two widths; each tile at its own bucket."),
    ("rf2_ma_segment", "segment", False, "RF2-MA segment: batch order, tiles = runs of one adapter, RANK_SKIP."),
):
    _fn, _prep = _skip_impl(_mode, _fw, _doc)
    register_multi(_name, prepare=_prep)(_fn)

_fn, _prep = _skip_impl("rank_sorted", False, "rf2_ma_splitk with each tile's width from the tile table's limit "
                        "column instead of its tokens' ranks (follows rank changes only through MultiMeta.load_).",
                        table_width=True)
register_multi("rf2_ma_splitk_table", prepare=_prep)(_fn)

_fn, _prep = _skip_impl("rank_sorted", True, "rf2_ma_splitk's binary with no skipping at all: every tile at the batch width "
                        "and every rank row loaded (the control for bytes vs MMA; needs every page resident).", full_rows=True)
register_multi("rf2_ma_splitk_full", prepare=_prep, check=lambda W: None if bool((W.page_table >= 0).all())
               else "needs every page resident")(_fn)


for _name, _mode, _doc in (
    ("rf1_ma_legacy", "group", "RF1-MA on the Stage 2 kernels (_ma_*), kept for comparison with rf1_ma."),
    ("rf2_ma_splitk_legacy", "rank_sorted", "rf2_ma_splitk on the Stage 2 kernels (_ma_*, BUCKETED)."),
    ("rf2_ma_segment_legacy", "segment", "rf2_ma_segment on the Stage 2 kernels (_ma_*, BUCKETED)."),
):
    _fn, _prep = _meta_impl(_mode, _doc)
    register_multi(_name, prepare=_prep)(_fn)


def _bucketed_prepare(x, W, token_adapter, token_ranks) -> dict:
    return dict(plan=BucketPlan(token_adapter, token_ranks, W.num_adapters, x.device))


@register_multi("rf2_ma_bucketed", prepare=_bucketed_prepare)
def rf2_ma_bucketed(x, W, token_adapter, token_ranks, plan: BucketPlan | None = None, out=None, add=False,
                    concurrent=True):
    """RF2-MA strategy (ii): one shared-kernel launch per rank width (tiles grouped by adapter inside
    each width), the widths on concurrent streams, each at its own _config settings."""
    if plan is None:
        plan = _bucketed_prepare(x, W, token_adapter, token_ranks)["plan"]
    return multi_lora_bucketed_splitk(x, W, token_ranks, plan, out=out, add=add, concurrent=concurrent)


@register_multi("rf2_ma_bucketed_seq", prepare=_bucketed_prepare)
def rf2_ma_bucketed_seq(x, W, token_adapter, token_ranks, plan: BucketPlan | None = None, out=None, add=False):
    """rf2_ma_bucketed with the widths launched one after another on one stream."""
    if plan is None:
        plan = _bucketed_prepare(x, W, token_adapter, token_ranks)["plan"]
    return multi_lora_bucketed_splitk(x, W, token_ranks, plan, out=out, add=add, concurrent=False)


@register_multi("rf2_ma_bucketed_table", prepare=_bucketed_prepare)
def rf2_ma_bucketed_table(x, W, token_adapter, token_ranks, plan: BucketPlan | None = None, out=None, add=False):
    """rf2_ma_bucketed with rank rows masked at the tile table's limit instead of the tokens' ranks."""
    if plan is None:
        plan = _bucketed_prepare(x, W, token_adapter, token_ranks)["plan"]
    return multi_lora_bucketed_splitk(x, W, token_ranks, plan, out=out, add=add, table_width=True)


@register_multi("rf2_ma_bucketed_legacy", prepare=_bucketed_prepare)
def rf2_ma_bucketed_legacy(x, W, token_adapter, token_ranks, plan: BucketPlan | None = None, out=None, add=False,
                           concurrent=True):
    """rf2_ma_bucketed on the Stage 2 kernels (_ma_*), kept for comparison."""
    if plan is None:
        plan = _bucketed_prepare(x, W, token_adapter, token_ranks)["plan"]
    return multi_lora_bucketed(x, W, token_ranks, plan, out=out, add=add, concurrent=concurrent)


# ------------------------------------------------------------------------------------------------
# Selector
# ------------------------------------------------------------------------------------------------

# Fitted on the same H200 sweep it is evaluated on (results/h200_2026-11-03_multi, table 10): over
# 1050 batches it gains 0.4% (geometric mean) over always running rf1_ma, the per-batch oracle 2.2%,
# and its worst pick is 0.76x. rf1_ma alone is the recommended default; this exists to show the
# size of the RF2 opportunity in Stage 2, as use_rf2 does for Stage 1.
BUCKETED_MAX_WORK_RATIO = 0.7
BUCKETED_MIN_H_IN = 2048


def use_bucketed(plan: BucketPlan, h_in: int) -> bool:
    """CPU-side choice: rf2_ma_bucketed when its rank work is <= 70% of rf1_ma's on a long reduction."""
    return plan.work_ratio <= BUCKETED_MAX_WORK_RATIO and h_in >= BUCKETED_MIN_H_IN


def _auto_prepare(x, W, token_adapter, token_ranks) -> dict:
    plan = BucketPlan(token_adapter, token_ranks, W.num_adapters, x.device)
    if use_bucketed(plan, W.h_in):
        return dict(plan=plan)
    return _meta_prepare("group")(x, W, token_adapter, token_ranks)


@register_multi("rf_ma_auto", prepare=_auto_prepare)
def rf_ma_auto(x, W, token_adapter, token_ranks, meta: MultiMeta | None = None, plan: BucketPlan | None = None,
               out=None, add=False):
    """rf1_ma or rf2_ma_bucketed, chosen per step by use_bucketed (from the prepared metadata).
    The selector was fitted with the legacy rf1_ma (H200, results/h200_2026-11-03_multi)."""
    if meta is None and plan is None:
        prepared = _auto_prepare(x, W, token_adapter, token_ranks)
        meta, plan = prepared.get("meta"), prepared.get("plan")
    if plan is not None:
        return multi_lora_bucketed_splitk(x, W, token_ranks, plan, out=out, add=add)
    return multi_lora_splitk(x, W, token_ranks, meta, out=out, add=add)
