"""RF2, sorted split-K (`rf2_splitk`): rf1_fused's split-K design plus rf2_fused's rank skipping.

    host     order = argsort(token_ranks, stable): token blocks hold tokens of similar rank (a
             serving engine can build it once per step and pass it in)
    kernel 1 grid (m_tiles, S): program (m, k) takes block m of the sorted order, picks the
             smallest rank bucket R in 16/32/64/128/256 covering the block's maximum rank, and
             reduces h_in slice k over R rank rows only; partial v -> ws[k, pos, :R] (fp32)
    kernel 2 the same block and bucket: sum the S partials over R columns, mask per token, scale,
             expand with R rows of B, write (or add into) each token's own output row; blocks
             whose ranks are all 0 write zeros (or nothing with add=True). Up to 128 tokens a
             column strip per program; above, one tile per program and no loop.

Slice count, token block size, tile sizes and shared-memory guards come from rankfuse_fused's
helpers, so launch shapes match rf1_fused except for R.

Measured on an H200 (4096 x 4096, r_max 64): within -6% .. +15% of rf1_fused. Sorting scatters
the loads and costs 6-46 us per batch, which cancels most of the rank skipping. Kept as an
ablation; rf2_segment and rf2_bucketed remove the sort.
"""

import torch
import triton
import triton.language as tl

from rankfuse.interface import MatryoshkaLoRA, register
from rankfuse.kernels import rankfuse_fused as rf
from rankfuse.kernels._common import check_layout, prepare_rank_order


@triton.jit
def _rf2s_shrink_part(x_ptr, a_ptr, ws_k, tok, pos, row_ok, H_IN, r_limit, h_lo, K_PER,
                      BM: tl.constexpr, BH: tl.constexpr, R: tl.constexpr, R_PAD: tl.constexpr):
    k = tl.arange(0, R)
    k_ok = k < r_limit
    v = tl.zeros((BM, R), dtype=tl.float32)
    for h_start in range(0, K_PER, BH):
        h = h_lo + h_start + tl.arange(0, BH)
        h_ok = h < H_IN
        x = tl.load(x_ptr + tok[:, None] * H_IN + h[None, :], mask=row_ok[:, None] & h_ok[None, :], other=0.0)
        a = tl.load(a_ptr + k[:, None] * H_IN + h[None, :], mask=k_ok[:, None] & h_ok[None, :], other=0.0)
        v += tl.dot(x, tl.trans(a), input_precision="ieee")
    tl.store(ws_k + pos[:, None] * R_PAD + k[None, :], v, mask=row_ok[:, None])


@triton.jit
def _rf2s_shrink_kernel(x_ptr, a_ptr, ranks_ptr, order_ptr, ws_ptr, T, H_IN, r_limit, K_PER,
                        BM: tl.constexpr, BH: tl.constexpr, R_PAD: tl.constexpr):
    pos = tl.program_id(0) * BM + tl.arange(0, BM)
    row_ok = pos < T
    tok = tl.load(order_ptr + pos, mask=row_ok, other=0)
    ranks = tl.load(ranks_ptr + tok, mask=row_ok, other=0)
    tile_rank = tl.max(ranks, axis=0)
    h_lo = tl.program_id(1) * K_PER
    ws_k = ws_ptr + tl.program_id(1) * T * R_PAD
    if tile_rank > 0:
        if tile_rank <= 16:
            _rf2s_shrink_part(x_ptr, a_ptr, ws_k, tok, pos, row_ok, H_IN, r_limit, h_lo, K_PER, BM=BM, BH=BH, R=16, R_PAD=R_PAD)
        elif tile_rank <= 32:
            if R_PAD >= 32:
                _rf2s_shrink_part(x_ptr, a_ptr, ws_k, tok, pos, row_ok, H_IN, r_limit, h_lo, K_PER, BM=BM, BH=BH, R=32, R_PAD=R_PAD)
        elif tile_rank <= 64:
            if R_PAD >= 64:
                _rf2s_shrink_part(x_ptr, a_ptr, ws_k, tok, pos, row_ok, H_IN, r_limit, h_lo, K_PER, BM=BM, BH=BH, R=64, R_PAD=R_PAD)
        elif tile_rank <= 128:
            if R_PAD >= 128:
                _rf2s_shrink_part(x_ptr, a_ptr, ws_k, tok, pos, row_ok, H_IN, r_limit, h_lo, K_PER, BM=BM, BH=BH, R=128, R_PAD=R_PAD)
        else:
            if R_PAD >= 256:
                _rf2s_shrink_part(x_ptr, a_ptr, ws_k, tok, pos, row_ok, H_IN, r_limit, h_lo, K_PER, BM=BM, BH=BH, R=256, R_PAD=R_PAD)


@triton.jit
def _rf2s_store_tile(v_in, b_ptr, out_ptr, tok, row_ok, k, k_ok, col_start, N, OUT_STRIDE,
                     BN: tl.constexpr, ADD_OUT: tl.constexpr):
    cols = col_start + tl.arange(0, BN)
    col_ok = cols < N
    b = tl.load(b_ptr + k[:, None] * N + cols[None, :], mask=k_ok[:, None] & col_ok[None, :], other=0.0)
    acc = tl.dot(v_in, b, input_precision="ieee")
    ptrs = out_ptr + tok[:, None] * OUT_STRIDE + cols[None, :]
    mask = row_ok[:, None] & col_ok[None, :]
    if ADD_OUT:
        acc += tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    tl.store(ptrs, acc, mask=mask)


@triton.jit
def _rf2s_expand_part(ws_ptr, scale_ptr, b_ptr, out_ptr, tok, pos, row_ok, ranks, T, N, OUT_STRIDE, r_limit, SPLIT_K,
                      col_base, cols_per_prog, BM: tl.constexpr, BN: tl.constexpr, R: tl.constexpr, R_PAD: tl.constexpr,
                      HAS_SCALE: tl.constexpr, ADD_OUT: tl.constexpr, LOOP: tl.constexpr):
    k = tl.arange(0, R)
    k_ok = k < r_limit
    v = tl.zeros((BM, R), dtype=tl.float32)
    for s in range(0, SPLIT_K):
        v += tl.load(ws_ptr + s * T * R_PAD + pos[:, None] * R_PAD + k[None, :], mask=row_ok[:, None], other=0.0)
    v = tl.where(k[None, :] < ranks[:, None], v, 0.0)
    if HAS_SCALE:
        v = v * tl.load(scale_ptr + ranks, mask=row_ok, other=0.0)[:, None]
    v_in = v.to(out_ptr.dtype.element_ty)
    if LOOP:
        for col_start in range(0, cols_per_prog, BN):
            _rf2s_store_tile(v_in, b_ptr, out_ptr, tok, row_ok, k, k_ok, col_base + col_start, N, OUT_STRIDE, BN=BN, ADD_OUT=ADD_OUT)
    else:
        _rf2s_store_tile(v_in, b_ptr, out_ptr, tok, row_ok, k, k_ok, col_base, N, OUT_STRIDE, BN=BN, ADD_OUT=ADD_OUT)


@triton.jit
def _rf2s_zero_part(out_ptr, tok, row_ok, N, OUT_STRIDE, col_base, cols_per_prog,
                    BM: tl.constexpr, BN: tl.constexpr, LOOP: tl.constexpr):
    zeros = tl.zeros((BM, BN), dtype=out_ptr.dtype.element_ty)
    if LOOP:
        for col_start in range(0, cols_per_prog, BN):
            cols = col_base + col_start + tl.arange(0, BN)
            tl.store(out_ptr + tok[:, None] * OUT_STRIDE + cols[None, :], zeros, mask=row_ok[:, None] & (cols[None, :] < N))
    else:
        cols = col_base + tl.arange(0, BN)
        tl.store(out_ptr + tok[:, None] * OUT_STRIDE + cols[None, :], zeros, mask=row_ok[:, None] & (cols[None, :] < N))


@triton.jit
def _rf2s_expand_kernel(ws_ptr, ranks_ptr, order_ptr, scale_ptr, b_ptr, out_ptr, T, N, OUT_STRIDE, r_limit, SPLIT_K,
                        cols_per_prog, BM: tl.constexpr, BN: tl.constexpr, R_PAD: tl.constexpr,
                        HAS_SCALE: tl.constexpr, ADD_OUT: tl.constexpr, LOOP: tl.constexpr):
    pos = tl.program_id(0) * BM + tl.arange(0, BM)
    row_ok = pos < T
    tok = tl.load(order_ptr + pos, mask=row_ok, other=0)
    ranks = tl.load(ranks_ptr + tok, mask=row_ok, other=0)
    tile_rank = tl.max(ranks, axis=0)
    col_base = tl.program_id(1) * cols_per_prog
    if tile_rank == 0:
        if not ADD_OUT:
            _rf2s_zero_part(out_ptr, tok, row_ok, N, OUT_STRIDE, col_base, cols_per_prog, BM=BM, BN=BN, LOOP=LOOP)
    elif tile_rank <= 16:
        _rf2s_expand_part(ws_ptr, scale_ptr, b_ptr, out_ptr, tok, pos, row_ok, ranks, T, N, OUT_STRIDE, r_limit, SPLIT_K,
                          col_base, cols_per_prog, BM=BM, BN=BN, R=16, R_PAD=R_PAD, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT, LOOP=LOOP)
    elif tile_rank <= 32:
        if R_PAD >= 32:
            _rf2s_expand_part(ws_ptr, scale_ptr, b_ptr, out_ptr, tok, pos, row_ok, ranks, T, N, OUT_STRIDE, r_limit, SPLIT_K,
                              col_base, cols_per_prog, BM=BM, BN=BN, R=32, R_PAD=R_PAD, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT, LOOP=LOOP)
    elif tile_rank <= 64:
        if R_PAD >= 64:
            _rf2s_expand_part(ws_ptr, scale_ptr, b_ptr, out_ptr, tok, pos, row_ok, ranks, T, N, OUT_STRIDE, r_limit, SPLIT_K,
                              col_base, cols_per_prog, BM=BM, BN=BN, R=64, R_PAD=R_PAD, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT, LOOP=LOOP)
    elif tile_rank <= 128:
        if R_PAD >= 128:
            _rf2s_expand_part(ws_ptr, scale_ptr, b_ptr, out_ptr, tok, pos, row_ok, ranks, T, N, OUT_STRIDE, r_limit, SPLIT_K,
                              col_base, cols_per_prog, BM=BM, BN=BN, R=128, R_PAD=R_PAD, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT, LOOP=LOOP)
    else:
        if R_PAD >= 256:
            _rf2s_expand_part(ws_ptr, scale_ptr, b_ptr, out_ptr, tok, pos, row_ok, ranks, T, N, OUT_STRIDE, r_limit, SPLIT_K,
                              col_base, cols_per_prog, BM=BM, BN=BN, R=256, R_PAD=R_PAD, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT, LOOP=LOOP)


def rf2_splitk(x, A_t, B, token_ranks, r_limit, order=None, out=None, add=False, scale_table=None):
    """Sorted split-K RF2 for one adapter: out (+)= delta. A_t [r_max, h_in], B [r_max, h_out]."""
    T, h_in = x.shape
    n = B.shape[1]
    if out is None:
        if add:
            raise ValueError("add=True needs an existing out to add into")
        out = torch.empty((T, n), dtype=x.dtype, device=x.device)
    if T == 0:
        return out
    if order is None:
        order = torch.argsort(token_ranks, stable=True)
    r_pad = max(16, triton.next_power_of_2(max(r_limit, 1)))
    dev = x.device.index or 0
    num_sms = rf._num_sms(dev)
    block_m = rf._block_m(T, r_pad)
    m_tiles = triton.cdiv(T, block_m)
    bh, stages = rf._splitk_shrink_tiles(block_m, r_pad, x.element_size(), dev)
    tiled = T > rf.EXPAND_TILED_MIN_TOKENS
    split_k = max(rf.MIN_SPLIT_K_TILED if tiled else 1, min(rf.MAX_SPLIT_K, rf.SPLITK_WAVES * num_sms // m_tiles))
    k_per = triton.cdiv(triton.cdiv(h_in, split_k), bh) * bh
    split_k = triton.cdiv(h_in, k_per)
    ws = torch.empty((split_k, T, r_pad), dtype=torch.float32, device=x.device)
    _rf2s_shrink_kernel[(m_tiles, split_k)](x, A_t, token_ranks, order, ws, T, h_in, r_limit, k_per,
                                            BM=block_m, BH=bh, R_PAD=r_pad, num_warps=4, num_stages=stages)
    if tiled:
        block_n = rf.EXPAND_TILED_BLOCK_N
        while block_n > 64 and r_pad * block_n * x.element_size() > 0.35 * rf._max_shared_mem(dev):
            block_n //= 2
        cols_per_prog, warps = block_n, rf.EXPAND_TILED_WARPS
    else:
        block_n, warps = rf.EXPAND_BLOCK_N, rf.EXPAND_WARPS
        n_tiles = triton.cdiv(n, block_n)
        n_split = min(n_tiles, max(1, rf.EXPAND_WAVES * num_sms // m_tiles))
        cols_per_prog = triton.cdiv(n_tiles, n_split) * block_n
    _rf2s_expand_kernel[(m_tiles, triton.cdiv(n, cols_per_prog))](
        ws, token_ranks, order, out if scale_table is None else scale_table, B, out, T, n, out.stride(0), r_limit,
        split_k, cols_per_prog, BM=block_m, BN=block_n, R_PAD=r_pad, HAS_SCALE=scale_table is not None,
        ADD_OUT=add, LOOP=not tiled, num_warps=warps, num_stages=2)
    return out


@register("rf2_splitk", requires_cuda=True, prepare=prepare_rank_order)
def rankfuse_rf2_splitk(
    x: torch.Tensor,
    lora: MatryoshkaLoRA,
    token_ranks: torch.Tensor,
    r_limit: int | None = None,
    order: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    add: bool = False,
) -> torch.Tensor:
    """Sorted split-K RF2. r_limit (batch maximum rank) and order (stable rank sort) may be
    precomputed by the caller; without r_limit the call pays one host sync."""
    check_layout(x, lora, token_ranks)
    if r_limit is None:
        r_limit = int(token_ranks.max()) if token_ranks.numel() else 0
    return rf2_splitk(x, lora.A_t, lora.B, token_ranks, r_limit, order=order, out=out, add=add,
                      scale_table=rf._scale_table(lora, x.device))
