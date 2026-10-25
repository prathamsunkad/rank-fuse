"""RF2, segment (`rf2_segment`): rank skipping without sorting tokens.

In a serving batch the tokens of one request sit next to each other and share its rank. This
version keeps the original token order and works on blocks of consecutive tokens, described by
a block table [num_blocks, 2] of (first token, length) that the caller builds once per step:

    contig_blocks   fixed blocks of BM tokens; in prompt batches most blocks lie inside one
                    request, a block spanning two requests takes the larger rank (default)
    aligned_blocks  blocks never cross a request boundary (every block holds one rank); only
                    useful for long requests, very slow for decode batches (one-token blocks)

Each block reads its tokens' ranks and runs at the smallest bucket (16/32/64/128/256 rank rows)
covering its maximum rank: split-K shrink into an fp32 workspace indexed by token, then sum +
mask + scale + expand, as in rf1_fused (rankfuse_fused.py, section 8). Loads and stores of x and
out stay contiguous and there is no sort.

Measured on an H200 (4096 x 4096, r_max 64): 6-11% faster than rf1_fused on prompt batches that
mix low ranks with a rank-64 request, 2-6% slower when the batch maximum rank is already low.
"""

import torch
import triton
import triton.language as tl

from rankfuse.interface import MatryoshkaLoRA, register
from rankfuse.kernels import rankfuse_fused as rf
from rankfuse.kernels._common import check_layout


@triton.jit
def _seg_shrink_part(x_ptr, a_ptr, ws_k, rows, row_ok, H_IN, r_limit, h_lo, K_PER,
                     BM: tl.constexpr, BH: tl.constexpr, R: tl.constexpr, R_PAD: tl.constexpr):
    k = tl.arange(0, R)
    k_ok = k < r_limit
    v = tl.zeros((BM, R), dtype=tl.float32)
    for h_start in range(0, K_PER, BH):
        h = h_lo + h_start + tl.arange(0, BH)
        h_ok = h < H_IN
        x = tl.load(x_ptr + rows[:, None] * H_IN + h[None, :], mask=row_ok[:, None] & h_ok[None, :], other=0.0)
        a = tl.load(a_ptr + k[:, None] * H_IN + h[None, :], mask=k_ok[:, None] & h_ok[None, :], other=0.0)
        v += tl.dot(x, tl.trans(a), input_precision="ieee")
    tl.store(ws_k + rows[:, None] * R_PAD + k[None, :], v, mask=row_ok[:, None])


@triton.jit
def _seg_shrink_kernel(x_ptr, a_ptr, ranks_ptr, blocks_ptr, ws_ptr, T, H_IN, r_limit, K_PER,
                       BM: tl.constexpr, BH: tl.constexpr, R_PAD: tl.constexpr):
    start = tl.load(blocks_ptr + 2 * tl.program_id(0))
    length = tl.load(blocks_ptr + 2 * tl.program_id(0) + 1)
    rows = start + tl.arange(0, BM)
    row_ok = tl.arange(0, BM) < length
    ranks = tl.load(ranks_ptr + rows, mask=row_ok, other=0)
    tile_rank = tl.max(ranks, axis=0)
    h_lo = tl.program_id(1) * K_PER
    ws_k = ws_ptr + tl.program_id(1) * T * R_PAD
    if tile_rank > 0:
        if tile_rank <= 16:
            _seg_shrink_part(x_ptr, a_ptr, ws_k, rows, row_ok, H_IN, r_limit, h_lo, K_PER, BM=BM, BH=BH, R=16, R_PAD=R_PAD)
        elif tile_rank <= 32:
            if R_PAD >= 32:
                _seg_shrink_part(x_ptr, a_ptr, ws_k, rows, row_ok, H_IN, r_limit, h_lo, K_PER, BM=BM, BH=BH, R=32, R_PAD=R_PAD)
        elif tile_rank <= 64:
            if R_PAD >= 64:
                _seg_shrink_part(x_ptr, a_ptr, ws_k, rows, row_ok, H_IN, r_limit, h_lo, K_PER, BM=BM, BH=BH, R=64, R_PAD=R_PAD)
        elif tile_rank <= 128:
            if R_PAD >= 128:
                _seg_shrink_part(x_ptr, a_ptr, ws_k, rows, row_ok, H_IN, r_limit, h_lo, K_PER, BM=BM, BH=BH, R=128, R_PAD=R_PAD)
        else:
            if R_PAD >= 256:
                _seg_shrink_part(x_ptr, a_ptr, ws_k, rows, row_ok, H_IN, r_limit, h_lo, K_PER, BM=BM, BH=BH, R=256, R_PAD=R_PAD)


@triton.jit
def _seg_store_tile(v_in, b_ptr, out_ptr, rows, row_ok, k, k_ok, col_start, N, OUT_STRIDE,
                    BN: tl.constexpr, ADD_OUT: tl.constexpr):
    cols = col_start + tl.arange(0, BN)
    col_ok = cols < N
    b = tl.load(b_ptr + k[:, None] * N + cols[None, :], mask=k_ok[:, None] & col_ok[None, :], other=0.0)
    acc = tl.dot(v_in, b, input_precision="ieee")
    ptrs = out_ptr + rows[:, None] * OUT_STRIDE + cols[None, :]
    mask = row_ok[:, None] & col_ok[None, :]
    if ADD_OUT:
        acc += tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    tl.store(ptrs, acc, mask=mask)


@triton.jit
def _seg_expand_part(ws_ptr, scale_ptr, b_ptr, out_ptr, rows, row_ok, ranks, T, N, OUT_STRIDE, r_limit, SPLIT_K,
                     col_base, cols_per_prog, BM: tl.constexpr, BN: tl.constexpr, R: tl.constexpr, R_PAD: tl.constexpr,
                     HAS_SCALE: tl.constexpr, ADD_OUT: tl.constexpr, LOOP: tl.constexpr):
    k = tl.arange(0, R)
    k_ok = k < r_limit
    v = tl.zeros((BM, R), dtype=tl.float32)
    for s in range(0, SPLIT_K):
        v += tl.load(ws_ptr + s * T * R_PAD + rows[:, None] * R_PAD + k[None, :], mask=row_ok[:, None], other=0.0)
    v = tl.where(k[None, :] < ranks[:, None], v, 0.0)
    if HAS_SCALE:
        v = v * tl.load(scale_ptr + ranks, mask=row_ok, other=0.0)[:, None]
    v_in = v.to(out_ptr.dtype.element_ty)
    if LOOP:
        for col_start in range(0, cols_per_prog, BN):
            _seg_store_tile(v_in, b_ptr, out_ptr, rows, row_ok, k, k_ok, col_base + col_start, N, OUT_STRIDE, BN=BN, ADD_OUT=ADD_OUT)
    else:
        _seg_store_tile(v_in, b_ptr, out_ptr, rows, row_ok, k, k_ok, col_base, N, OUT_STRIDE, BN=BN, ADD_OUT=ADD_OUT)


@triton.jit
def _seg_zero_part(out_ptr, rows, row_ok, N, OUT_STRIDE, col_base, cols_per_prog,
                   BM: tl.constexpr, BN: tl.constexpr, LOOP: tl.constexpr):
    zeros = tl.zeros((BM, BN), dtype=out_ptr.dtype.element_ty)
    if LOOP:
        for col_start in range(0, cols_per_prog, BN):
            cols = col_base + col_start + tl.arange(0, BN)
            tl.store(out_ptr + rows[:, None] * OUT_STRIDE + cols[None, :], zeros, mask=row_ok[:, None] & (cols[None, :] < N))
    else:
        cols = col_base + tl.arange(0, BN)
        tl.store(out_ptr + rows[:, None] * OUT_STRIDE + cols[None, :], zeros, mask=row_ok[:, None] & (cols[None, :] < N))


@triton.jit
def _seg_expand_kernel(ws_ptr, ranks_ptr, blocks_ptr, scale_ptr, b_ptr, out_ptr, T, N, OUT_STRIDE, r_limit, SPLIT_K,
                       cols_per_prog, BM: tl.constexpr, BN: tl.constexpr, R_PAD: tl.constexpr,
                       HAS_SCALE: tl.constexpr, ADD_OUT: tl.constexpr, LOOP: tl.constexpr):
    start = tl.load(blocks_ptr + 2 * tl.program_id(0))
    length = tl.load(blocks_ptr + 2 * tl.program_id(0) + 1)
    rows = start + tl.arange(0, BM)
    row_ok = tl.arange(0, BM) < length
    ranks = tl.load(ranks_ptr + rows, mask=row_ok, other=0)
    tile_rank = tl.max(ranks, axis=0)
    col_base = tl.program_id(1) * cols_per_prog
    if tile_rank == 0:
        if not ADD_OUT:
            _seg_zero_part(out_ptr, rows, row_ok, N, OUT_STRIDE, col_base, cols_per_prog, BM=BM, BN=BN, LOOP=LOOP)
    elif tile_rank <= 16:
        _seg_expand_part(ws_ptr, scale_ptr, b_ptr, out_ptr, rows, row_ok, ranks, T, N, OUT_STRIDE, r_limit, SPLIT_K,
                         col_base, cols_per_prog, BM=BM, BN=BN, R=16, R_PAD=R_PAD, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT, LOOP=LOOP)
    elif tile_rank <= 32:
        if R_PAD >= 32:
            _seg_expand_part(ws_ptr, scale_ptr, b_ptr, out_ptr, rows, row_ok, ranks, T, N, OUT_STRIDE, r_limit, SPLIT_K,
                             col_base, cols_per_prog, BM=BM, BN=BN, R=32, R_PAD=R_PAD, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT, LOOP=LOOP)
    elif tile_rank <= 64:
        if R_PAD >= 64:
            _seg_expand_part(ws_ptr, scale_ptr, b_ptr, out_ptr, rows, row_ok, ranks, T, N, OUT_STRIDE, r_limit, SPLIT_K,
                             col_base, cols_per_prog, BM=BM, BN=BN, R=64, R_PAD=R_PAD, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT, LOOP=LOOP)
    elif tile_rank <= 128:
        if R_PAD >= 128:
            _seg_expand_part(ws_ptr, scale_ptr, b_ptr, out_ptr, rows, row_ok, ranks, T, N, OUT_STRIDE, r_limit, SPLIT_K,
                             col_base, cols_per_prog, BM=BM, BN=BN, R=128, R_PAD=R_PAD, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT, LOOP=LOOP)
    else:
        if R_PAD >= 256:
            _seg_expand_part(ws_ptr, scale_ptr, b_ptr, out_ptr, rows, row_ok, ranks, T, N, OUT_STRIDE, r_limit, SPLIT_K,
                             col_base, cols_per_prog, BM=BM, BN=BN, R=256, R_PAD=R_PAD, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT, LOOP=LOOP)


def block_m_for(num_tokens: int, r_limit: int) -> int:
    """Token block size, as rf1_fused picks it for this batch."""
    return rf._block_m(num_tokens, max(16, triton.next_power_of_2(max(r_limit, 1))))


def contig_blocks(num_tokens: int, bm: int, device) -> torch.Tensor:
    """[(first token, length)] for fixed blocks of bm consecutive tokens."""
    starts = torch.arange(0, num_tokens, bm)
    lens = torch.clamp(num_tokens - starts, max=bm)
    return torch.stack([starts, lens], dim=1).to(torch.int32).contiguous().to(device)


def aligned_blocks(seg_lens: list[int], bm: int, device) -> torch.Tensor:
    """[(first token, length)] for blocks of at most bm tokens that never cross a request."""
    rows, start = [], 0
    for n in seg_lens:
        for off in range(0, n, bm):
            rows.append((start + off, min(bm, n - off)))
        start += n
    return torch.tensor(rows, dtype=torch.int32, device=device)


def rf2_segment(x, A_t, B, token_ranks, r_limit, blocks, bm, out=None, add=False, scale_table=None):
    """Segment RF2 for one adapter: out (+)= delta over the given block table (block size bm)."""
    T, h_in = x.shape
    n = B.shape[1]
    if out is None:
        if add:
            raise ValueError("add=True needs an existing out to add into")
        out = torch.empty((T, n), dtype=x.dtype, device=x.device)
    if T == 0:
        return out
    r_pad = max(16, triton.next_power_of_2(max(r_limit, 1)))
    dev = x.device.index or 0
    num_sms = rf._num_sms(dev)
    nb = blocks.shape[0]
    bh, stages = rf._splitk_shrink_tiles(bm, r_pad, x.element_size(), dev)
    tiled = T > rf.EXPAND_TILED_MIN_TOKENS
    split_k = max(rf.MIN_SPLIT_K_TILED if tiled else 1, min(rf.MAX_SPLIT_K, rf.SPLITK_WAVES * num_sms // nb))
    k_per = triton.cdiv(triton.cdiv(h_in, split_k), bh) * bh
    split_k = triton.cdiv(h_in, k_per)
    ws = torch.empty((split_k, T, r_pad), dtype=torch.float32, device=x.device)
    _seg_shrink_kernel[(nb, split_k)](x, A_t, token_ranks, blocks, ws, T, h_in, r_limit, k_per,
                                      BM=bm, BH=bh, R_PAD=r_pad, num_warps=4, num_stages=stages)
    if tiled:
        block_n = rf.EXPAND_TILED_BLOCK_N
        while block_n > 64 and r_pad * block_n * x.element_size() > 0.35 * rf._max_shared_mem(dev):
            block_n //= 2
        cols_per_prog, warps = block_n, rf.EXPAND_TILED_WARPS
    else:
        block_n, warps = rf.EXPAND_BLOCK_N, rf.EXPAND_WARPS
        n_tiles = triton.cdiv(n, block_n)
        n_split = min(n_tiles, max(1, rf.EXPAND_WAVES * num_sms // nb))
        cols_per_prog = triton.cdiv(n_tiles, n_split) * block_n
    _seg_expand_kernel[(nb, triton.cdiv(n, cols_per_prog))](
        ws, token_ranks, blocks, out if scale_table is None else scale_table, B, out, T, n, out.stride(0), r_limit,
        split_k, cols_per_prog, BM=bm, BN=block_n, R_PAD=r_pad, HAS_SCALE=scale_table is not None, ADD_OUT=add,
        LOOP=not tiled, num_warps=warps, num_stages=2)
    return out


@register("rf2_segment", requires_cuda=True)
def rankfuse_rf2_segment(
    x: torch.Tensor,
    lora: MatryoshkaLoRA,
    token_ranks: torch.Tensor,
    r_limit: int | None = None,
    blocks: torch.Tensor | None = None,
    bm: int | None = None,
    out: torch.Tensor | None = None,
    add: bool = False,
) -> torch.Tensor:
    """Segment RF2. r_limit, blocks and bm may be precomputed by the caller (once per step);
    by default the batch maximum rank is read (one host sync) and contig blocks are used."""
    check_layout(x, lora, token_ranks)
    if r_limit is None:
        r_limit = int(token_ranks.max()) if token_ranks.numel() else 0
    if blocks is None:
        bm = block_m_for(x.shape[0], r_limit)
        blocks = contig_blocks(x.shape[0], bm, x.device)
    return rf2_segment(x, lora.A_t, lora.B, token_ranks, r_limit, blocks, bm, out=out, add=add,
                       scale_table=rf._scale_table(lora, x.device))
