"""rf2_ragged: the first RF2, rf1_masked's one-kernel layout over a rank-sorted order.

The host sorts token indices by rank (argsort per call). Each program takes BLOCK_M positions of
that order and one output column tile, reads its tokens' ranks, and loops over 16-row rank bands
only up to the tile's own maximum rank, writing each result to the token's own row. Like
rf1_masked it recomputes the shrink for every column tile; the current RF2 variants run on
rf1_fused's split-K base instead (rankfuse_rf2_segment.py and the others). Kept as an ablation.
"""

import torch
import triton
import triton.language as tl

from rankfuse.interface import MatryoshkaLoRA, register
from rankfuse.kernels._common import check_inputs, dtype_flags, token_scales


@triton.jit
def _rf2_kernel(
    x_ptr,
    a_ptr,
    b_ptr,
    ranks_ptr,
    order_ptr,
    scales_ptr,
    y_ptr,
    T: tl.constexpr,
    H_IN: tl.constexpr,
    H_OUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_R: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_FP32: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    sorted_rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = sorted_rows < T
    rows = tl.load(order_ptr + sorted_rows, mask=row_mask, other=0)
    ranks = tl.load(ranks_ptr + rows, mask=row_mask, other=0)
    tile_rank = tl.max(ranks, axis=0)
    out = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    # Sorting makes most tiles rank-homogeneous; low-rank tiles never visit high bands.
    rank_start = 0
    while rank_start < tile_rank:
        rank_ids = rank_start + tl.arange(0, BLOCK_R)
        mid = tl.zeros((BLOCK_M, BLOCK_R), tl.float32)

        for h_start in range(0, H_IN, BLOCK_H):
            h = h_start + tl.arange(0, BLOCK_H)
            x = tl.load(
                x_ptr + rows[:, None] * H_IN + h[None, :],
                mask=row_mask[:, None] & (h[None, :] < H_IN),
                other=0.0,
            )
            a = tl.load(
                a_ptr + rank_ids[:, None] * H_IN + h[None, :],
                mask=(rank_ids[:, None] < tile_rank) & (h[None, :] < H_IN),
                other=0.0,
            )
            mid += tl.dot(x, tl.trans(a), input_precision="ieee")

        mid = tl.where(rank_ids[None, :] < ranks[:, None], mid, 0.0)
        b = tl.load(
            b_ptr + rank_ids[:, None] * H_OUT + cols[None, :],
            mask=(rank_ids[:, None] < tile_rank) & (cols[None, :] < H_OUT),
            other=0.0,
        )
        if IS_FP32:
            mid_for_dot = mid
        elif IS_BF16:
            mid_for_dot = mid.to(tl.bfloat16)
        else:
            mid_for_dot = mid.to(tl.float16)
        out += tl.dot(mid_for_dot, b, input_precision="ieee")
        rank_start += BLOCK_R

    if HAS_SCALE:
        scale = tl.load(scales_ptr + rows, mask=row_mask, other=0.0)
        out *= scale[:, None]
    tl.store(
        y_ptr + rows[:, None] * H_OUT + cols[None, :],
        out,
        mask=row_mask[:, None] & (cols[None, :] < H_OUT),
    )


@register("rf2_ragged", requires_cuda=True)
def rankfuse_ragged(
    x: torch.Tensor, lora: MatryoshkaLoRA, token_ranks: torch.Tensor
) -> torch.Tensor:
    check_inputs(x, lora, token_ranks)
    out = torch.empty((x.shape[0], lora.h_out), dtype=x.dtype, device=x.device)
    if x.shape[0] == 0:
        return out

    # Serving code can prepare this order once and reuse it across adapted layers.
    order = torch.argsort(token_ranks)
    scales, has_scale = token_scales(lora, token_ranks, x)
    is_fp32, is_bf16 = dtype_flags(x.dtype)
    block_m, block_n = 16, 64
    grid = (triton.cdiv(x.shape[0], block_m), triton.cdiv(lora.h_out, block_n))
    _rf2_kernel[grid](
        x,
        lora.A_t,
        lora.B,
        token_ranks,
        order,
        scales,
        out,
        T=x.shape[0],
        H_IN=lora.h_in,
        H_OUT=lora.h_out,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_H=32,
        BLOCK_R=16,
        HAS_SCALE=has_scale,
        IS_FP32=is_fp32,
        IS_BF16=is_bf16,
        num_warps=4,
        num_stages=2,
    )
    return out
