"""rf1_masked: the first RF1, fused padded/masked LoRA in one Triton kernel.

    grid    (cdiv(T, BLOCK_M), cdiv(h_out, BLOCK_N)): each program owns one output tile
    kernel  for each 16-row rank band below the batch maximum rank:
                mid = x[rows] @ A_t[band].T         shrink over all of h_in, in fp32
                mid[t, k] = 0 for k >= ranks[t]     per-token rank mask
                out += mid @ B[band, cols]
            then the per-token scale and one store

Every column tile of a token block recomputes the same shrink (h_out / BLOCK_N times), which is
what rf1_fused's split-K path (rankfuse_fused.py) removes. Kept as an ablation.

Layouts (rankfuse/interface.py): x [T, h_in]; A_t [r_max, h_in] and B [r_max, h_out], rank-r
prefix = rows [:r], contiguous; y [T, h_out], the LoRA delta only.

The commented-out code below is the original two-kernel draft (shrink to v in HBM, then
expand), never finished.

`TRITON_INTERPRET=1 python -m pytest -k rf1_masked` runs the kernel in Triton's CPU interpreter
(Linux only), which is slow but allows print debugging.
"""


import torch
import triton
import triton.language as tl

# from rankfuse.interface import MatryoshkaLoRA, apply_scale, register  # noqa: F401
# @triton.jit
# def _shrink_kernel(
#     x_ptr, a_t_ptr, v_ptr, ranks_ptr,
#     T, H_IN, R,
#     stride_xt, stride_xh,
#     stride_ar, stride_ah,
#     stride_vt, stride_vr,
#     BLOCK_T: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_K: tl.constexpr,
# ):
#     """v[t, k] = sum_h x[t, h] * A_t[k, h]  for k < token_ranks[t], else 0.

#     Grid: (cdiv(T, BLOCK_T), cdiv(R, BLOCK_R)), one program per (token tile, rank tile).

#     Each program should:
#     1. Compute its token offsets and rank offsets from the program ids.
#     2. Load token_ranks for its token tile. If the rank tile starts at or beyond the largest
#        rank in the token tile, write zeros and return early (no wasted work).
#     3. Loop over h_in in chunks of BLOCK_K: load an x tile [BLOCK_T, BLOCK_K] and an A_t tile
#        [BLOCK_R, BLOCK_K], accumulate tl.dot(x_tile, tl.trans(a_tile)) in fp32.
#     4. Zero entries where rank offset >= that token's rank.
#     5. Store the [BLOCK_T, BLOCK_R] tile to v, masking out-of-bounds tokens and ranks.
#     """
#     pass


# @triton.jit
# def _expand_kernel(
#     v_ptr, b_ptr, y_ptr, ranks_ptr,
#     T, H_OUT, R,
#     stride_vt, stride_vr,
#     stride_br, stride_bh,
#     stride_yt, stride_yh,
#     BLOCK_T: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_R: tl.constexpr,
# ):
#     """y[t, n] = sum_{k < token_ranks[t]} v[t, k] * B[k, n].

#     Grid: (cdiv(T, BLOCK_T), cdiv(H_OUT, BLOCK_N)), one program per (token tile, output tile).

#     Each program should:
#     1. Compute its token offsets and output-column offsets from the program ids.
#     2. Load token_ranks for its token tile; the rank loop only needs to reach the largest
#        rank in this tile, not R.
#     3. Loop over ranks in chunks of BLOCK_R: load a v tile [BLOCK_T, BLOCK_R] (masking
#        k >= each token's rank) and a B tile [BLOCK_R, BLOCK_N], accumulate tl.dot in fp32.
#     4. Store the [BLOCK_T, BLOCK_N] tile to y in y's dtype, masking out-of-bounds entries.
#     """
#     pass


# # Uncomment once the kernels pass tests/test_correctness.py on Phoenix. Registered before
# # then, the empty kernels would fail the GPU tests and stop the sweep job.
# # @register("rf1_masked", requires_cuda=True)
# def rf1_masked(x: torch.Tensor, lora: MatryoshkaLoRA, token_ranks: torch.Tensor) -> torch.Tensor:
#     """Host wrapper with the shared signature: (x [T, h_in], lora, token_ranks [T]) -> y [T, h_out].

#     Should:
#     1. Check inputs are on CUDA, contiguous, and share a dtype; token_ranks is int32.
#     2. r_b = int(token_ranks.max()); allocate v [T, r_b] and y [T, h_out] in x.dtype.
#     3. Launch _shrink_kernel over (cdiv(T, BLOCK_T), cdiv(r_b, BLOCK_R)).
#     4. Launch _expand_kernel over (cdiv(T, BLOCK_T), cdiv(h_out, BLOCK_N)).
#     5. Return apply_scale(y, lora, token_ranks).

#     Start with fixed block sizes (e.g. BLOCK_T=16, BLOCK_R=16, BLOCK_K=64, BLOCK_N=64);
#     add @triton.autotune once correct.
#     """
#     raise NotImplementedError("rf1_masked: kernels not written yet")

from rankfuse.interface import MatryoshkaLoRA, register
from rankfuse.kernels._common import check_inputs, dtype_flags, token_scales


@triton.jit
def _rf1_kernel(
    x_ptr,
    a_ptr,
    b_ptr,
    ranks_ptr,
    scales_ptr,
    y_ptr,
    T: tl.constexpr,
    H_IN: tl.constexpr,
    H_OUT: tl.constexpr,
    R_LIMIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_R: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_FP32: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < T
    ranks = tl.load(ranks_ptr + rows, mask=row_mask, other=0)
    out = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    # Every token tile visits the batch maximum rank; per-token masks keep it correct.
    for rank_start in range(0, R_LIMIT, BLOCK_R):
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
                mask=(rank_ids[:, None] < R_LIMIT) & (h[None, :] < H_IN),
                other=0.0,
            )
            mid += tl.dot(x, tl.trans(a), input_precision="ieee")

        mid = tl.where(rank_ids[None, :] < ranks[:, None], mid, 0.0)
        b = tl.load(
            b_ptr + rank_ids[:, None] * H_OUT + cols[None, :],
            mask=(rank_ids[:, None] < R_LIMIT) & (cols[None, :] < H_OUT),
            other=0.0,
        )
        if IS_FP32:
            mid_for_dot = mid
        elif IS_BF16:
            mid_for_dot = mid.to(tl.bfloat16)
        else:
            mid_for_dot = mid.to(tl.float16)
        out += tl.dot(mid_for_dot, b, input_precision="ieee")

    if HAS_SCALE:
        scale = tl.load(scales_ptr + rows, mask=row_mask, other=0.0)
        out *= scale[:, None]
    tl.store(
        y_ptr + rows[:, None] * H_OUT + cols[None, :],
        out,
        mask=row_mask[:, None] & (cols[None, :] < H_OUT),
    )


@register("rf1_masked", requires_cuda=True)
def rankfuse_masked(
    x: torch.Tensor, lora: MatryoshkaLoRA, token_ranks: torch.Tensor
) -> torch.Tensor:
    rank_limit = check_inputs(x, lora, token_ranks)
    out = torch.empty((x.shape[0], lora.h_out), dtype=x.dtype, device=x.device)
    if x.shape[0] == 0:
        return out

    scales, has_scale = token_scales(lora, token_ranks, x)
    is_fp32, is_bf16 = dtype_flags(x.dtype)
    block_m, block_n = 16, 64
    grid = (triton.cdiv(x.shape[0], block_m), triton.cdiv(lora.h_out, block_n))
    _rf1_kernel[grid](
        x,
        lora.A_t,
        lora.B,
        token_ranks,
        scales,
        out,
        T=x.shape[0],
        H_IN=lora.h_in,
        H_OUT=lora.h_out,
        R_LIMIT=max(rank_limit, 1),
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

