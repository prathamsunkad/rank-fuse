"""S-LoRA heterogeneous-rank BGMV (Batched Gather Matrix-Vector) kernel.

Provides Triton implementations of BGMV shrink and expand for multi-adapter Matryoshka LoRA:
- b4_ma_copies: one unpadded copy per (adapter, rank)
- b4_ma_shared: one shared copy per adapter at r_max, sliced to each token's rank
"""

from __future__ import annotations

import os
import torch
import triton
import triton.language as tl


# =====================================================================
# Triton BGMV Kernels
# =====================================================================

@triton.jit
def _bgmv_shrink_kernel(
    x_ptr,
    a_ptr,
    v_ptr,
    indices_ptr,
    token_ranks_ptr,
    T,
    H_IN,
    stride_xt, stride_xh,
    stride_an, stride_ar, stride_ah,
    stride_vt, stride_vr,
    BK: tl.constexpr,
    BR: tl.constexpr,
):
    pid_t = tl.program_id(0)
    if pid_t >= T:
        return
    idx = tl.load(indices_ptr + pid_t)
    r = tl.load(token_ranks_ptr + pid_t)
    if idx < 0 or r <= 0:
        return

    r_offsets = tl.arange(0, BR)
    r_mask = r_offsets < r
    acc = tl.zeros((BR,), dtype=tl.float32)
    k_offsets = tl.arange(0, BK)

    for k_start in range(0, H_IN, BK):
        k_mask = (k_start + k_offsets) < H_IN
        x_val = tl.load(x_ptr + pid_t * stride_xt + (k_start + k_offsets) * stride_xh, mask=k_mask, other=0.0)
        a_ptrs = (
            a_ptr
            + idx.to(tl.int64) * stride_an
            + r_offsets[:, None].to(tl.int64) * stride_ar
            + (k_start + k_offsets)[None, :].to(tl.int64) * stride_ah
        )
        a_val = tl.load(a_ptrs, mask=r_mask[:, None] & k_mask[None, :], other=0.0)
        acc += tl.sum(a_val.to(tl.float32) * x_val[None, :].to(tl.float32), axis=1)

    v_ptrs = v_ptr + pid_t.to(tl.int64) * stride_vt + r_offsets.to(tl.int64) * stride_vr
    tl.store(v_ptrs, acc.to(v_ptr.dtype.element_ty), mask=r_mask)


@triton.jit
def _bgmv_expand_kernel(
    v_ptr,
    b_ptr,
    out_ptr,
    indices_ptr,
    token_ranks_ptr,
    scales_ptr,
    T,
    H_OUT,
    stride_vt, stride_vr,
    stride_bn, stride_br, stride_bh,
    stride_ot, stride_oh,
    BN: tl.constexpr,
    BR: tl.constexpr,
    ADD_OUT: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_t >= T:
        return

    idx = tl.load(indices_ptr + pid_t)
    r = tl.load(token_ranks_ptr + pid_t)
    if idx < 0 or r <= 0:
        return

    n_start = pid_n * BN
    n_offsets = n_start + tl.arange(0, BN)
    n_mask = n_offsets < H_OUT
    r_offsets = tl.arange(0, BR)
    r_mask = r_offsets < r

    v_ptrs = v_ptr + pid_t.to(tl.int64) * stride_vt + r_offsets.to(tl.int64) * stride_vr
    v_val = tl.load(v_ptrs, mask=r_mask, other=0.0).to(tl.float32)

    b_ptrs = (
        b_ptr
        + idx.to(tl.int64) * stride_bn
        + r_offsets[:, None].to(tl.int64) * stride_br
        + n_offsets[None, :].to(tl.int64) * stride_bh
    )
    b_val = tl.load(b_ptrs, mask=r_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)

    out_val = tl.sum(v_val[:, None] * b_val, axis=0)

    if scales_ptr is not None:
        scale = tl.load(scales_ptr + r).to(tl.float32)
        out_val = out_val * scale

    out_ptrs = out_ptr + pid_t.to(tl.int64) * stride_ot + n_offsets.to(tl.int64) * stride_oh
    if ADD_OUT:
        prev = tl.load(out_ptrs, mask=n_mask, other=0.0).to(tl.float32)
        out_val = out_val + prev

    tl.store(out_ptrs, out_val.to(out_ptr.dtype.element_ty), mask=n_mask)


# =====================================================================
# Python Launcher & Eager Fallback
# =====================================================================

def bgmv_forward(
    x: torch.Tensor,
    a_pool: torch.Tensor,
    b_pool: torch.Tensor,
    indices: torch.Tensor,
    token_ranks: torch.Tensor,
    *,
    scales: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    add: bool = False,
    v_buf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute S-LoRA BGMV shrink and expand.

    Args:
        x: [T, H_in]
        a_pool: [num_adapters, r_max, H_in]
        b_pool: [num_adapters, r_max, H_out]
        indices: [T] adapter/copy indices
        token_ranks: [T] per-token ranks
        scales: optional [r_max + 1] scale table
        out: optional [T, H_out] output tensor
        add: whether to add delta to out
        v_buf: optional [T, r_max] workspace buffer
    """
    T, h_in = x.shape
    num_adapters, r_max, h_out = b_pool.shape

    if out is None:
        out = torch.zeros(T, h_out, dtype=x.dtype, device=x.device)
    elif not add:
        out.zero_()

    if T == 0:
        return out

    use_cuda = x.is_cuda or os.environ.get("TRITON_INTERPRET") == "1"
    if not use_cuda:
        # CPU eager fallback
        for t in range(T):
            idx = int(indices[t])
            r = int(token_ranks[t])
            if idx < 0 or r <= 0:
                continue
            vt = x[t].float() @ a_pool[idx, :r].float().T
            yt = vt @ b_pool[idx, :r].float()
            if scales is not None:
                yt = yt * float(scales[r])
            if add:
                out[t] += yt.to(out.dtype)
            else:
                out[t] = yt.to(out.dtype)
        return out

    # Allocate or reuse workspace v
    if v_buf is None:
        v_buf = torch.zeros(T, r_max, dtype=torch.float32, device=x.device)
    else:
        v_buf.zero_()

    BR = max(16, triton.next_power_of_2(max(r_max, 1)))
    BK = 64
    BN = 64

    # 1. Launch BGMV Shrink
    grid_shrink = (T,)
    _bgmv_shrink_kernel[grid_shrink](
        x, a_pool, v_buf, indices, token_ranks,
        T, h_in,
        x.stride(0), x.stride(1),
        a_pool.stride(0), a_pool.stride(1), a_pool.stride(2),
        v_buf.stride(0), v_buf.stride(1),
        BK=BK, BR=BR,
    )

    # 2. Launch BGMV Expand
    grid_expand = (T, triton.cdiv(h_out, BN))
    _bgmv_expand_kernel[grid_expand](
        v_buf, b_pool, out, indices, token_ranks, scales,
        T, h_out,
        v_buf.stride(0), v_buf.stride(1),
        b_pool.stride(0), b_pool.stride(1), b_pool.stride(2),
        out.stride(0), out.stride(1),
        BN=BN, BR=BR, ADD_OUT=add,
    )

    return out
