"""Small host-side helpers shared by the Triton kernels."""

import torch

from rankfuse.interface import MatryoshkaLoRA


SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


def check_inputs(x: torch.Tensor, lora: MatryoshkaLoRA, token_ranks: torch.Tensor) -> int:
    if not x.is_cuda:
        raise ValueError("RankFuse kernels require CUDA tensors")
    if x.ndim != 2 or token_ranks.ndim != 1 or token_ranks.numel() != x.shape[0]:
        raise ValueError("expected x [T, h_in] and token_ranks [T]")
    if tuple(lora.A_t.shape) != (lora.r_max, x.shape[1]):
        raise ValueError("A_t shape must be [r_max, h_in]")
    if x.dtype not in SUPPORTED_DTYPES or lora.A_t.dtype != x.dtype or lora.B.dtype != x.dtype:
        raise ValueError("x, A_t, and B must share a supported floating dtype")
    if token_ranks.dtype not in (torch.int32, torch.int64):
        raise ValueError("token_ranks must be int32 or int64")
    if any(t.device != x.device for t in (lora.A_t, lora.B, token_ranks)):
        raise ValueError("all inputs must be on the same CUDA device")
    if not all(t.is_contiguous() for t in (x, lora.A_t, lora.B, token_ranks)):
        raise ValueError("RankFuse currently requires contiguous inputs")
    if x.shape[0] == 0:
        return 0

    lo, hi = int(token_ranks.min()), int(token_ranks.max())
    if lo < 0 or hi > lora.r_max:
        raise ValueError(f"token ranks must be in [0, {lora.r_max}]")
    return hi


def token_scales(lora: MatryoshkaLoRA, token_ranks: torch.Tensor, fallback: torch.Tensor):
    scales = lora.token_scales(token_ranks)
    return (fallback, False) if scales is None else (scales, True)


def dtype_flags(dtype: torch.dtype) -> tuple[bool, bool]:
    return dtype == torch.float32, dtype == torch.bfloat16
