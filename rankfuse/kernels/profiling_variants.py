"""Timing-only RF2 ablations used by the component profiling study.

These functions intentionally are not registered implementations and are not used by
correctness tests.  They isolate the host-visible alternatives that motivate the
step-level measurements: contiguous token rows, a fixed rank bucket, and contiguous
output rows.  The returned tensors are useful for timing but are not correctness
claims because the intermediate permutations are deliberately kept out of the
serving path.
"""

from __future__ import annotations

import torch

from rankfuse.interface import MatryoshkaLoRA
from rankfuse.kernels import rankfuse_fused as rf1
from rankfuse.kernels import rankfuse_rf2_splitk as rf2


def rf2_contiguous_rows(
    x: torch.Tensor,
    lora: MatryoshkaLoRA,
    token_ranks: torch.Tensor,
    order: torch.Tensor,
) -> torch.Tensor:
    """Run the RF1 kernel on rank-contiguous rows (timing-only ablation)."""
    sorted_x = x.index_select(0, order).contiguous()
    sorted_ranks = token_ranks.index_select(0, order).contiguous()
    return rf1.rankfuse_lora(
        sorted_x,
        lora.A_t,
        lora.B,
        sorted_ranks,
        r_limit=int(sorted_ranks.max()) if sorted_ranks.numel() else 0,
        scale_table=rf1._scale_table(lora, x.device),
    )


def rf2_fixed_bucket(
    x: torch.Tensor,
    lora: MatryoshkaLoRA,
    token_ranks: torch.Tensor,
    order: torch.Tensor,
    bucket: int,
) -> torch.Tensor:
    """Run RF2 with a caller-provided bucket (timing-only control)."""
    return rf2.rf2_splitk(
        x,
        lora.A_t,
        lora.B,
        token_ranks,
        min(bucket, lora.r_max),
        order=order,
        scale_table=rf1._scale_table(lora, x.device),
    )


def rf2_contiguous_output(
    x: torch.Tensor,
    lora: MatryoshkaLoRA,
    token_ranks: torch.Tensor,
    order: torch.Tensor,
) -> torch.Tensor:
    """Run RF2 and materialize output in sorted order (timing-only control)."""
    y_sorted = rf2.rankfuse_rf2_splitk(
        x,
        lora,
        token_ranks,
        r_limit=int(token_ranks.max()) if token_ranks.numel() else 0,
        order=order,
    )
    return torch.empty_like(y_sorted).index_copy(0, order, y_sorted)


def rf1_with_index_setup(
    x: torch.Tensor,
    lora: MatryoshkaLoRA,
    token_ranks: torch.Tensor,
    order: torch.Tensor,
) -> torch.Tensor:
    """Add RF2's index setup loads before RF1 (timing-only control)."""
    tok = order.to(token_ranks.dtype)
    _ = token_ranks.index_select(0, tok)
    _ = torch.amax(token_ranks.index_select(0, tok))
    return rf1.rankfuse_fused(x, lora, token_ranks, r_limit=int(token_ranks.max()))


VARIANTS = {
    "rf2_contiguous_rows": rf2_contiguous_rows,
    "rf2_fixed_bucket": rf2_fixed_bucket,
    "rf2_contiguous_output": rf2_contiguous_output,
    "rf1_with_index_setup": rf1_with_index_setup,
}
