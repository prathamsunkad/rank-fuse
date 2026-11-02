"""RF2, segment (`rf2_segment`): rank skipping over the batch's own token order.

In a serving batch the tokens of one request sit next to each other and share its rank. This
version keeps the batch order and rf1_fused's blocks of BM consecutive tokens. Each block reads
its tokens' ranks and runs at the smallest bucket (16/32/64/128/256 rank rows) covering its
maximum rank; a block spanning two requests takes the larger rank. Everything else is rf1_fused's
kernel (rankfuse_fused.py, section 8), so the only additions are the rank read and the branch.
Rank-ordered requests make more blocks hold one rank, which is the scheduler's choice.

`rf2_segment_fw` is the control: the same compiled kernels with every block forced to the batch's
bucket. Nothing is skipped, so rf2_segment vs rf2_segment_fw measures skipping alone, and
rf2_segment_fw vs rf1_fused measures the rank read and branch.
"""

import torch

from rankfuse.interface import MatryoshkaLoRA, register
from rankfuse.kernels import rankfuse_fused as rf
from rankfuse.kernels._common import check_layout, prepare_r_limit


def rf2_segment(x, A_t, B, token_ranks, r_limit, out=None, add=False, scale_table=None, slice_sizes=None, full_width=False):
    """Segment RF2 for one adapter or several output slices: out (+)= delta."""
    return rf.rankfuse_lora(x, A_t, B, token_ranks, out=out, add=add, r_limit=r_limit, scale_table=scale_table,
                            slice_sizes=slice_sizes, rank_skip=True, full_width=full_width)


@register("rf2_segment", requires_cuda=True, prepare=prepare_r_limit)
def rankfuse_rf2_segment(
    x: torch.Tensor,
    lora: MatryoshkaLoRA,
    token_ranks: torch.Tensor,
    r_limit: int | None = None,
    out: torch.Tensor | None = None,
    add: bool = False,
) -> torch.Tensor:
    """Segment RF2. r_limit (batch maximum rank) may be precomputed by the caller once per step;
    without it the call pays one host sync."""
    check_layout(x, lora, token_ranks)
    return rf2_segment(x, lora.A_t, lora.B, token_ranks, r_limit, out=out, add=add, scale_table=rf._scale_table(lora, x.device))


@register("rf2_segment_fw", requires_cuda=True, prepare=prepare_r_limit)
def rankfuse_rf2_segment_fw(
    x: torch.Tensor,
    lora: MatryoshkaLoRA,
    token_ranks: torch.Tensor,
    r_limit: int | None = None,
    out: torch.Tensor | None = None,
    add: bool = False,
) -> torch.Tensor:
    """rf2_segment at full width: the same kernels, nothing skipped."""
    check_layout(x, lora, token_ranks)
    return rf2_segment(x, lora.A_t, lora.B, token_ranks, r_limit, out=out, add=add,
                       scale_table=rf._scale_table(lora, x.device), full_width=True)
