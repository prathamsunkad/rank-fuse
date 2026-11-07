"""RF2, sorted index (`rf2_index`): rf2_segment's rank skipping, reading tokens through a sorted index.

    host     order = argsort(token_ranks, stable), a list of token positions in rank order. The
             tokens themselves stay in batch order; a serving engine builds the list once per step
             and reuses it in every layer (vLLM already sorts tokens by adapter this way).
    kernels  rf1_fused's kernels (rankfuse_fused.py, section 8) with INDEXED and RANK_SKIP: block m
             takes positions m*BM.. of the order, runs at the bucket of its maximum rank, and writes
             each token's result to its own row.

On rank-sorted input this forms the same blocks as rf2_segment, so the two differ only in the index
load per row: rf2_index vs rf2_segment on sorted batches is the cost of assuming a sorted index list
instead of physically sorted tokens. `rf2_index_fw` forces every block to the batch's bucket, the
control for skipping. The sort's time is reported separately (run_sweep's prepare_us).
"""

import torch

from rankfuse.interface import MatryoshkaLoRA, register
from rankfuse.kernels import rankfuse_fused as rf
from rankfuse.kernels._common import check_layout, prepare_rank_order


def rf2_index(x, A_t, B, token_ranks, r_limit, order=None, out=None, add=False, scale_table=None, slice_sizes=None,
              full_width=False):
    """Sorted-index RF2 for one adapter or several output slices: out (+)= delta."""
    return rf.rankfuse_lora(x, A_t, B, token_ranks, out=out, add=add, r_limit=r_limit, scale_table=scale_table,
                            slice_sizes=slice_sizes, rank_sort=True, order=order, rank_skip=True, full_width=full_width)


@register("rf2_index", requires_cuda=True, prepare=prepare_rank_order)
def rankfuse_rf2_index(
    x: torch.Tensor,
    lora: MatryoshkaLoRA,
    token_ranks: torch.Tensor,
    r_limit: int | None = None,
    order: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    add: bool = False,
) -> torch.Tensor:
    """Sorted-index RF2. r_limit and order may be precomputed by the caller once per step; without
    them the call pays one host sync and sorts here."""
    check_layout(x, lora, token_ranks)
    return rf2_index(x, lora.A_t, lora.B, token_ranks, r_limit, order=order, out=out, add=add,
                     scale_table=rf._scale_table(lora, x.device))


@register("rf2_index_fw", requires_cuda=True, prepare=prepare_rank_order)
def rankfuse_rf2_index_fw(
    x: torch.Tensor,
    lora: MatryoshkaLoRA,
    token_ranks: torch.Tensor,
    r_limit: int | None = None,
    order: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    add: bool = False,
) -> torch.Tensor:
    """rf2_index at full width: the same kernels, nothing skipped."""
    check_layout(x, lora, token_ranks)
    return rf2_index(x, lora.A_t, lora.B, token_ranks, r_limit, order=order, out=out, add=add,
                     scale_table=rf._scale_table(lora, x.device), full_width=True)
