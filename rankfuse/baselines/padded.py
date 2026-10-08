"""B1: pad to the batch's largest rank, then mask the intermediate.

All requests share one A and one B, so we cannot mask the weights per request. Instead we
compute every rank column for every token and zero the columns beyond each token's rank:

    v = x @ A[:, :r]            r = max rank in this batch (not the global r_max)
    v[i, k] = 0 for k >= r_i
    y = v @ B[:r]

Correct for every request. Wastes FLOPs on low-rank tokens; loads each weight row once.
This is the headline baseline for Claim 2.
"""

import torch

from rankfuse.interface import MatryoshkaLoRA, apply_scale, register


@register("b1_padded")
def padded(x: torch.Tensor, lora: MatryoshkaLoRA, token_ranks: torch.Tensor) -> torch.Tensor:
    r = int(token_ranks.max())
    v = x @ lora.A_t[:r].T  # [T, r]
    mask = torch.arange(r, device=x.device) < token_ranks.unsqueeze(1)  # [T, r]
    v = v * mask.to(v.dtype)
    y = v @ lora.B[:r]  # [T, h_out]
    return apply_scale(y, lora, token_ranks)
