"""B0: reference implementation. Slow, obviously correct, used only as the correctness oracle."""

import torch

from rankfuse.interface import MatryoshkaLoRA, apply_scale, register


@register("b0_reference")
def reference(x: torch.Tensor, lora: MatryoshkaLoRA, token_ranks: torch.Tensor) -> torch.Tensor:
    """Per-token loop: y_i = x_i @ A[:, :r_i] @ B[:r_i]. Computes in fp32 and returns fp32."""
    xf = x.float()
    A_t = lora.A_t.float()
    B = lora.B.float()
    y = torch.zeros(x.shape[0], lora.h_out, dtype=torch.float32, device=x.device)
    for i, r in enumerate(token_ranks.tolist()):
        if r == 0:
            continue
        y[i] = (xf[i] @ A_t[:r].T) @ B[:r]
    return apply_scale(y, lora, token_ranks)
