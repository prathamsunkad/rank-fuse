"""B0-multi: the multi-adapter correctness oracle. Slow, obviously correct, fp32.

    y[t, slice s] = scale(a_t, r_t) * x[t] @ A[a_t, s, :r_t].T @ B[a_t, s, :r_t, :size_s]

computed in fp32 from the dense weights (W.A_dense / W.B_dense: the original stacked tensors,
not the page pools, so a page-table bug in a kernel shows up as a mismatch), one matmul pair
per distinct (adapter, rank) pair. Rank 0 or adapter < 0: the row is zero.
"""

import torch

from rankfuse.multi import PagedWeights, register_multi


@register_multi("b0_multi", requires_cuda=False)
def reference_multi(x: torch.Tensor, W: PagedWeights, token_adapter: torch.Tensor, token_ranks: torch.Tensor,
                    out=None, add=False) -> torch.Tensor:
    xf = x.float()
    y = torch.zeros(x.shape[0], W.h_out, dtype=torch.float32, device=x.device)
    ad = token_adapter.long().cpu()
    rk = token_ranks.long().cpu()
    live = (rk > 0) & (ad >= 0)
    pairs = torch.unique(torch.stack([ad[live], rk[live]], 1), dim=0).tolist() if live.any() else []
    col0 = [sum(W.slice_sizes[:s]) for s in range(W.n_slices)]
    for a, r in pairs:
        idx = torch.nonzero(live & (ad == a) & (rk == r)).flatten().to(x.device)
        for s, n in enumerate(W.slice_sizes):
            A = W.A_dense[a, s, :r].float()
            B = W.B_dense[a, s, :r, :n].float()
            y[idx, col0[s]:col0[s] + n] = (xf[idx] @ A.T) @ B * W.scale(a, r)
    if out is not None:
        return out.float() + y if add else y
    return y
