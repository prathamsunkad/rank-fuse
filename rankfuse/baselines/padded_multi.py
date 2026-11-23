"""B1-multi: plain PyTorch, padded per adapter group (Stage 1's B1, once per adapter).

For each adapter group g of the step (adapter a, its tokens, its maximum rank r_g):

    v = x[tokens] @ A[a, s, :r_g].T        cuBLAS
    v *= arange(r_g) < ranks[tokens]       the rank mask
    y = v @ B[a, s, :r_g]                  cuBLAS
    y *= scale(a, ranks[tokens])
    out[tokens, slice s] (+)= y            index_add_

The groups (token indices, r_g) come from the per-step metadata built in `prepare`, so the call
makes no host sync and is capturable; it launches about 7 kernels per group and slice.
"""

import torch

from rankfuse.multi import PagedWeights, build_meta, output_buffer, register_multi


def _prepare(x, W: PagedWeights, token_adapter, token_ranks) -> dict:
    meta = build_meta(token_adapter, token_ranks, num_adapters=W.num_adapters, mode="group", device="cpu")
    order = meta.order.long()
    groups = []
    for g in range(meta.num_groups):
        a = int(meta.group_adapter[g])
        if a < 0:
            continue
        s0, n = int(meta.group_start[g]), int(meta.group_len[g])
        idx = order[s0:s0 + n].to(x.device)
        groups.append((a, idx, int(meta.group_rank_limit[g]), token_ranks.index_select(0, idx).long()))
    return dict(groups=groups)


@register_multi("b1_multi", requires_cuda=False, prepare=_prepare)
def padded_multi(x: torch.Tensor, W: PagedWeights, token_adapter, token_ranks, groups=None, out=None, add=False):
    if groups is None:
        groups = _prepare(x, W, token_adapter, token_ranks)["groups"]
    out = output_buffer(x, W, out, add)
    if not add:
        out.zero_()
    col0 = [sum(W.slice_sizes[:s]) for s in range(W.n_slices)]
    for a, idx, r, ranks in groups:
        xg = x.index_select(0, idx)
        mask = (torch.arange(r, device=x.device) < ranks.unsqueeze(1)).to(x.dtype)
        scale = None if W.scale_table is None else W.scale_table[a].index_select(0, ranks).unsqueeze(1).to(x.dtype)
        for s, n in enumerate(W.slice_sizes):
            v = (xg @ W.A_dense[a, s, :r].T) * mask
            y = v @ W.B_dense[a, s, :r, :n]
            if scale is not None:
                y = y * scale
            if W.n_slices == 1:
                out.index_add_(0, idx, y)
            else:
                out[:, col0[s]:col0[s] + n].index_add_(0, idx, y)
    return out
