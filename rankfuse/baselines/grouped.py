"""B2: group tokens by rank and run one matmul pair per rank group.

No wasted FLOPs, but one pair of launches per distinct rank, smaller matmuls, and x/y are
gathered and scattered once per group. Tests whether splitting the batch costs more than
B1's padding wastes (Claim 2).

Two variants:
* b2_grouped         - eager PyTorch. The grouping (unique ranks + index lists) is computed on
                       every call, including a host sync, the way naive serving code would.
* b2_grouped_graphed - CUDA only. Grouping is computed once and the per-group launches are
                       captured in a CUDA graph, so launch overhead is (mostly) removed.
                       Comparing the two isolates launch/host overhead from the matmul cost.
"""

import torch

from rankfuse.interface import MatryoshkaLoRA, apply_scale, register


def _groups(token_ranks: torch.Tensor) -> list[tuple[int, torch.Tensor]]:
    groups = []
    for r in torch.unique(token_ranks).tolist():
        if r == 0:
            continue
        groups.append((r, torch.nonzero(token_ranks == r).squeeze(1)))
    return groups


def _grouped_compute(x, lora, groups, out):
    for r, idx in groups:
        out[idx] = (x[idx] @ lora.A_t[:r].T) @ lora.B[:r]
    return out


@register("b2_grouped")
def grouped(x: torch.Tensor, lora: MatryoshkaLoRA, token_ranks: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(x.shape[0], lora.h_out, dtype=x.dtype, device=x.device)
    _grouped_compute(x, lora, _groups(token_ranks), out)
    return apply_scale(out, lora, token_ranks)


# Graphs are cached per (x, lora, token_ranks) storage. The benchmark harness reuses the same
# tensors across repetitions, so the first call captures and later calls only replay.
_GRAPH_CACHE: dict[
    tuple,
    tuple[
        torch.cuda.CUDAGraph,
        torch.Tensor,
        list[tuple[int, torch.Tensor]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
] = {}


@register("b2_grouped_graphed", requires_cuda=True)
def grouped_graphed(x: torch.Tensor, lora: MatryoshkaLoRA, token_ranks: torch.Tensor) -> torch.Tensor:
    key = (x.data_ptr(), tuple(x.shape), lora.A_t.data_ptr(), lora.B.data_ptr(), token_ranks.data_ptr())
    if key not in _GRAPH_CACHE:
        groups = _groups(token_ranks)
        out = torch.zeros(x.shape[0], lora.h_out, dtype=x.dtype, device=x.device)
        # Warm up on a side stream before capture, as recommended for CUDA graphs.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            _grouped_compute(x, lora, groups, out)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _grouped_compute(x, lora, groups, out)
        _GRAPH_CACHE[key] = (graph, out, groups, x, lora.A_t, lora.B, token_ranks)
    graph, out, _groups_keepalive, _x, _a_t, _b, _token_ranks = _GRAPH_CACHE[key]
    graph.replay()
    return apply_scale(out, lora, token_ranks)


def clear_graph_cache() -> None:
    _GRAPH_CACHE.clear()
