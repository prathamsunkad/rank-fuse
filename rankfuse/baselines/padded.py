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


def _padded_body(
    x: torch.Tensor, lora: MatryoshkaLoRA, token_ranks: torch.Tensor, r: int, scale_table: torch.Tensor | None
) -> torch.Tensor:
    """b1_padded without the host sync: r and the per-rank scale table are supplied."""
    v = x @ lora.A_t[:r].T
    mask = torch.arange(r, device=x.device) < token_ranks.unsqueeze(1)
    v = v * mask.to(v.dtype)
    y = v @ lora.B[:r]
    if scale_table is None:
        return y
    return y * scale_table[token_ranks.long()].unsqueeze(1).to(y.dtype)


# b1_padded_graphed: B1's ~6 kernels captured once in a CUDA graph and replayed, the way vLLM
# runs decode. This is the fair decode baseline for RF1: most of B1's launch overhead is gone,
# so what remains is the cost of the extra kernels and v's trips through HBM.
#
# Graphs are cached per (x, lora, token_ranks) storage, like b2_grouped_graphed. A graph replays
# raw device addresses, so each entry also holds the tensors it reads; otherwise a freed
# address could be reused by a new tensor and hit a stale graph. The batch maximum rank is
# read once at capture (or passed as r_b); later changes to token_ranks must not raise it.
# The returned tensor is the graph's static output and is overwritten by the next replay.
_GRAPH_CACHE: dict[tuple, tuple] = {}


@register("b1_padded_graphed", requires_cuda=True)
def padded_graphed(
    x: torch.Tensor, lora: MatryoshkaLoRA, token_ranks: torch.Tensor, r_b: int | None = None
) -> torch.Tensor:
    key = (x.data_ptr(), tuple(x.shape), lora.A_t.data_ptr(), lora.B.data_ptr(), token_ranks.data_ptr(), r_b)
    if key not in _GRAPH_CACHE:
        r = int(token_ranks.max()) if r_b is None else r_b
        # Built before capture: filling a table from Python numbers copies host data, which a
        # capturing stream does not allow. Inside the graph only the lookup remains.
        scale_table = None
        if lora.scale_per_rank is not None:
            scale_table = torch.zeros(lora.r_max + 1, dtype=torch.float32)
            for rank, s in lora.scale_per_rank.items():
                scale_table[rank] = s
            scale_table = scale_table.to(x.device)
        # Warm up on a side stream before capture, as recommended for CUDA graphs.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            _padded_body(x, lora, token_ranks, r, scale_table)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = _padded_body(x, lora, token_ranks, r, scale_table)
        _GRAPH_CACHE[key] = (graph, out, x, lora, token_ranks, scale_table)
    graph, out = _GRAPH_CACHE[key][:2]
    graph.replay()
    return out


def clear_graph_cache() -> None:
    _GRAPH_CACHE.clear()
