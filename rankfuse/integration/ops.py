"""`torch.ops.rankfuse.lora_linear`: the RankFuse LoRA op that vLLM's LoRA layers call.

vLLM runs its model under torch.compile and CUDA graphs. Its own LoRA kernels are custom ops so
that Dynamo records one opaque call instead of tracing Triton launch code, and so that values that
change every step (the number of active adapters) are read at run time from CPU tensors instead of
being baked into the traced graph as Python constants. This op follows the same rules:

    y             [T, sum(slice_sizes)] base-layer output; the LoRA delta is added in place
                  (add=True) or written (add=False). Mutated, never reallocated.
    x             [T, h_in] layer input.
    A_t, B        [S, r_max, h_in] / [S, r_max, n_max] rank-major weights of ONE adapter slot
                  (rankfuse_lora's layout; the plugin keeps these copies in sync with vLLM's
                  slot buffers, see vllm_plugin.py).
    token_ranks   int32 static buffer of the Punica wrapper, at least T entries, refilled every
                  step; 0 means "no adapter" for that token. Only the first T entries are read.
    r_limit_cpu   int32 CPU tensor [1]: the step's maximum rank, computed on the host from the
                  request ranks. Reading it is not a GPU sync.
    scale_table   optional fp32 [r_max + 1] per-rank scale (only for adapters whose B was folded
                  with one alpha / r for every rank); None for alpha == r adapters.
    max_tokens    the engine's max_num_batched_tokens (sizes the shared split-K workspace).

Under CUDA-graph capture the kernels are recorded with r_limit = r_max: a graph is replayed for
every later step whatever its ranks, so it must cover every rank up to r_max. Ranks are data
(read from token_ranks at replay), so a replay at lower ranks is still exact; it only computes
the padded rank rows that an eager call at that step would skip. Outside capture (eager mode,
or batches larger than the largest captured size) r_limit is the step's own maximum rank.

The launch itself is rankfuse.kernels.rankfuse_fused.RF1Launcher (one per layer, created on the
first call, before any capture since vLLM warms up every shape first).
"""

import torch

from rankfuse.kernels.rankfuse_fused import RF1Launcher

_LIB = torch.library.Library("rankfuse", "FRAGMENT")
# (A_t ptr, B ptr, slice sizes, max_tokens, scale ptr) -> launcher. Each entry holds A_t and B,
# so an address cannot be reused by another tensor while its entry exists.
_LAUNCHERS: dict[tuple, RF1Launcher] = {}


def launcher_for(A_t, B, slice_sizes, max_tokens, scale_table=None) -> RF1Launcher:
    key = (A_t.data_ptr(), B.data_ptr(), tuple(slice_sizes), max_tokens,
           None if scale_table is None else scale_table.data_ptr())
    launcher = _LAUNCHERS.get(key)
    if launcher is None:
        launcher = _LAUNCHERS[key] = RF1Launcher(A_t, B, slice_sizes, max_tokens=max_tokens, scale_table=scale_table)
    return launcher


def _lora_linear(
    y: torch.Tensor,
    x: torch.Tensor,
    A_t: torch.Tensor,
    B: torch.Tensor,
    token_ranks: torch.Tensor,
    r_limit_cpu: torch.Tensor,
    slice_sizes: list[int],
    scale_table: torch.Tensor | None,
    add: bool,
    max_tokens: int,
) -> None:
    if x.size(0) == 0:
        return
    r_limit = int(r_limit_cpu[0])
    if r_limit > 0 and torch.cuda.is_current_stream_capturing():
        r_limit = A_t.shape[1]
    if not x.is_contiguous():
        x = x.contiguous()
    launcher_for(A_t, B, slice_sizes, max_tokens, scale_table)(x, token_ranks, y, r_limit=r_limit, add=add)


def _lora_linear_fake(y, x, A_t, B, token_ranks, r_limit_cpu, slice_sizes, scale_table, add, max_tokens) -> None:
    return None


# Registered the way vLLM registers its own ops (vllm.utils.torch_utils.direct_register_custom_op):
# define + impl on a Library, which costs less per call than torch.library.custom_op.
_LIB.define(
    "lora_linear(Tensor(a!) y, Tensor x, Tensor A_t, Tensor B, Tensor token_ranks, Tensor r_limit_cpu, "
    "int[] slice_sizes, Tensor? scale_table, bool add, int max_tokens) -> ()"
)
_LIB.impl("lora_linear", _lora_linear, "CUDA")
_LIB._register_fake("lora_linear", _lora_linear_fake)

lora_linear = torch.ops.rankfuse.lora_linear
