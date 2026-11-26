"""`torch.ops.rankfuse.multi_lora_linear`: the multi-adapter RankFuse op vLLM's LoRA layers call.

One call per dense LoRA layer per step: out (+)= RF1-MA delta (rankfuse/kernels/rankfuse_multi.py,
MULTI_REGISTRY['rf1_ma']) over the step's tile table, every token with its own adapter and rank.

Arguments (all step-varying state lives in tensors the wrapper owns; see multi_punica.py):

    y            [T, sum(slices)] base output, the delta is added in place (add=True)
    x            [T, h_in]
    A_pool,      the layer's PagedWeights pools (dense slot copies, or the shared page pool) and
    B_pool,      page table. Passed so the compiled graph records the dependency; the op reads
    page_table   the layer's PagedWeights from _WEIGHTS[wid] (offsets and strides are per-layer
                 constants fixed at load)
    token_ranks  int32 [max_tokens] static buffer; only the first T entries are read
    order        int32 [max_tokens] static buffer: MultiMeta.order of this step (first T entries)
    tiles        int32 [max_capacity, 5] static buffer: MultiMeta.tiles (first `cap` rows)
    meta_cpu     int64 CPU [6]: (T, cap, bm, r_pad, active, kernel index) of the metadata in the buffers
    wid          index into _WEIGHTS

Why the metadata is split this way. vLLM captures one CUDA graph per padded token count T (and
replays it for every later step of that size); a captured launch bakes in its grid and its
pointers. The buffers never move, the grid is (cap, split_k, slices) with cap = tile_capacity(T,
bm, N) fixed per T, and r_pad / bm are fixed per T for every T that can be replayed
(MultiPunicaWrapper.update_metadata), so `MultiMeta.load_`-style refills of the same buffers are
all a replay needs. Steps that always run eagerly (T above the largest captured size, or
enforce_eager) get a per-step meta (exact tile count, the step's own r_pad and block size), read
from meta_cpu at call time, which is not a GPU sync (CPU tensor).

`active` is 0 when no token of the step has an adapter: the call launches nothing (vLLM replays a
separate graph, captured with no LoRA, for such steps).
"""

from __future__ import annotations

import torch

from rankfuse.kernels.rankfuse_multi import multi_lora_splitk
from rankfuse.multi import MultiMeta, PagedWeights

import os

# Which multi-adapter kernel the op runs (MULTI_REGISTRY names; both go through
# rankfuse_multi.multi_lora, they differ in the metadata mode):
#   rf1_ma         mode "group": every tile at its adapter group's max rank; R_PAD constexpr, the
#                  limit masks A/B loads (fewer bytes at low rank, the MMA always R_PAD wide)
#   rf2_ma_splitk  mode "rank_sorted": tokens sorted by (adapter, rank), every tile branches at run
#                  time to the smallest rank bucket covering its own limit (BUCKETED), so lower ranks
#                  also skip MMA work under CUDA-graph replay (limits are read from the tile table)
# (2c30e56, kernels-v2): both run on rankfuse_fused.splitk_pass (MULTI); RF2-MA = RANK_SKIP per tile.
#   rf2_ma_widths  mode "adapter_bucket": groups of (adapter, width bucket), RANK_SKIP
#   (047f671 / 226368e) every tile reads its tokens' ranks at run time, also at graph replay; the
#   *_table variants take the tile width from the tile table's limit column instead (cheaper).
#   name -> (metadata mode, rank_skip, table_width)
_KERNELS = {
    "rf1_ma": ("group", False, False), "rf1_ma_table": ("group", False, True),
    "rf2_ma_splitk": ("rank_sorted", True, False), "rf2_ma_splitk_table": ("rank_sorted", True, True),
    "rf2_ma_widths": ("adapter_bucket", True, False),
}
KERNELS = {k: v[0] for k, v in _KERNELS.items()}
KERNEL = os.environ.get("RANKFUSE_MULTI_KERNEL", "rf1_ma")
if KERNEL not in KERNELS:
    raise ValueError(f"RANKFUSE_MULTI_KERNEL must be one of {sorted(KERNELS)}, got {KERNEL!r}")
MODE = KERNELS[KERNEL]
RANK_SKIP, TABLE_WIDTH = _KERNELS[KERNEL][1:]
KERNEL_NAMES = tuple(_KERNELS)  # meta_cpu[5] = index of the kernel the step's metadata was built for

_LIB = torch.library.Library("rankfuse", "FRAGMENT")
_WEIGHTS: list[PagedWeights] = []
# Test hook: HOOK(stage, y, x, wid, token_ranks, meta) with stage "pre" / "post" around the kernel (eager calls only;
# a CUDA-graph replay runs no Python). None in normal operation.
HOOK = None
_EMPTY: dict[torch.device, torch.Tensor] = {}


NAMES: list[str] = []  # module name of each registered PagedWeights (tests, logs)
CONTEXT: dict = {}  # the plugin's LoRA model manager and wrapper (tests)


def register_weights(W: PagedWeights, name: str = "") -> int:
    _WEIGHTS.append(W)
    NAMES.append(name)
    return len(_WEIGHTS) - 1


def weights(wid: int) -> PagedWeights:
    return _WEIGHTS[wid]


def _empty(dev) -> torch.Tensor:
    t = _EMPTY.get(dev)
    if t is None:
        t = _EMPTY[dev] = torch.zeros(1, dtype=torch.int32, device=dev)
    return t


def make_meta(T: int, cap: int, bm: int, r_pad: int, order: torch.Tensor, tiles: torch.Tensor,
              mode: str = MODE) -> MultiMeta:
    """A MultiMeta over views of the static buffers (only what multi_lora reads: order, tiles)."""
    e = _empty(order.device)
    return MultiMeta(mode=mode, T=T, bm=bm, r_pad=r_pad, order=order[:T], group_start=e, group_len=e,
                     group_adapter=e, group_rank_limit=e, tiles=tiles[:cap], num_tiles=cap, num_groups=0)


def _multi_lora_linear(
    y: torch.Tensor,
    x: torch.Tensor,
    A_pool: torch.Tensor,
    B_pool: torch.Tensor,
    page_table: torch.Tensor,
    token_ranks: torch.Tensor,
    order: torch.Tensor,
    tiles: torch.Tensor,
    meta_cpu: torch.Tensor,
    wid: int,
    add: bool,
) -> None:
    T = x.size(0)
    if T == 0:
        return
    T_meta, cap, bm, r_pad, active, mode = (int(v) for v in meta_cpu.tolist())
    if not active:
        return
    if T_meta != T:
        raise RuntimeError(
            f"RankFuse: LoRA metadata was built for {T_meta} tokens but the layer sees {T} "
            "(padded token count not passed to the wrapper)")
    W = _WEIGHTS[wid]
    if not x.is_contiguous():
        x = x.contiguous()
    mode_name, rank_skip, table_width = _KERNELS[KERNEL_NAMES[mode]]
    meta = make_meta(T, cap, bm, r_pad, order, tiles, mode_name)
    if HOOK is not None:
        HOOK("pre", y, x, wid, token_ranks[:T], meta)
    multi_lora_splitk(x, W, token_ranks[:T], meta, out=y, add=add, rank_skip=rank_skip, table_width=table_width)
    if HOOK is not None:
        HOOK("post", y, x, wid, token_ranks[:T], meta)


def _multi_lora_linear_fake(y, x, A_pool, B_pool, page_table, token_ranks, order, tiles, meta_cpu, wid, add) -> None:
    return None


_LIB.define(
    "multi_lora_linear(Tensor(a!) y, Tensor x, Tensor A_pool, Tensor B_pool, Tensor page_table, "
    "Tensor token_ranks, Tensor order, Tensor tiles, Tensor meta_cpu, int wid, bool add) -> ()"
)
_LIB.impl("multi_lora_linear", _multi_lora_linear, "CUDA")
_LIB._register_fake("multi_lora_linear", _multi_lora_linear_fake)

multi_lora_linear = torch.ops.rankfuse.multi_lora_linear
