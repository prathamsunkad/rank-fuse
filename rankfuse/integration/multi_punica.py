"""MultiPunicaWrapper: per-step multi-adapter metadata for RF1-MA in place of vLLM's punica kernels.

vLLM keeps one Punica wrapper per model, shared by every LoRA layer, refilled once per step by
`update_metadata(mapping, lora_index_to_id, ...)`. This wrapper turns the step's token -> LoRA id
mapping (vLLM's) and token -> rank mapping (the plugin's, `index_rank_mapping`) into

    token_ranks  int32 [max_tokens]     rank of each token (0: no adapter), clamped to the
                                        adapter's own rank and to max_lora_rank
    order        int32 [max_tokens]     MultiMeta.order  (tokens sorted by adapter row)
    tiles        int32 [cap_max, 5]     MultiMeta.tiles  (RF1-MA tile table, mode "group")
    meta_cpu     int64 CPU [6]          (T, cap, bm, r_pad, active, mode)

static buffers (allocated once, refilled with pinned non-blocking copies on the compute stream, so
they are ordered after the previous step's kernels and never need a host sync), and replaces
`add_lora_linear` with `torch.ops.rankfuse.multi_lora_linear` (multi_ops.py) for every layer the
plugin prepared.

Adapter rows. The kernels index weights by row a of a page table: in dense mode (no paging) the
row is vLLM's slot (lora_index_to_id); in paged mode it is the page manager's slot for the
adapter (warm-up / capture dummy adapters map to the manager's zero row). `resolver` does that
mapping; the plugin sets it.

Padded token counts and CUDA graphs. A layer sees x of the step's padded size T (vLLM pads to its
graph bucket); the plugin passes T in the mapping (`num_tokens_padded`). Padded tokens get rank 0.
For every T that can be replayed (T <= graph_max_tokens) the metadata is built with
r_pad = max_lora_rank rounded to a power of two, a block size that depends on T only, and
capacity = tile_capacity(T, bm, N): what the captured launch was recorded with. Larger T always
runs eagerly and gets the step's own r_pad, block size and tile count.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU
from vllm.utils.gpu_sync_debug import gpu_sync_allowed

from rankfuse.integration import multi_ops
from rankfuse.integration.punica import FULL_RANK, rank_scale_table
from rankfuse.multi import _r_pad, build_meta, default_block_m, tile_capacity


def graph_block_m(T: int, r_pad: int) -> int:
    """Block size for graph-replayable steps: a function of T and r_pad only."""
    if T <= 128:
        return 16
    return 32 if r_pad <= 128 else 16


def step_meta(rows: np.ndarray, ranks: np.ndarray, T: int, graph_max: int, num_adapters: int, r_pad_full: int,
              mode: str | None = None):
    """Host MultiMeta (mode of multi_ops.KERNEL) of one step: rows/ranks [T] (padded, -1 / 0 for no LoRA).

    T <= graph_max (replayable): r_pad_full, graph_block_m(T) and tile_capacity(T, bm, N), what a
    graph captured at T was recorded with. Otherwise the step's own r_pad, block size and tile count."""
    if T <= graph_max:
        r_pad, bm = r_pad_full, graph_block_m(T, r_pad_full)
        cap = tile_capacity(T, bm, num_adapters, mode or multi_ops.MODE)
    else:
        r_pad = _r_pad(max(int(ranks.max()), 1))
        groups = len(np.unique(rows[rows >= 0]))
        bm, cap = default_block_m(T, groups, r_pad), None
    mode = mode or multi_ops.MODE
    return build_meta(torch.from_numpy(rows), torch.from_numpy(ranks), num_adapters=num_adapters, mode=mode,
                      device="cpu", bm=bm, r_pad=r_pad, capacity=cap)


class MultiPunicaWrapper(PunicaWrapperGPU):
    def __init__(self, max_num_batched_tokens: int, max_batches: int, device: torch.device | str, **kwargs):
        super().__init__(max_num_batched_tokens, max_batches, device, **kwargs)
        import os

        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_lora_rank = self.lora_config.max_lora_rank
        self.r_pad_full = _r_pad(self.max_lora_rank)
        self._token_ranks = torch.zeros(max_num_batched_tokens, dtype=torch.int32, device=device)
        self._order = torch.zeros(max_num_batched_tokens, dtype=torch.int32, device=device)
        self._tiles: torch.Tensor | None = None
        self._meta_cpu = torch.zeros(6, dtype=torch.int64)
        self.scale_mode = os.environ.get("RANKFUSE_RANK_SCALE", "none")
        self.num_adapters = 0
        self.resolver: Callable | None = None
        self.rankfuse_only = False
        self.paged = False
        self.graph_max_tokens = 0
        self.last_step: dict[str, Any] = {}  # host copy of the last step's (slots, ranks), for tests
        self.keep_last_step = os.environ.get("RANKFUSE_KEEP_STEP", "0") == "1"
        self.steps = 0

    def configure(self, num_adapters: int, resolver: Callable, paged: bool) -> None:
        self.num_adapters = num_adapters
        self.resolver = resolver
        self.paged = paged
        cap = tile_capacity(self.max_num_batched_tokens, 16, num_adapters, "adapter_bucket")  # largest mode
        self._tiles = torch.zeros(cap, 5, dtype=torch.int32, device=self.device)

    def scale_table(self, rows: int) -> torch.Tensor | None:
        t = rank_scale_table(self.scale_mode, self.max_lora_rank, self.device)
        return None if t is None else t.unsqueeze(0).expand(rows, -1).contiguous()

    # ------------------------------------------------------------------ per-step metadata

    def update_metadata(self, mapping, lora_index_to_id, max_loras, vocab_size, **kwargs):
        self.is_prefill = mapping.is_prefill
        if not self.rankfuse_only:
            if self.paged:
                raise RuntimeError("RankFuse paging needs every LoRA layer on RankFuse kernels")
            self._update_base_metadata(mapping, lora_index_to_id, max_loras, vocab_size)
            with gpu_sync_allowed():
                self.token_mapping_meta.prepare_tensors(self.token_lora_indices)
                self.prompt_mapping_meta.prepare_tensors(self.sampler_indices)
        self._update_multi_metadata(mapping, lora_index_to_id)

    def _update_multi_metadata(self, mapping, lora_index_to_id) -> None:
        ids = np.asarray(mapping.index_mapping, dtype=np.int64)
        n = ids.shape[0]
        ranks = getattr(mapping, "index_rank_mapping", ())
        ranks = np.asarray(ranks, dtype=np.int64) if len(ranks) == n else np.where(ids > 0, FULL_RANK, 0)
        T = max(int(getattr(mapping, "num_tokens_padded", 0) or 0), n)
        graph_max = int(getattr(mapping, "graph_max_tokens", 0) or 0)
        rows, cap_rank = self.resolver(ids, lora_index_to_id)  # row per token (-1 none), adapter rank
        rk = np.where(rows >= 0, np.minimum(np.minimum(ranks, cap_rank), self.max_lora_rank), 0)
        rows = np.where(rk > 0, rows, -1)
        if T > n:
            rows = np.concatenate([rows, np.full(T - n, -1, dtype=np.int64)])
            rk = np.concatenate([rk, np.zeros(T - n, dtype=np.int64)])
        active = bool(n and rk.max() > 0)
        self.steps += 1
        if self.keep_last_step:
            self.last_step = dict(rows=rows.copy(), ranks=rk.copy(), T=T, n=n, active=active)
        if not active or T == 0:
            self._meta_cpu.copy_(torch.tensor([T, 0, 16, 16, 0, 0]))
            return
        meta = step_meta(rows, rk, T, graph_max, self.num_adapters, self.r_pad_full)
        if meta.capacity > self._tiles.shape[0]:
            raise RuntimeError(f"RankFuse: {meta.capacity} tiles exceed the static table ({self._tiles.shape[0]})")
        self._order[:T].copy_(meta.order.pin_memory(), non_blocking=True)
        self._tiles[: meta.capacity].copy_(meta.tiles.pin_memory(), non_blocking=True)
        self._token_ranks[:T].copy_(torch.from_numpy(rk.astype(np.int32)).pin_memory(), non_blocking=True)
        self._meta_cpu.copy_(torch.tensor([T, meta.capacity, meta.bm, meta.r_pad, 1,
                                           multi_ops.KERNEL_NAMES.index(multi_ops.KERNEL)]))

    # ------------------------------------------------------------------ the kernel swap

    def add_lora_linear(self, y, x, lora_a_stacked, lora_b_stacked, scale, output_slices, *, buffer=None,
                        rankfuse_wid: int | None = None, rankfuse_tensors=None, **kwargs: Any) -> None:
        if rankfuse_wid is None:
            if self.paged:
                raise RuntimeError("RankFuse paging: a LoRA layer without RankFuse weights")
            return super().add_lora_linear(y, x, lora_a_stacked, lora_b_stacked, scale, output_slices,
                                           buffer=buffer, **kwargs)
        assert scale == 1.0, "vLLM folds the LoRA scaling into B at load time"
        A_pool, B_pool, page_table = rankfuse_tensors
        multi_ops.multi_lora_linear(
            y, x.view(-1, x.shape[-1]), A_pool, B_pool, page_table, self._token_ranks, self._order,
            self._tiles, self._meta_cpu, rankfuse_wid, kwargs.get("add_inputs", True),
        )
