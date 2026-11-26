"""Prefix paging behind vLLM: the plugin's page manager, layout and adapter packing.

Built on `rankfuse.paging.model_pages` (ModelPageLayout / ModelPageManager, copied from wt/paging,
see docs/VLLM_MULTI.md); this file only adds what the vLLM integration needs and does not change
those classes:

  ModuleLayout        a ModelPageLayout built from the LoRA modules vLLM actually wrapped (one
                      block per module, key (module index, module name)), so the frame matches
                      the model whatever its config spells the projections as.
  PluginPageManager   ModelPageManager plus
                        * frame 0 reserved and zeroed, and one extra page-table row (the dummy
                          row, every entry -> frame 0) that vLLM's warm-up / CUDA-graph capture
                          adapters map to, so captures read real (zero) memory at every rank;
                        * `unregister` (vLLM's CPU LRU drops adapters) with slot reuse: the row
                          goes back to -1 through the same fenced table copy as an eviction;
                        * `epoch`, bumped on every (un)registration, part of the step's
                          LoRAMapping so vLLM's "mapping unchanged" shortcut never reuses
                          metadata across a slot reassignment.
  pack_lora_model     vLLM LoRAModel (after _create_merged_loras_inplace + optimize, i.e. B
                      already scaled) -> pinned host pages [ceil(r/16), frame_elems] with B made
                      rank-major.
  pool_frames         frames from RANKFUSE_PAGE_FRAMES, else RANKFUSE_PAGE_POOL_GB (default 2).

Process-level registry: the engine core's scheduler and the worker share a process with vLLM's
UniProcExecutor (single GPU), so the scheduler reads the worker's manager through
`current_pager()` for the page budget (and, when deferring cold requests, for residency).
"""

from __future__ import annotations

import json
import math
import os
from typing import Hashable

import numpy as np
import torch

from rankfuse.multi import PAGE
from rankfuse.paging.bridge import FrameLayout
from rankfuse.paging.model_pages import ModelPageLayout, ModelPageManager

_PAGER: "PluginPageManager | None" = None


def paging_enabled() -> bool:
    return os.environ.get("RANKFUSE_PAGING", "0") == "1"


def current_pager() -> "PluginPageManager | None":
    return _PAGER


def set_current_pager(p: "PluginPageManager | None") -> None:
    global _PAGER
    _PAGER = p


def pool_frames(page_bytes: int) -> int:
    if os.environ.get("RANKFUSE_PAGE_FRAMES"):
        return int(os.environ["RANKFUSE_PAGE_FRAMES"])
    gb = float(os.environ.get("RANKFUSE_PAGE_POOL_GB", "2"))
    return max(1, int(gb * 1e9 // page_bytes))


def reserve_frames() -> int:
    return int(os.environ.get("RANKFUSE_PAGE_RESERVE", "0"))


class ModuleLayout(ModelPageLayout):
    """One FrameLayout block per wrapped LoRA module, in module order."""

    def __init__(self, blocks, page_rows: int = PAGE) -> None:  # noqa: D107 (no super().__init__: no ModelSpec)
        self.spec = None
        self.page_rows = page_rows
        self.blocks: dict[tuple[int, str], FrameLayout] = {}
        base = 0
        for key, h_in, slices in blocks:
            fl = FrameLayout(int(h_in), tuple(int(s) for s in slices), base=base, page_rows=page_rows)
            self.blocks[key] = fl
            base += fl.elems
        self.frame_elems = base


class PluginPageManager(ModelPageManager):
    def __init__(self, layout: ModelPageLayout, num_frames: int, *, num_adapter_slots: int, r_max: int,
                 dtype: torch.dtype = torch.float16, device="cuda") -> None:
        super().__init__(layout, num_frames + 1, num_adapter_slots=num_adapter_slots + 1, r_max=r_max,
                         dtype=dtype, device=device)
        self.usable_frames = num_frames
        self.zero_frame = self.pool.allocate("__zero__")
        self.pool.frames[self.zero_frame].zero_()
        self.dummy_row = num_adapter_slots
        self.cpu_page_table[self.dummy_row] = self.zero_frame
        self.page_table.copy_(self.cpu_page_table)
        self._free_slots = list(range(num_adapter_slots))
        self.epoch = 0
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @property
    def num_rows(self) -> int:
        """Page-table rows = adapter ids the kernels see (real slots + the dummy row)."""
        return self.num_adapter_slots

    def register(self, adapter_id: Hashable, host_pages: torch.Tensor, rank: int | None = None) -> int:
        if adapter_id in self._ids:
            raise ValueError(f"adapter {adapter_id!r} is already registered")
        if not self._free_slots:
            raise RuntimeError("adapter table is full; increase max_cpu_loras")
        if host_pages.dim() != 2 or host_pages.shape[1] != self.layout.frame_elems:
            raise ValueError(f"host pages must be [P, {self.layout.frame_elems}]")
        if host_pages.dtype != self.pool.pages.dtype:
            raise ValueError(f"host pages are {host_pages.dtype}, the pool is {self.pool.pages.dtype}")
        P = host_pages.shape[0]
        if P > self.pages_per_adapter:
            raise ValueError(f"adapter has {P} pages, the table holds {self.pages_per_adapter}")
        if self.device.type == "cuda" and not host_pages.is_pinned():
            host_pages = host_pages.pin_memory()
        slot = self._free_slots.pop(0)
        self._ids[adapter_id] = slot
        self._host[adapter_id] = host_pages.contiguous()
        self._ranks[adapter_id] = rank if rank is not None else P * self.page_rows
        self._resident[adapter_id] = 0
        self._frames[adapter_id] = []
        self.epoch += 1
        return slot

    def is_registered(self, adapter_id: Hashable) -> bool:
        return adapter_id in self._ids

    def adapter_rank(self, adapter_id: Hashable) -> int:
        return self._ranks[adapter_id]

    def unregister(self, adapter_id: Hashable) -> bool:
        if adapter_id not in self._ids:
            return False
        if adapter_id in self._in_use:
            raise RuntimeError(f"adapter {adapter_id!r} is in use by the current step")
        self.evict(adapter_id, 0)  # frames freed, row -> -1 after the adapter's last-use event
        slot = self._ids.pop(adapter_id)
        for d in (self._host, self._ranks, self._resident, self._frames, self._last_use):
            d.pop(adapter_id, None)
        self._lru.pop(adapter_id, None)
        self._free_slots.append(slot)
        self.epoch += 1
        return True

    # -------------------------------------------------------------------- trimming above the need
    # ModelPageManager never evicts any page of an adapter used by the current step. An adapter that
    # holds more pages than the step asks of it (resident from an earlier, higher-rank request) can
    # then block a load the page budget allows. Pages above an in-step adapter's need are not read by
    # this step, so they are evicted (top first, after the copy stream waits for the adapter's last
    # use) when nothing else can be.
    def ensure(self, ranks, *, allow_partial: bool = False, protect: bool = True):
        need = getattr(self, "_need_pages", {})
        if protect:
            need = dict(need)
            for a, r in ranks.items():
                need[a] = max(need.get(a, 0), math.ceil(r / self.page_rows))
            self._need_pages = need
        # Trim only for the step's own demand (protect=True), never for a prefetch: a prefetch of an
        # in-step adapter above its step rank would otherwise load and trim the same page forever
        # (found in the 20-frame engine run). The step's own ensure loads every adapter only up to
        # its need and trims only above it, so it cannot loop.
        self._trim_ok = protect
        try:
            return super().ensure(ranks, allow_partial=allow_partial, protect=protect)
        finally:
            self._trim_ok = False

    def begin_step(self, adapter_ids) -> None:
        super().begin_step(adapter_ids)
        self._need_pages = {}

    def end_step(self, stream=None) -> None:
        super().end_step(stream)
        self._need_pages = {}

    def _evict_one(self, keep: set, waits: list) -> bool:
        if super()._evict_one(keep, waits):
            return True
        if not getattr(self, "_trim_ok", False):
            return False
        need = getattr(self, "_need_pages", {})
        for a in self._lru:  # least recently used first
            if a in keep and self._resident[a] > need.get(a, self.pages_per_adapter):
                self._evict_top(a, waits)
                self.stats_trimmed = getattr(self, "stats_trimmed", 0) + 1
                return True
        return False

    def check_invariants(self) -> None:
        """ModelPageManager's invariants with the zero frame and the dummy row accounted for."""
        used = []
        for a, i in self._ids.items():
            row = self.cpu_page_table[i].tolist()
            k = self._resident[a]
            assert all(f >= 0 for f in row[:k]) and all(f == -1 for f in row[k:]), (a, row, k)
            assert row[:k] == self._frames[a]
            used += row[:k]
        assert self.zero_frame not in used
        assert len(used) == len(set(used)), "a frame is mapped twice"
        assert len(used) + 1 + self.pool.free_pages == self.pool.num_pages
        assert all(f == self.zero_frame for f in self.cpu_page_table[self.dummy_row].tolist())
        for s in self._free_slots:
            assert all(f == -1 for f in self.cpu_page_table[s].tolist()), f"free slot {s} has frames"

    def registered(self) -> list[Hashable]:
        return list(self._ids)

    def pages_needed(self, rank: int) -> int:
        return math.ceil(rank / self.page_rows)


# ------------------------------------------------------------------------------------------------
# vLLM LoRAModel -> host pages
# ------------------------------------------------------------------------------------------------


def _as_list(v):
    return list(v) if isinstance(v, (list, tuple)) else [v]


def lora_model_rank(lora_model, module_names) -> int:
    r = int(getattr(lora_model, "rank", 0) or 0)
    for name in module_names:
        lw = lora_model.get_lora(name)
        if lw is None:
            continue
        for a in _as_list(lw.lora_a):
            if a is not None:
                r = max(r, int(a.shape[0]))
    return r


def pack_lora_model(layout: ModuleLayout, keys: dict[str, tuple[int, str]], lora_model, get_weights,
                    *, dtype: torch.dtype, pin: bool = True) -> tuple[torch.Tensor, int]:
    """Pinned host pages of one vLLM LoRAModel. keys: module name -> layout block key;
    get_weights(lora_model, module_name) -> LoRALayerWeights | PackedLoRALayerWeights | None."""
    rank = lora_model_rank(lora_model, keys)
    weights = {}
    for name, key in keys.items():
        fl = layout.blocks[key]
        A = torch.zeros(fl.n_slices, rank, fl.h_in, dtype=dtype)
        Bs = [torch.zeros(rank, n, dtype=dtype) for n in fl.slice_sizes]
        lw = get_weights(lora_model, name)
        if lw is not None:
            las, lbs = _as_list(lw.lora_a), _as_list(lw.lora_b)
            scal = _as_list(lw.scaling) if isinstance(lw.scaling, (list, tuple)) else [lw.scaling] * len(las)
            if len(las) != fl.n_slices:
                raise ValueError(f"{name}: adapter has {len(las)} slices, the layer {fl.n_slices}")
            for s, (a, b, sc) in enumerate(zip(las, lbs, scal)):
                if a is None:
                    continue
                r = a.shape[0]
                A[s, :r] = a.to("cpu", dtype)
                bt = b.to("cpu", torch.float32).T  # [r, out]
                if sc not in (None, 1, 1.0):
                    bt = bt * float(sc)
                Bs[s][:r, : bt.shape[1]] = bt.to(dtype)
        weights[key] = (A, Bs)
    return layout.pack(weights, rank, dtype=dtype, pin=pin), rank


# ------------------------------------------------------------------------------------------------
# Scheduler side: adapter ranks without loading the adapter
# ------------------------------------------------------------------------------------------------

_CFG_RANK: dict[str, int | None] = {}


def adapter_config_rank(path: str | None) -> int | None:
    """r from a PEFT adapter directory's adapter_config.json (max over rank_pattern), cached."""
    if not path:
        return None
    if path in _CFG_RANK:
        return _CFG_RANK[path]
    r = None
    try:
        with open(os.path.join(path, "adapter_config.json")) as f:
            cfg = json.load(f)
        r = int(cfg.get("r", 0)) or None
        pat = cfg.get("rank_pattern") or {}
        if pat:
            r = max([r or 0, *(int(v) for v in pat.values())])
    except (OSError, ValueError, TypeError):
        r = None
    _CFG_RANK[path] = r
    return r


def request_pages(lora_request, r_max: int, full_rank: int) -> tuple[int, int]:
    """(lora_int_id, pages) a request needs: ceil(min(rank, adapter r, r_max) / 16)."""
    rank = getattr(lora_request, "matryoshka_rank", None)
    rank = full_rank if rank is None else rank
    ar = adapter_config_rank(getattr(lora_request, "lora_path", None))
    rank = min(rank, r_max, ar if ar else r_max)
    return lora_request.lora_int_id, math.ceil(rank / PAGE)


def step_pages(requests, r_max: int, full_rank: int) -> dict[int, int]:
    need: dict[int, int] = {}
    for req in requests:
        lr = getattr(req, "lora_request", None)
        if lr is None or lr.lora_int_id <= 0:
            continue
        a, p = request_pages(lr, r_max, full_rank)
        if p > need.get(a, 0):
            need[a] = p
    return need


def max_rank_per_adapter(ids: np.ndarray, ranks: np.ndarray) -> dict[int, int]:
    """{id: max rank} over tokens with id > 0 and rank > 0."""
    m = (ids > 0) & (ranks > 0)
    if not m.any():
        return {}
    u, inv = np.unique(ids[m], return_inverse=True)
    mx = np.zeros(u.shape[0], dtype=np.int64)
    np.maximum.at(mx, inv, ranks[m])
    return {int(a): int(r) for a, r in zip(u, mx)}
