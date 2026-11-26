"""Whole-model prefix pages: one frame holds 16 rank rows of one adapter for every layer and projection.

Layout
------
A page k of adapter a is rank rows 16k..16k+15 of every LoRA matrix of the adapter: for each layer
and each projection group (q/k/v merged, o, gate/up merged, down; or the 7 projections unmerged),
A rows [S, 16, h_in] and B rows [16, n_s], in the block layout of `bridge.FrameLayout`. The groups
sit one after another in the frame, layer-major:

    frame = [ L0 qkv | L0 o | L0 gate_up | L0 down | L1 qkv | ... | L{n-1} down ]

so one page is one contiguous range of `frame_elems` elements, on the host and on the GPU. The
pool is one tensor [F, frame_elems]; every (layer, group) is a `PagedWeights` over that same
tensor with frame stride frame_elems and its block's offsets, sharing one page table
[slots, r_max / 16]. This needs no change to `PagedWeights` or to the kernels (a per-layer pool
design would need one table per layer or per-layer pool pointers moved in lock step).

Sizes (fp16): Qwen2.5-0.5B 1.10 MB per rank row, page 17.6 MB; Qwen2.5-7B 5.05 MB per rank row,
page 80.7 MB (`ModelPageLayout(...).page_bytes`).

Manager
-------
`ModelPageManager` keeps each adapter's resident pages a prefix 0..k-1, loads a whole-model page
as one async copy (one cudaMemcpyAsync of `page_bytes` from pinned host memory into its frame) on
a copy stream, and updates the device page table with one more small copy per `ensure` call.
Eviction takes the top page of the least-recently-used adapter that the current step does not use.

Stream rules (see `ensure` and `end_step`):
  * the compute stream waits for `copy_done` (`wait_for_copies`) before kernels that need the
    new pages;
  * an evicted frame is reused only after the compute stream has passed the end of the last step
    that used its adapter (one event per step, recorded in `end_step`), so loads into free frames
    overlap with the running step and loads into frames evicted from long-idle adapters do too;
  * the pool tensor never moves; only frames' contents and the page table's contents change, so a
    CUDA graph captured once over these `PagedWeights` stays valid.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Hashable, Iterable, Mapping, Sequence

import torch

from rankfuse.multi import PAGE, PagedWeights
from rankfuse.paging.bridge import FrameLayout, paged_weights
from rankfuse.paging.page_pool import PagePool


# ------------------------------------------------------------------------------------------------
# Model and frame layout
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupSpec:
    """One projection group of a layer: S output slices sharing the input x."""

    name: str
    h_in: int
    slice_sizes: tuple[int, ...]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    num_layers: int
    groups: tuple[GroupSpec, ...]  # the same in every layer

    @staticmethod
    def transformer(name: str, num_layers: int, hidden: int, intermediate: int, n_heads: int, n_kv: int,
                    head_dim: int, merged: bool = True) -> "ModelSpec":
        q, kv = n_heads * head_dim, n_kv * head_dim
        if merged:
            groups = (
                GroupSpec("qkv", hidden, (q, kv, kv)),
                GroupSpec("o", q, (hidden,)),
                GroupSpec("gate_up", hidden, (intermediate, intermediate)),
                GroupSpec("down", intermediate, (hidden,)),
            )
        else:
            groups = (
                GroupSpec("q", hidden, (q,)), GroupSpec("k", hidden, (kv,)), GroupSpec("v", hidden, (kv,)),
                GroupSpec("o", q, (hidden,)), GroupSpec("gate", hidden, (intermediate,)),
                GroupSpec("up", hidden, (intermediate,)), GroupSpec("down", intermediate, (hidden,)),
            )
        return ModelSpec(name, num_layers, groups)

    @staticmethod
    def qwen25(size: str, merged: bool = True) -> "ModelSpec":
        """Qwen2.5 configs (config.json of Qwen/Qwen2.5-{0.5B,1.5B,7B}-Instruct)."""
        cfg = {
            "0.5B": (24, 896, 4864, 14, 2, 64),
            "1.5B": (28, 1536, 8960, 12, 2, 128),
            "7B": (28, 3584, 18944, 28, 4, 128),
        }[size]
        return ModelSpec.transformer(f"Qwen2.5-{size}", *cfg, merged=merged)


class ModelPageLayout:
    """Offsets of every (layer, group) block inside one whole-model frame."""

    def __init__(self, spec: ModelSpec, page_rows: int = PAGE) -> None:
        self.spec = spec
        self.page_rows = page_rows
        self.blocks: dict[tuple[int, str], FrameLayout] = {}
        base = 0
        for layer in range(spec.num_layers):
            for g in spec.groups:
                fl = FrameLayout(g.h_in, g.slice_sizes, base=base, page_rows=page_rows)
                self.blocks[(layer, g.name)] = fl
                base += fl.elems
        self.frame_elems = base

    def group(self, layer: int, name: str) -> FrameLayout:
        return self.blocks[(layer, name)]

    def page_bytes(self, dtype: torch.dtype = torch.float16) -> int:
        return self.frame_elems * torch.empty(0, dtype=dtype).element_size()

    def row_bytes(self, dtype: torch.dtype = torch.float16) -> float:
        return self.page_bytes(dtype) / self.page_rows

    def adapter_bytes(self, rank: int, dtype: torch.dtype = torch.float16) -> int:
        """Bytes of the pages holding rank rows 0..rank-1 (whole pages)."""
        return math.ceil(rank / self.page_rows) * self.page_bytes(dtype)

    def pack(self, weights: Mapping[tuple[int, str], tuple[torch.Tensor, Sequence[torch.Tensor]]],
             rank: int, *, dtype=torch.float16, pin: bool = True) -> torch.Tensor:
        """Host pages [ceil(rank / 16), frame_elems] from weights[(layer, group)] = (A [S, rank, h_in],
        [B_s [rank, n_s]]); rows past `rank` in the top page are zero."""
        pages = math.ceil(rank / self.page_rows)
        host = torch.zeros(pages, self.frame_elems, dtype=dtype, pin_memory=pin and torch.cuda.is_available())
        for key, fl in self.blocks.items():
            A, Bs = weights[key]
            A = A.to("cpu", dtype)
            Bs = [b.to("cpu", dtype) for b in Bs]
            for k in range(pages):
                lo, hi = k * self.page_rows, min((k + 1) * self.page_rows, rank)
                fl.pack(host[k, fl.base:fl.base + fl.elems], A, Bs, lo, hi)
        return host

    def random_weights(self, rank: int, *, seed: int = 0, dtype=torch.float16):
        """Seeded weights[(layer, group)] = (A [S, rank, h_in], [B_s]) for tests."""
        gen = torch.Generator().manual_seed(seed)
        out = {}
        for key, fl in self.blocks.items():
            A = (torch.randn(fl.n_slices, rank, fl.h_in, generator=gen) / fl.h_in**0.5).to(dtype)
            Bs = [(torch.randn(rank, n, generator=gen) / max(rank, 1)**0.5).to(dtype) for n in fl.slice_sizes]
            out[key] = (A, Bs)
        return out


# ------------------------------------------------------------------------------------------------
# Manager
# ------------------------------------------------------------------------------------------------


@dataclass
class Load:
    adapter: Hashable
    page: int
    frame: int


@dataclass
class PagingStats:
    page_loads: int = 0
    evictions: int = 0
    bytes_copied: int = 0
    copy_calls: int = 0  # cudaMemcpyAsync calls for pages (page-table copies not included)
    table_copies: int = 0
    ensure_calls: int = 0
    partial: int = 0  # ensure calls that stopped short of the requested rank (allow_partial)


class ModelPageManager:
    """Prefix residency of whole-model pages in one fixed GPU pool (separate from the KV cache).

    register(adapter, host_pages): pinned host pages [P, frame_elems] (ModelPageLayout.pack), or
        register_weights(adapter, weights, rank) to pack them here.
    begin_step(adapters) -> ensure({adapter: rank}) -> wait_for_copies() -> kernels -> end_step()

    The page table row of an adapter is `slot(adapter)`; it is the adapter id the kernels see.
    """

    def __init__(self, layout: ModelPageLayout, num_frames: int, *, num_adapter_slots: int, r_max: int,
                 dtype: torch.dtype = torch.float16, device="cuda") -> None:
        if r_max % layout.page_rows:
            raise ValueError(f"r_max must be a multiple of {layout.page_rows}")
        self.layout = layout
        self.page_rows = layout.page_rows
        self.r_max = r_max
        self.pages_per_adapter = r_max // layout.page_rows
        self.num_adapter_slots = num_adapter_slots
        # Her PagePool is the frame allocator: frames [F, frame_elems], CPU free list.
        self.pool = PagePool(num_frames, (layout.frame_elems,), dtype=dtype, device=device)
        self.device = self.pool.device
        cuda = self.device.type == "cuda"
        self.page_table = torch.full((num_adapter_slots, self.pages_per_adapter), -1, dtype=torch.int32,
                                     device=self.device)
        self.cpu_page_table = torch.full((num_adapter_slots, self.pages_per_adapter), -1, dtype=torch.int32)
        # Two pinned staging copies of the table: the async copy of one may still be reading it
        # while the next ensure() writes the other.
        self._stage = [self.cpu_page_table.clone().pin_memory() if cuda else self.cpu_page_table.clone()
                       for _ in range(2)]
        self._stage_ev = [torch.cuda.Event() if cuda else None for _ in range(2)]
        self._stage_used = [False, False]
        self._stage_i = 0
        self.copy_stream = torch.cuda.Stream(device=self.device) if cuda else None
        self.copy_done = torch.cuda.Event() if cuda else None
        self._host: dict[Hashable, torch.Tensor] = {}
        self._ranks: dict[Hashable, int] = {}
        self._ids: dict[Hashable, int] = {}
        self._resident: dict[Hashable, int] = {}
        self._frames: dict[Hashable, list[int]] = {}
        self._lru: OrderedDict[Hashable, None] = OrderedDict()
        self._in_use: set[Hashable] = set()
        self._last_use: dict[Hashable, torch.cuda.Event] = {}
        self._weights: dict[tuple[int, str], PagedWeights] = {}
        self.stats = PagingStats()

    # -------------------------------------------------------------------------------- adapters
    def register(self, adapter_id: Hashable, host_pages: torch.Tensor, rank: int | None = None) -> int:
        if adapter_id in self._ids:
            raise ValueError(f"adapter {adapter_id!r} is already registered")
        if len(self._ids) >= self.num_adapter_slots:
            raise RuntimeError("adapter table is full; increase num_adapter_slots")
        if host_pages.dim() != 2 or host_pages.shape[1] != self.layout.frame_elems:
            raise ValueError(f"host pages must be [P, {self.layout.frame_elems}]")
        if host_pages.dtype != self.pool.pages.dtype:
            raise ValueError(f"host pages are {host_pages.dtype}, the pool is {self.pool.pages.dtype}")
        P = host_pages.shape[0]
        if P > self.pages_per_adapter:
            raise ValueError(f"adapter has {P} pages, the table holds {self.pages_per_adapter}")
        if self.device.type == "cuda" and not host_pages.is_pinned():
            host_pages = host_pages.pin_memory()
        self._ids[adapter_id] = len(self._ids)
        self._host[adapter_id] = host_pages.contiguous()
        self._ranks[adapter_id] = rank if rank is not None else P * self.page_rows
        self._resident[adapter_id] = 0
        self._frames[adapter_id] = []
        return self._ids[adapter_id]

    def register_weights(self, adapter_id: Hashable, weights, rank: int) -> int:
        host = self.layout.pack(weights, rank, dtype=self.pool.pages.dtype, pin=self.device.type == "cuda")
        return self.register(adapter_id, host, rank)

    def slot(self, adapter_id: Hashable) -> int:
        return self._ids[adapter_id]

    def resident_pages(self, adapter_id: Hashable) -> int:
        return self._resident[adapter_id]

    def resident_rank(self, adapter_id: Hashable) -> int:
        return min(self._ranks[adapter_id], self._resident[adapter_id] * self.page_rows)

    def serveable_rank(self, adapter_id: Hashable, rank: int) -> int:
        """The rank a request can run at now without waiting (prefix property: any rank <= resident)."""
        return min(rank, self.resident_rank(adapter_id))

    @property
    def free_frames(self) -> int:
        return self.pool.free_pages

    @property
    def num_frames(self) -> int:
        return self.pool.num_pages

    # -------------------------------------------------------------------------------- steps
    def begin_step(self, adapter_ids: Iterable[Hashable]) -> None:
        self._in_use = set(adapter_ids)

    def end_step(self, stream: torch.cuda.Stream | None = None) -> None:
        """After the step's kernels are enqueued: frames of the step's adapters may be reused once
        the compute stream passes this point."""
        if self.device.type == "cuda" and self._in_use:
            ev = torch.cuda.Event()
            ev.record(stream or torch.cuda.current_stream())
            for a in self._in_use:
                self._last_use[a] = ev
        self._in_use = set()

    def wait_for_copies(self, stream: torch.cuda.Stream | None = None) -> None:
        if self.copy_done is not None:
            (stream or torch.cuda.current_stream()).wait_event(self.copy_done)

    def ensure_prefix(self, adapter_id: Hashable, rank: int, **kw) -> int:
        return self.ensure({adapter_id: rank}, **kw)[adapter_id]

    def prefetch(self, adapter_id: Hashable, rank: int) -> int:
        """Load toward `rank` using only free frames and idle adapters' pages; never raises."""
        return self.ensure({adapter_id: rank}, allow_partial=True, protect=False)[adapter_id]

    def ensure(self, ranks: Mapping[Hashable, int], *, allow_partial: bool = False,
               protect: bool = True) -> dict[Hashable, int]:
        """Make ceil(rank / 16) pages of every adapter resident; returns the rank each can run at
        (the requested rank, or less with allow_partial when the pool cannot hold it).

        Adapters in `ranks` (and those of the current step) are never evicted for each other.
        protect: also add them to the current step's in-use set.
        Copies are enqueued on the copy stream; call wait_for_copies() on the compute stream.
        """
        self.stats.ensure_calls += 1
        keep = set(self._in_use) | set(ranks)
        if protect:
            self._in_use |= set(ranks)
        loads: list[Load] = []
        waits: list[torch.cuda.Event] = []
        got: dict[Hashable, int] = {}
        for a, rank in ranks.items():
            if a not in self._ids:
                raise KeyError(f"unknown adapter {a!r}")
            if not 0 <= rank <= self._ranks[a]:
                raise ValueError(f"rank of {a!r} must be in [0, {self._ranks[a]}], got {rank}")
            need = math.ceil(rank / self.page_rows)
            while self._resident[a] < need:
                if not self.pool.free_pages and not self._evict_one(keep, waits):
                    if allow_partial:
                        self.stats.partial += 1
                        break
                    raise RuntimeError("page pool is full and every resident page is in use by this step")
                k = self._resident[a]
                f = self.pool.allocate((a, k))
                self._frames[a].append(f)
                self.cpu_page_table[self._ids[a], k] = f
                self._resident[a] += 1
                loads.append(Load(a, k, f))
            got[a] = min(rank, self.resident_rank(a))
            self._lru.pop(a, None)
            if self._resident[a]:
                self._lru[a] = None
        if loads or waits:
            self._issue(loads, waits)
        return got

    def evict(self, adapter_id: Hashable, from_page: int = 0) -> None:
        """Drop pages from_page.. of an adapter (top first); the frames are reusable after the
        adapter's last step (enforced on the next copy)."""
        waits: list[torch.cuda.Event] = []
        while self._resident[adapter_id] > from_page:
            self._evict_top(adapter_id, waits)
        self._issue([], waits)

    # -------------------------------------------------------------------------------- internals
    def _evict_top(self, a: Hashable, waits: list) -> None:
        k = self._resident[a] - 1
        f = self._frames[a].pop()
        self.pool.free(f)
        self.cpu_page_table[self._ids[a], k] = -1
        self._resident[a] = k
        self.stats.evictions += 1
        ev = self._last_use.get(a)
        if ev is not None and ev not in waits:
            waits.append(ev)
        if k == 0:
            self._lru.pop(a, None)

    def _evict_one(self, keep: set, waits: list) -> bool:
        for a in self._lru:
            if a not in keep and self._resident[a] > 0:
                self._evict_top(a, waits)
                return True
        return False

    def _issue(self, loads: list[Load], waits: list) -> None:
        frames = self.pool.frames
        if self.copy_stream is None:
            for ld in loads:
                frames[ld.frame].copy_(self._host[ld.adapter][ld.page])
            self.page_table.copy_(self.cpu_page_table)
        else:
            with torch.cuda.stream(self.copy_stream):
                for ev in waits:
                    self.copy_stream.wait_event(ev)
                for ld in loads:
                    frames[ld.frame].copy_(self._host[ld.adapter][ld.page], non_blocking=True)
                i = self._stage_i
                self._stage_i ^= 1
                if self._stage_used[i]:
                    self._stage_ev[i].synchronize()  # its last table copy (normally long done)
                self._stage[i].copy_(self.cpu_page_table)
                self.page_table.copy_(self._stage[i], non_blocking=True)
                self._stage_ev[i].record(self.copy_stream)
                self._stage_used[i] = True
                self.copy_done.record(self.copy_stream)
        self.stats.table_copies += 1
        self.stats.page_loads += len(loads)
        self.stats.copy_calls += len(loads)
        self.stats.bytes_copied += len(loads) * self.layout.page_bytes(self.pool.pages.dtype)

    # -------------------------------------------------------------------------------- kernels
    def weights(self, layer: int, group: str, scale_table: torch.Tensor | None = None) -> PagedWeights:
        """`PagedWeights` of one (layer, group), reading the shared pool through the shared table."""
        key = (layer, group)
        if scale_table is not None:
            return paged_weights(self.pool.frames, self.page_table, self.layout.group(layer, group),
                                 r_max=self.r_max, scale_table=scale_table)
        if key not in self._weights:
            self._weights[key] = paged_weights(self.pool.frames, self.page_table, self.layout.group(layer, group),
                                               r_max=self.r_max)
        return self._weights[key]

    def check_invariants(self) -> None:
        """Prefix invariant and frame accounting (CPU state only)."""
        used = []
        for a, i in self._ids.items():
            row = self.cpu_page_table[i].tolist()
            k = self._resident[a]
            assert all(f >= 0 for f in row[:k]) and all(f == -1 for f in row[k:]), (a, row, k)
            assert row[:k] == self._frames[a]
            used += row[:k]
        assert len(used) == len(set(used)), "a frame is mapped twice"
        assert len(used) + self.pool.free_pages == self.pool.num_pages
