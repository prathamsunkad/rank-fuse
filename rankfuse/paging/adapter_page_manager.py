"""Adapter prefix residency and LRU management."""

from __future__ import annotations

from collections import OrderedDict
import math
from typing import Hashable

import numpy as np
import torch

from rankfuse.paging.page_pool import PagePool


class AdapterPageManager:
    """Keep adapter pages resident as prefixes and evict suffixes only."""

    def __init__(
        self,
        pool: PagePool,
        page_rank: int = 16,
        *,
        num_adapter_slots: int | None = None,
        max_adapters: int | None = None,
        r_max: int | None = None,
        max_pages_per_adapter: int | None = None,
    ) -> None:
        if page_rank <= 0:
            raise ValueError("page_rank must be positive")
        slots = num_adapter_slots if num_adapter_slots is not None else max_adapters
        if slots is None or slots <= 0:
            raise ValueError("num_adapter_slots must be a positive integer")
        if r_max is not None:
            if r_max <= 0:
                raise ValueError("r_max must be positive")
            pages_per_adapter = r_max // page_rank
        elif max_pages_per_adapter is not None:
            if max_pages_per_adapter <= 0:
                raise ValueError("max_pages_per_adapter must be positive")
            pages_per_adapter = max_pages_per_adapter
        else:
            raise ValueError("r_max or max_pages_per_adapter must be provided (no default)")
        if pages_per_adapter <= 0:
            raise ValueError(f"pages_per_adapter must be positive, got {pages_per_adapter}")
        self.pool = pool
        self.page_rank = page_rank
        self.num_adapter_slots = slots
        self.max_adapters = slots
        self.r_max = r_max if r_max is not None else pages_per_adapter * page_rank
        self.max_pages_per_adapter = pages_per_adapter
        self.cpu_page_table = np.full(
            (slots, pages_per_adapter), -1, dtype=np.int32
        )
        self.page_table = torch.full(
            (slots, pages_per_adapter), -1,
            dtype=torch.int32, device=pool.device
        )
        self._adapters: dict[Hashable, tuple[torch.Tensor, torch.Tensor]] = {}
        self.cpu_store: dict[tuple[Hashable, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._ids: dict[Hashable, int] = {}
        self._resident: dict[Hashable, int] = {}
        self._lru: OrderedDict[Hashable, None] = OrderedDict()
        self._in_use: set[Hashable] = set()
        self.copy_stream = (
            torch.cuda.Stream(device=pool.device) if pool.device.type == "cuda" else None
        )
        self.copy_event = (
            torch.cuda.Event() if pool.device.type == "cuda" else None
        )

    def register(self, adapter_id: Hashable, A_t: torch.Tensor, B: torch.Tensor) -> None:
        if adapter_id in self._adapters:
            raise ValueError(f"adapter {adapter_id!r} is already registered")
        if A_t.ndim != 2 or B.ndim != 2 or A_t.shape[0] != B.shape[0]:
            raise ValueError("A_t and B must be rank-major 2-D tensors with equal rank")
        if A_t.device.type != "cpu" or B.device.type != "cpu":
            raise ValueError("adapter CPU page store requires CPU A_t and B tensors")
        pages = math.ceil(A_t.shape[0] / self.page_rank)
        if self.max_pages_per_adapter is not None and pages > self.max_pages_per_adapter:
            raise ValueError(f"adapter needs {pages} pages, limit is {self.max_pages_per_adapter}")
        if len(self._ids) >= self.max_adapters:
            raise RuntimeError("adapter table is full; increase max_adapters")
        self._ids[adapter_id] = len(self._ids)
        pin = False
        if self.pool.pin_memory or self.pool.device.type == "cuda":
            if torch.cuda.is_available():
                pin = True
            elif self.pool.pin_memory:
                raise RuntimeError("pinned host storage requested but unavailable")
        if pin:
            try:
                A_t, B = A_t.pin_memory(), B.pin_memory()
            except RuntimeError as exc:
                raise RuntimeError("pinned host storage requested but unavailable") from exc
        self._adapters[adapter_id] = (A_t.contiguous(), B.contiguous())
        self._resident[adapter_id] = 0
        self._lru[adapter_id] = None
        h_in = A_t.shape[1]
        h_out = B.shape[1]
        for page in range(pages):
            lo, hi = page * self.page_rank, min((page + 1) * self.page_rank, A_t.shape[0])
            frame_host = torch.zeros(
                (self.page_rank, h_in + h_out), dtype=A_t.dtype, device="cpu", pin_memory=pin
            )
            frame_host[: hi - lo, :h_in].copy_(A_t[lo:hi])
            frame_host[: hi - lo, h_in:].copy_(B[lo:hi])
            self.cpu_store[(adapter_id, page)] = frame_host

    add_adapter = register

    def ensure_resident(
        self,
        adapter_ids: list[Hashable] | tuple[Hashable, ...],
        needed_rank: int | dict[Hashable, int],
    ) -> list[tuple[int, ...]]:
        """Ensure prefixes for one step and mark those adapters non-evictable.

        ``needed_rank`` may be shared by all adapters or supplied per adapter.
        The returned tuples contain the physical frame for each required page.
        Call :meth:`end_step` after the compute stream has finished.
        """
        self._in_use = set(adapter_ids)
        copies: list[tuple[int, ...]] = []
        for adapter_id in adapter_ids:
            rank = needed_rank[adapter_id] if isinstance(needed_rank, dict) else needed_rank
            copies.append(self.ensure_prefix(adapter_id, rank))
        return copies

    def begin_step(self, adapter_ids: list[Hashable] | tuple[Hashable, ...]) -> None:
        self._in_use = set(adapter_ids)

    def end_step(self) -> None:
        self._in_use.clear()

    def ensure_prefix(self, adapter_id: Hashable, rank: int) -> tuple[int, ...]:
        A_t, _ = self._adapter(adapter_id)
        if not isinstance(rank, int) or not 0 < rank <= A_t.shape[0]:
            raise ValueError(f"rank must be in [1, {A_t.shape[0]}], got {rank!r}")
        needed = math.ceil(rank / self.page_rank)
        while self._resident[adapter_id] < needed:
            if not self.pool.free_pages:
                self._evict_one()
            page = self._resident[adapter_id]
            slot = self.pool.allocate((adapter_id, page))
            if self.copy_stream is None:
                self.pool.put(slot, self.cpu_store[(adapter_id, page)], owner=(adapter_id, page))
            else:
                with torch.cuda.stream(self.copy_stream):
                    self.pool.put(slot, self.cpu_store[(adapter_id, page)], owner=(adapter_id, page))
            adapter_idx = self._ids[adapter_id]
            self.cpu_page_table[adapter_idx, page] = slot
            if self.copy_stream is None:
                self.page_table[adapter_idx, page] = slot
            else:
                with torch.cuda.stream(self.copy_stream):
                    self.page_table[adapter_idx, page].fill_(slot)
            self._resident[adapter_id] += 1
            self._lru.pop(adapter_id, None)
            self._lru[adapter_id] = None
        if self.copy_stream is not None and self.copy_event is not None:
            with torch.cuda.stream(self.copy_stream):
                self.copy_event.record()
        self.touch(adapter_id, needed)
        adapter_idx = self._ids[adapter_id]
        return tuple(int(self.cpu_page_table[adapter_idx, i]) for i in range(needed))

    def wait_for_copies(self, stream: torch.cuda.Stream | None = None) -> None:
        """Make the compute stream wait for the most recent asynchronous loads."""
        if self.copy_event is None:
            return
        (stream or torch.cuda.current_stream()).wait_event(self.copy_event)

    def touch(self, adapter_id: Hashable, pages: int | None = None) -> None:
        self._adapter(adapter_id)
        n = self._resident[adapter_id] if pages is None else pages
        if not 0 <= n <= self._resident[adapter_id]:
            raise ValueError("pages touched must be within the resident prefix")
        self._lru.pop(adapter_id, None)
        self._lru[adapter_id] = None

    def evict(self, adapter_id: Hashable, from_page: int = 0) -> None:
        self._adapter(adapter_id)
        resident = self._resident[adapter_id]
        if not 0 <= from_page < resident:
            raise ValueError(f"from_page must be in [0, {resident}), got {from_page}")
        adapter_idx = self._ids[adapter_id]
        for page in range(resident - 1, from_page - 1, -1):
            slot = int(self.cpu_page_table[adapter_idx, page])
            self.pool.free(slot)
            self.cpu_page_table[adapter_idx, page] = -1
            if self.copy_stream is None:
                self.page_table[adapter_idx, page] = -1
            else:
                with torch.cuda.stream(self.copy_stream):
                    self.page_table[adapter_idx, page].fill_(-1)
        self._resident[adapter_id] = from_page
        if self._resident[adapter_id] == 0:
            self._lru.pop(adapter_id, None)

    def _evict_one(self) -> None:
        if not self._lru:
            raise RuntimeError("page pool is full and no resident adapter page can be evicted")
        for adapter_id in list(self._lru.keys()):
            if adapter_id not in self._in_use and self._resident[adapter_id] > 0:
                self.evict(adapter_id, from_page=self._resident[adapter_id] - 1)
                return
        raise RuntimeError("page pool is full and all resident pages are in use")

    def _adapter(self, adapter_id: Hashable) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return self._adapters[adapter_id]
        except KeyError as exc:
            raise KeyError(f"unknown adapter {adapter_id!r}") from exc
