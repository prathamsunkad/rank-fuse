"""Run the RankFuse multi-adapter kernels (RF1-MA, RF2-MA) directly on a `PagePool`.

`PagePool` (page_pool.py) gives one GPU tensor of frames [F, 16, W] and `AdapterPageManager`
(adapter_page_manager.py) keeps each adapter's pages resident as a prefix and maintains the
device page table [slots, pages]. `PagedWeights` (rankfuse/multi.py) describes weights to the
kernels as (pool, page_table, frame stride, per-slice offsets, row strides). This module builds
that description from a pool and a manager, so the kernels read the pool in place: no gather, no
copy, and the pool pointer never changes (CUDA-graph safe; only the page table's contents do).

Frame layout ("block")
----------------------
The kernels read A row j of a page at `frame * frame_stride + a_off[s] + j * h_in`, i.e. the 16 A
rows of one slice must be contiguous with row stride h_in. The stock `PagePool.put((A, B))` and
`AdapterPageManager.register` interleave A and B per row ([16, h_in + h_out]: row j = A_j | B_j),
which puts A rows h_in + h_out apart; the kernels have no A row stride, so they cannot read that
layout without a kernel change. The block layout keeps the frame's size and shape
([16, h_in + h_out] elements per frame, frame stride 16 * (h_in + h_out)) but orders the
elements as

    [ A slice 0: 16 x h_in | ... | A slice S-1 | B slice 0: 16 x n_0 | ... | B slice S-1 ]

so a_off[s] = 16 * s * h_in, b_off[s] = 16 * (S * h_in + n_0 + ... + n_{s-1}), b_ld[s] = n_s.
`BlockPageManager` is `AdapterPageManager` with `register` packing host frames this way.

Stream safety (fixed in `BlockPageManager`)
-------------------------------------------
The stock manager copies on a side stream and the compute stream waits for it
(`wait_for_copies`), but nothing makes the copy stream wait for the compute stream: evicting a
page frees its frame, and the next load can overwrite that frame (and set its page-table entry
to -1) while a kernel of the previous step is still reading it. `BlockPageManager.end_step`
records an event on the compute stream and `evict` makes the copy stream wait for it first.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Hashable, Sequence

import torch

from rankfuse.multi import PAGE, PagedWeights
from rankfuse.paging.adapter_page_manager import AdapterPageManager
from rankfuse.paging.page_pool import PagePool


@dataclass(frozen=True)
class FrameLayout:
    """Element offsets of one projection group (S slices sharing x) inside a 16-row page."""

    h_in: int
    slice_sizes: tuple[int, ...]
    base: int = 0  # element offset of this group inside a (possibly wider) frame
    page_rows: int = PAGE

    @property
    def n_slices(self) -> int:
        return len(self.slice_sizes)

    @property
    def a_off(self) -> tuple[int, ...]:
        return tuple(self.base + self.page_rows * s * self.h_in for s in range(self.n_slices))

    @property
    def b_off(self) -> tuple[int, ...]:
        b0 = self.base + self.page_rows * self.n_slices * self.h_in
        return tuple(b0 + self.page_rows * sum(self.slice_sizes[:s]) for s in range(self.n_slices))

    @property
    def b_ld(self) -> tuple[int, ...]:
        return tuple(self.slice_sizes)

    @property
    def elems(self) -> int:
        """Elements of one page of this group (A and B, every slice)."""
        return self.page_rows * (self.n_slices * self.h_in + sum(self.slice_sizes))

    @property
    def row_width(self) -> int:
        """Width W of a [page_rows, W] PagePool frame holding exactly this group."""
        return self.n_slices * self.h_in + sum(self.slice_sizes)

    def pack(self, out: torch.Tensor, A: torch.Tensor, B: Sequence[torch.Tensor], lo: int, hi: int) -> None:
        """Write rank rows lo..hi-1 (hi - lo <= page_rows) of A [S, r, h_in] and B[s] [r, n_s] into
        the flat page `out` (rows past hi - lo stay as they are: zero them first)."""
        n = hi - lo
        for s in range(self.n_slices):
            a0 = self.a_off[s] - self.base
            out[a0:a0 + n * self.h_in].view(n, self.h_in).copy_(A[s, lo:hi])
            b0 = self.b_off[s] - self.base
            ns = self.slice_sizes[s]
            out[b0:b0 + n * ns].view(n, ns).copy_(B[s][lo:hi])


def _as_slices(A: torch.Tensor, B, slice_sizes):
    """A [r, h_in] or [S, r, h_in]; B [r, h_out] (S = 1), [S, r, n_max] or a list of [r, n_s]."""
    if A.dim() == 2:
        A = A.unsqueeze(0)
    S = A.shape[0]
    if isinstance(B, torch.Tensor):
        if B.dim() == 2:
            B = B.unsqueeze(0)
        sizes = tuple(slice_sizes) if slice_sizes is not None else (B.shape[2],) * S
        Bs = [B[s, :, : sizes[s]] for s in range(S)]
    else:
        Bs = list(B)
        sizes = tuple(int(b.shape[1]) for b in Bs)
    if len(Bs) != S or any(b.shape[0] != A.shape[1] for b in Bs):
        raise ValueError("A and B must have the same slices and rank")
    return A, Bs, sizes


def make_pool(layout: FrameLayout, num_frames: int, *, dtype=torch.float16, device="cuda") -> PagePool:
    """A `PagePool` whose [16, W] frames hold exactly one page of `layout`."""
    if layout.base != 0:
        raise ValueError("a per-projection pool holds one group at offset 0")
    if layout.n_slices == 1:
        return PagePool(num_frames=num_frames, page_rows=layout.page_rows, h_in=layout.h_in,
                        h_out=layout.slice_sizes[0], dtype=dtype, device=device)
    return PagePool(num_frames, (layout.page_rows, layout.row_width), dtype=dtype, device=device)


class BlockPageManager(AdapterPageManager):
    """`AdapterPageManager` with block-layout host frames and a compute-to-copy stream fence.

    register(adapter, A, B): A [r, h_in] or [S, r, h_in]; B [r, h_out], [S, r, n_max] or a list
    of [r, n_s] (merged projections: S slices sharing x). The pool's frames must have
    `layout.elems` elements.
    """

    def __init__(self, pool: PagePool, layout: FrameLayout, **kw) -> None:
        if pool.pages is None or pool.pages[0].numel() != layout.elems:
            raise ValueError(f"pool frames must hold {layout.elems} elements (one page of the layout)")
        super().__init__(pool, layout.page_rows, **kw)
        self.layout = layout
        self._ranks: dict[Hashable, int] = {}
        self.compute_done = torch.cuda.Event() if self.copy_stream is not None else None
        self._compute_recorded = False

    def register(self, adapter_id: Hashable, A_t: torch.Tensor, B, slice_sizes=None) -> None:  # type: ignore[override]
        if adapter_id in self._adapters:
            raise ValueError(f"adapter {adapter_id!r} is already registered")
        A, Bs, sizes = _as_slices(A_t, B, slice_sizes)
        L = self.layout
        if A.shape[2] != L.h_in or sizes != L.slice_sizes:
            raise ValueError(f"adapter shape (h_in {A.shape[2]}, slices {sizes}) does not match the layout")
        r = A.shape[1]
        pages = math.ceil(r / self.page_rank)
        if pages > self.max_pages_per_adapter:
            raise ValueError(f"adapter needs {pages} pages, limit is {self.max_pages_per_adapter}")
        if len(self._ids) >= self.max_adapters:
            raise RuntimeError("adapter table is full; increase max_adapters")
        pin = self.pool.device.type == "cuda"
        self._ids[adapter_id] = len(self._ids)
        # The parent class keys rank checks on _adapters[id][0].shape[0]; keep (A, B) there.
        self._adapters[adapter_id] = (A[0].cpu(), Bs[0].cpu())
        self._ranks[adapter_id] = r
        self._resident[adapter_id] = 0
        self._lru[adapter_id] = None
        A, Bs = A.cpu(), [b.cpu() for b in Bs]
        for page in range(pages):
            lo, hi = page * self.page_rank, min((page + 1) * self.page_rank, r)
            frame = torch.zeros(self.pool.page_shape, dtype=self.pool.pages.dtype, pin_memory=pin)
            L.pack(frame.view(-1), A.to(frame.dtype), [b.to(frame.dtype) for b in Bs], lo, hi)
            self.cpu_store[(adapter_id, page)] = frame

    add_adapter = register

    def end_step(self, stream: torch.cuda.Stream | None = None) -> None:  # type: ignore[override]
        """Call after the step's kernels are enqueued: frames they read stay untouched until the
        compute stream passes this point."""
        if self.compute_done is not None:
            self.compute_done.record(stream or torch.cuda.current_stream())
            self._compute_recorded = True
        super().end_step()

    def evict(self, adapter_id: Hashable, from_page: int = 0) -> None:
        if self.copy_stream is not None and self._compute_recorded:
            self.copy_stream.wait_event(self.compute_done)
        super().evict(adapter_id, from_page)

    def slot(self, adapter_id: Hashable) -> int:
        """Row of the page table (the adapter id the kernels see)."""
        return self._ids[adapter_id]

    def resident_rank(self, adapter_id: Hashable) -> int:
        return min(self._ranks[adapter_id], self._resident[adapter_id] * self.page_rank)

    def host_weights(self, adapter_id: Hashable) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """A [S, r, h_in], B list of [r, n_s] rebuilt from the host pages (for references)."""
        L, r = self.layout, self._ranks[adapter_id]
        pages = math.ceil(r / self.page_rank)
        A = torch.zeros(L.n_slices, pages * L.page_rows, L.h_in, dtype=self.pool.pages.dtype)
        Bs = [torch.zeros(pages * L.page_rows, n, dtype=A.dtype) for n in L.slice_sizes]
        for k in range(pages):
            f = self.cpu_store[(adapter_id, k)].view(-1)
            rows = slice(k * L.page_rows, (k + 1) * L.page_rows)
            for s in range(L.n_slices):
                A[s, rows] = f[L.a_off[s]:L.a_off[s] + L.page_rows * L.h_in].view(L.page_rows, L.h_in)
                n = L.slice_sizes[s]
                Bs[s][rows] = f[L.b_off[s]:L.b_off[s] + L.page_rows * n].view(L.page_rows, n)
        return A[:, :r], [b[:r] for b in Bs]


def paged_weights(
    frames: torch.Tensor,
    page_table: torch.Tensor,
    layout: FrameLayout,
    *,
    r_max: int,
    scale_table: torch.Tensor | None = None,
) -> PagedWeights:
    """`PagedWeights` over a frame tensor [F, ...] (A pool = B pool = frames) and a page table."""
    if not frames.is_contiguous():
        raise ValueError("frames must be contiguous")
    if page_table.dtype != torch.int32 or page_table.shape[1] * PAGE < r_max:
        raise ValueError(f"page_table must be int32 with at least {r_max // PAGE} pages per adapter")
    frame_stride = frames[0].numel()
    if layout.base + layout.elems > frame_stride:
        raise ValueError("the layout does not fit in one frame")
    return PagedWeights(
        A_pool=frames, B_pool=frames, page_table=page_table,
        frame_stride_a=frame_stride, frame_stride_b=frame_stride,
        a_off=layout.a_off, b_off=layout.b_off, b_ld=layout.b_ld,
        h_in=layout.h_in, slice_sizes=layout.slice_sizes, r_max=r_max, scale_table=scale_table,
    )


def weights_from_manager(manager: BlockPageManager, scale_table: torch.Tensor | None = None) -> PagedWeights:
    """`PagedWeights` that read `manager.pool` in place through `manager.page_table`."""
    return paged_weights(manager.pool.frames, manager.page_table, manager.layout, r_max=manager.r_max,
                         scale_table=scale_table)


def reference_delta(x: torch.Tensor, A: torch.Tensor, Bs: Sequence[torch.Tensor], ranks, scale: float = 1.0):
    """fp32 delta of one adapter for tokens x [T, h_in] at per-token ranks (list or tensor)."""
    xf = x.float()
    out = torch.zeros(x.shape[0], sum(b.shape[1] for b in Bs), dtype=torch.float32, device=x.device)
    rk = torch.as_tensor(ranks).tolist()
    for r in sorted(set(rk)):
        if r <= 0:
            continue
        idx = torch.tensor([i for i, v in enumerate(rk) if v == r], device=x.device)
        c = 0
        for s, B in enumerate(Bs):
            n = B.shape[1]
            a = A[s, :r].to(x.device).float()
            b = B[:r].to(x.device).float()
            out[idx, c:c + n] = (xf[idx] @ a.T) @ b * scale
            c += n
    return out
