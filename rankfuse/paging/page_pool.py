"""Fixed-capacity page storage used by adapter paging."""

from __future__ import annotations

from typing import Any

import torch


class PagePool:
    """A fixed page allocator with an optional combined projection layout.

    With ``h_in`` and ``h_out``, each frame is one contiguous
    ``[page_rows, h_in + h_out]`` tensor: A occupies the first ``h_in``
    columns and B the remaining columns.  A whole-model implementation can
    use the same frame allocator with a wider frame and keep per-layer
    ``(offset, width)`` entries in its host layout table.
    """

    def __init__(
        self,
        num_pages: int | None = None,
        page_shape: tuple[int, ...] | None = None,
        *,
        num_frames: int | None = None,
        page_rows: int = 16,
        h_in: int | None = None,
        h_out: int | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str = "cuda",
        pin_memory: bool = False,
    ) -> None:
        if num_pages is None:
            num_pages = num_frames
        if not isinstance(num_pages, int) or num_pages <= 0:
            raise ValueError(f"num_pages must be a positive integer, got {num_pages!r}")
        if h_in is not None or h_out is not None:
            if h_in is None or h_out is None or h_in <= 0 or h_out <= 0:
                raise ValueError("h_in and h_out must both be positive")
            if page_shape is not None:
                raise ValueError("page_shape cannot be combined with projection dimensions")
            page_shape = (page_rows, h_in + h_out)
            self.h_in, self.h_out = h_in, h_out
        else:
            self.h_in = self.h_out = None
        self.num_pages = num_pages
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("PagePool requested CUDA storage, but CUDA is unavailable")
        self.page_shape = tuple(page_shape) if page_shape is not None else None
        self.dtype = dtype
        self.pin_memory = pin_memory
        self._free = list(range(num_pages - 1, -1, -1))
        self._payloads: list[Any | None] = [None] * num_pages
        self._owners: list[Any | None] = [None] * num_pages
        self.cpu_store: dict[Any, Any] = {}
        self.pages: torch.Tensor | None = None
        if self.page_shape is not None:
            if any(not isinstance(n, int) or n <= 0 for n in self.page_shape):
                raise ValueError(f"page_shape must contain positive integers, got {page_shape!r}")
            self.pages = torch.empty(
                (num_pages, *self.page_shape), dtype=dtype or torch.float32, device=self.device
            )

    @property
    def free_pages(self) -> int:
        return len(self._free)

    def allocate(self, owner: Any = None) -> int:
        if not self._free:
            raise RuntimeError("page pool is full; evict a page before allocating another")
        slot = self._free.pop()
        self._owners[slot] = owner if owner is not None else True
        return slot

    def put(self, slot: int, payload: Any, *, owner: Any = None) -> None:
        self._check_slot(slot)
        if self._owners[slot] is None:
            raise RuntimeError(f"page slot {slot} is not allocated")
        if self.page_shape is not None:
            if isinstance(payload, torch.Tensor) and tuple(payload.shape) == self.page_shape:
                if payload.dtype != self.pages.dtype:  # type: ignore[union-attr]
                    raise ValueError(f"payload dtype {payload.dtype} does not match pool dtype {self.pages.dtype}")  # type: ignore[union-attr]
                self.pages[slot].copy_(payload, non_blocking=self.device.type == "cuda")  # type: ignore[index]
                self._payloads[slot] = self.pages[slot]
            elif self.h_in is not None and isinstance(payload, tuple):
                a, b = payload
                if a.shape[0] != b.shape[0] or a.shape[1] != self.h_in or b.shape[1] != self.h_out:
                    raise ValueError("projection page payload has incompatible dimensions")
                padded = torch.zeros(self.page_shape, dtype=a.dtype, device=a.device)
                padded[:, :self.h_in].copy_(a)
                padded[:, self.h_in:].copy_(b)
                payload = padded
                if payload.dtype != self.pages.dtype:  # type: ignore[union-attr]
                    raise ValueError(f"payload dtype {payload.dtype} does not match pool dtype {self.pages.dtype}")  # type: ignore[union-attr]
                self.pages[slot].copy_(payload, non_blocking=self.device.type == "cuda")  # type: ignore[index]
                self._payloads[slot] = self.pages[slot]
            else:
                raise ValueError(f"payload must have shape {self.page_shape}")
        else:
            self._payloads[slot] = self._move_payload(payload)
        if owner is not None:
            self._owners[slot] = owner
        self.cpu_store[self._owners[slot]] = self._host_payload(payload)

    def get(self, slot: int) -> Any:
        self._check_slot(slot)
        if self._owners[slot] is None:
            raise RuntimeError(f"page slot {slot} is not allocated")
        return self._payloads[slot]

    @property
    def frames(self) -> torch.Tensor:
        if self.pages is None:
            raise RuntimeError("frames require a tensor-backed page pool")
        return self.pages

    @property
    def page_stride(self) -> int:
        if self.page_shape is None:
            raise RuntimeError("page_stride requires a tensor-backed page pool")
        return self.page_shape[0] * self.page_shape[1]

    def free(self, slot: int) -> None:
        self._check_slot(slot)
        if self._owners[slot] is None:
            raise RuntimeError(f"page slot {slot} is already free")
        owner = self._owners[slot]
        self.cpu_store.pop(owner, None)
        self._owners[slot] = None
        self._payloads[slot] = None
        self._free.append(slot)

    def owner(self, slot: int) -> Any:
        self._check_slot(slot)
        return self._owners[slot]

    def _move_payload(self, payload: Any) -> Any:
        if isinstance(payload, torch.Tensor):
            if self.pin_memory and payload.device.type == "cpu":
                payload = payload.pin_memory()
            return payload.to(self.device, non_blocking=self.device.type == "cuda")
        if isinstance(payload, tuple):
            return tuple(self._move_payload(x) for x in payload)
        if isinstance(payload, list):
            return [self._move_payload(x) for x in payload]
        raise TypeError("page payloads must be tensors, tuples, or lists of tensors")

    def _host_payload(self, payload: Any) -> Any:
        if isinstance(payload, torch.Tensor):
            value = payload.detach().to("cpu").contiguous()
            return value.pin_memory() if self.pin_memory else value
        if isinstance(payload, tuple):
            return tuple(self._host_payload(x) for x in payload)
        if isinstance(payload, list):
            return [self._host_payload(x) for x in payload]
        raise TypeError("page payloads must be tensors, tuples, or lists of tensors")

    def _check_slot(self, slot: int) -> None:
        if not isinstance(slot, int) or not 0 <= slot < self.num_pages:
            raise IndexError(f"page slot must be in [0, {self.num_pages}), got {slot!r}")
