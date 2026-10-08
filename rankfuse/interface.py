"""Shared contract that every implementation (baselines and RankFuse kernels) follows.

Conventions
-----------
* One MatryoshkaLoRA adapter is stored as two row-major tensors:
    A_t : [r_max, h_in]   (A transposed, so the rank-r prefix is rows [:r], contiguous)
    B   : [r_max, h_out]  (rank-r prefix is rows [:r], contiguous)
* A batch is flattened to tokens. Each token carries the rank its request asked for:
    x           : [T, h_in]
    token_ranks : [T] int32
  Decode (1 token per request) and prefill (many tokens per request) use the same format.
* Every implementation returns ONLY the LoRA delta  y = scale(r_i) * x_i @ A[:, :r_i] @ B[:r_i]
  with shape [T, h_out]. Adding it to the base projection is out of scope for the operator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import torch


@dataclass
class MatryoshkaLoRA:
    A_t: torch.Tensor  # [r_max, h_in]
    B: torch.Tensor  # [r_max, h_out]
    # Optional per-rank scaling (e.g. alpha / r). None means scale 1.0 for every rank.
    scale_per_rank: dict[int, float] | None = None

    def __post_init__(self) -> None:
        if self.A_t.shape[0] != self.B.shape[0]:
            raise ValueError(f"rank mismatch: A_t {tuple(self.A_t.shape)} vs B {tuple(self.B.shape)}")

    @property
    def r_max(self) -> int:
        return self.A_t.shape[0]

    @property
    def h_in(self) -> int:
        return self.A_t.shape[1]

    @property
    def h_out(self) -> int:
        return self.B.shape[1]

    def token_scales(self, token_ranks: torch.Tensor) -> torch.Tensor | None:
        """Per-token scale vector [T, 1], or None if no scaling is configured."""
        if self.scale_per_rank is None:
            return None
        table = torch.zeros(self.r_max + 1, dtype=torch.float32, device=token_ranks.device)
        for r, s in self.scale_per_rank.items():
            table[r] = s
        return table[token_ranks.long()].unsqueeze(1)


@dataclass
class Batch:
    x: torch.Tensor  # [T, h_in]
    token_ranks: torch.Tensor  # [T] int32
    # Request boundaries, kept for reporting and for kernels that want segment metadata.
    seg_starts: torch.Tensor = field(default=None)  # [num_requests + 1] int32

    @staticmethod
    def from_segments(x: torch.Tensor, seg_lens: list[int], seg_ranks: list[int]) -> "Batch":
        """Build a batch from per-request token counts and ranks."""
        if len(seg_lens) != len(seg_ranks):
            raise ValueError("seg_lens and seg_ranks must have the same length")
        if sum(seg_lens) != x.shape[0]:
            raise ValueError(f"sum(seg_lens)={sum(seg_lens)} but x has {x.shape[0]} tokens")
        token_ranks = torch.repeat_interleave(
            torch.tensor(seg_ranks, dtype=torch.int32), torch.tensor(seg_lens)
        ).to(x.device)
        starts = torch.zeros(len(seg_lens) + 1, dtype=torch.int32)
        starts[1:] = torch.cumsum(torch.tensor(seg_lens), 0)
        return Batch(x=x, token_ranks=token_ranks, seg_starts=starts.to(x.device))

    @property
    def num_tokens(self) -> int:
        return self.x.shape[0]

# defines what an implementation should look like
# takes in a tensor, the adapter, and the ranks for each token
# returns the LoRA delta tensor
class LoRAImpl(Protocol):
    name: str
    requires_cuda: bool

    def __call__(self, x: torch.Tensor, lora: MatryoshkaLoRA, token_ranks: torch.Tensor) -> torch.Tensor: ...


REGISTRY: dict[str, LoRAImpl] = {}

# called using @register("name")
# def padded(...)
def register(name: str, requires_cuda: bool = False) -> Callable:
    """Decorator: register a function as a LoRA implementation under `name`."""

    def wrap(fn: Callable) -> Callable:
        fn.name = name
        fn.requires_cuda = requires_cuda
        if name in REGISTRY:
            raise ValueError(f"duplicate implementation name: {name}")
        REGISTRY[name] = fn
        return fn

    return wrap

# does the scaling based on token_ranks and the provdied scaleperrank
def apply_scale(y: torch.Tensor, lora: MatryoshkaLoRA, token_ranks: torch.Tensor) -> torch.Tensor:
    s = lora.token_scales(token_ranks)
    return y if s is None else y * s.to(y.dtype)
