"""Seeded synthetic workloads: adapter shapes, batch shapes, and rank distributions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import torch

from rankfuse.interface import Batch, MatryoshkaLoRA

# Probability over n rank tiers, ordered smallest -> largest rank. The skews weight tiers
# linearly (n, n-1, ..., 1): with 4 tiers that is 0.4/0.3/0.2/0.1.
DISTRIBUTIONS: dict[str, Callable[[int], list[float]]] = {
    "all_min": lambda n: [1.0] + [0.0] * (n - 1),
    "all_max": lambda n: [0.0] * (n - 1) + [1.0],
    "uniform": lambda n: [1.0 / n] * n,
    "skew_low": lambda n: [(n - i) / (n * (n + 1) / 2) for i in range(n)],
    "skew_high": lambda n: [(i + 1) / (n * (n + 1) / 2) for i in range(n)],
}

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def default_tiers(r_max: int) -> list[int]:
    """Matryoshka granularities: powers of two from 1 up to r_max (1, 2, 4, ..., 64 for 64)."""
    tiers = [1]
    while tiers[-1] * 2 < r_max:
        tiers.append(tiers[-1] * 2)
    return tiers + [r_max] if r_max > 1 else [1]


@dataclass(frozen=True)
class Workload:
    num_requests: int
    tokens_per_request: int  # 1 = decode, >1 = prefill
    h_in: int = 4096
    h_out: int = 4096
    r_max: int = 64
    dist: str = "uniform"
    dtype: str = "fp16"
    seed: int = 0
    tiers: tuple[int, ...] | None = None  # defaults to default_tiers(r_max)

    @property
    def num_tokens(self) -> int:
        return self.num_requests * self.tokens_per_request

    @property
    def rank_tiers(self) -> list[int]:
        return list(self.tiers) if self.tiers else default_tiers(self.r_max)

    def as_row(self) -> dict:
        row = asdict(self)
        row["tiers"] = "/".join(map(str, self.rank_tiers))
        row["num_tokens"] = self.num_tokens
        return row


def sample_request_ranks(w: Workload, gen: torch.Generator) -> list[int]:
    tiers = w.rank_tiers
    probs = torch.tensor(DISTRIBUTIONS[w.dist](len(tiers)), dtype=torch.float64)
    idx = torch.multinomial(probs, w.num_requests, replacement=True, generator=gen)
    return [tiers[i] for i in idx.tolist()]


def make_lora(w: Workload, device: str | torch.device = "cpu") -> MatryoshkaLoRA:
    gen = torch.Generator().manual_seed(w.seed)
    # Scaled so activations stay O(1) in fp16.
    A_t = torch.randn(w.r_max, w.h_in, generator=gen) / w.h_in**0.5
    B = torch.randn(w.r_max, w.h_out, generator=gen) / w.r_max**0.5
    dt = DTYPES[w.dtype]
    return MatryoshkaLoRA(A_t=A_t.to(device, dt), B=B.to(device, dt))


def make_batch(w: Workload, device: str | torch.device = "cpu") -> Batch:
    gen = torch.Generator().manual_seed(w.seed + 1)
    ranks = sample_request_ranks(w, gen)
    x = torch.randn(w.num_tokens, w.h_in, generator=gen).to(device, DTYPES[w.dtype])
    return Batch.from_segments(x, [w.tokens_per_request] * w.num_requests, ranks)
