"""Seeded synthetic workloads: adapter shapes, batch shapes, and rank distributions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from rankfuse.interface import Batch, MatryoshkaLoRA

# Probability over rank tiers, ordered smallest -> largest rank.
DISTRIBUTIONS: dict[str, list[float]] = {
    "all_min": [1.0, 0.0, 0.0, 0.0],
    "all_max": [0.0, 0.0, 0.0, 1.0],
    "uniform": [0.25, 0.25, 0.25, 0.25],
    "skew_low": [0.4, 0.3, 0.2, 0.1],
    "skew_high": [0.1, 0.2, 0.3, 0.4],
}

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def default_tiers(r_max: int) -> list[int]:
    """Matryoshka granularities 1/8, 1/4, 1/2, full."""
    return [r_max // 8, r_max // 4, r_max // 2, r_max]


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
    probs = torch.tensor(DISTRIBUTIONS[w.dist], dtype=torch.float64)
    tiers = w.rank_tiers
    if len(tiers) != len(probs):
        raise ValueError(f"distribution {w.dist} has {len(probs)} tiers, workload has {len(tiers)}")
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
