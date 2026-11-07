"""Seeded synthetic workloads: adapter shapes, batch shapes, and rank distributions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import torch

from rankfuse.interface import Batch, MatryoshkaLoRA

def _zipf(n: int, s: float) -> list[float]:
    w = [1.0 / (i + 1) ** s for i in range(n)]
    return [v / sum(w) for v in w]


# Probability over n rank tiers, ordered smallest -> largest rank. The skews weight tiers
# linearly (n, n-1, ..., 1): with 4 tiers that is 0.4/0.3/0.2/0.1.
DISTRIBUTIONS: dict[str, Callable[[int], list[float]]] = {
    "all_min": lambda n: [1.0] + [0.0] * (n - 1),
    "all_max": lambda n: [0.0] * (n - 1) + [1.0],
    "uniform": lambda n: [1.0 / n] * n,
    "skew_low": lambda n: [(n - i) / (n * (n + 1) / 2) for i in range(n)],
    "skew_high": lambda n: [(i + 1) / (n * (n + 1) / 2) for i in range(n)],
    # Low-rank heavy with a long tail of high ranks: weight 1/(i+1)^s for tier i. With r_max 256's
    # nine pow2 tiers, ranks 1-16 take 81% (s=1) or 95% (s=2) of requests and r_max 4% or 1%.
    "zipf_1": lambda n: _zipf(n, 1.0),
    "zipf_2": lambda n: _zipf(n, 2.0),
}

# Controlled mixes, not sampled: "low_frac_<pct>" puts round(pct% of requests) at the lowest tier
# and the rest at the top tier, always keeping at least one request at the top tier so the batch
# maximum rank is r_max (RF1 always pays full padding). Request order is shuffled by the seed.
# "low_frac_0" equals all_max; a single request is always at the top tier.
LOW_FRAC_PREFIX = "low_frac_"

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

# One continuous-batching step, a synthetic stand-in with no real trace behind it: "serving_<k>"
# adds k new prompts to num_requests decode requests. Prompt lengths are log-normal (median 256,
# sigma 1, at most 4096 tokens) and the step holds at most 8192 tokens, as with vLLM's
# max_num_batched_tokens. Prompt positions in the batch are drawn from the seed.
SERVING_PREFIX = "serving_"
SERVING_MEDIAN_TOKENS = 256
SERVING_MAX_PROMPT = 4096
SERVING_MAX_TOKENS = 8192


def default_tiers(r_max: int) -> list[int]:
    """Matryoshka granularities: powers of two from 1 up to r_max (1, 2, 4, ..., 64 for 64)."""
    tiers = [1]
    while tiers[-1] * 2 < r_max:
        tiers.append(tiers[-1] * 2)
    return tiers + [r_max] if r_max > 1 else [1]


# Named tier sets, so one config can sweep r_max without listing tiers per r_max.
TIER_PRESETS: dict[str, Callable[[int], list[int]]] = {
    "pow2": default_tiers,
    "fractions": lambda r: sorted({max(1, r // 8), max(1, r // 4), max(1, r // 2), r}),  # r/8, r/4, r/2, full
}


def is_known_dist(dist: str) -> bool:
    if dist in DISTRIBUTIONS:
        return True
    pct = dist.removeprefix(LOW_FRAC_PREFIX)
    return dist.startswith(LOW_FRAC_PREFIX) and pct.isdigit() and int(pct) <= 100


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
    tiers: tuple[int, ...] | str | None = None  # explicit ranks, a TIER_PRESETS name, or default_tiers(r_max)
    request_order: str = "random"  # "sorted": whole requests physically ordered by rank
    lengths: str | None = None  # "serving_<k>": see SERVING_PREFIX; None: tokens_per_request each

    @property
    def request_lengths(self) -> list[int]:
        """Tokens of every request, in sampling order (the batch order unless sorted by rank)."""
        if self.lengths is None:
            return [self.tokens_per_request] * self.num_requests
        if not self.lengths.startswith(SERVING_PREFIX) or not self.lengths.removeprefix(SERVING_PREFIX).isdigit():
            raise ValueError(f"unknown lengths preset {self.lengths!r}")
        if self.tokens_per_request != 1:
            raise ValueError("a serving step's decode requests have one token each")
        # Its own generator, so the same step shape appears whatever the rank mix.
        gen = torch.Generator().manual_seed(self.seed + 2)
        k = int(self.lengths.removeprefix(SERVING_PREFIX))
        drawn = (SERVING_MEDIAN_TOKENS * torch.randn(k, generator=gen, dtype=torch.float64).exp()).round().clamp(1, SERVING_MAX_PROMPT)
        lens, room = [1] * self.num_requests, SERVING_MAX_TOKENS - self.num_requests
        for n in drawn.tolist():
            lens.append(max(1, min(int(n), room - (k - 1))))
            room -= lens[-1]
            k -= 1
        return [lens[i] for i in torch.randperm(len(lens), generator=gen).tolist()]

    @property
    def num_tokens(self) -> int:
        return sum(self.request_lengths)

    @property
    def rank_tiers(self) -> list[int]:
        if isinstance(self.tiers, str):
            return TIER_PRESETS[self.tiers](self.r_max)
        return list(self.tiers) if self.tiers else default_tiers(self.r_max)

    def as_row(self) -> dict:
        row = asdict(self)
        row["tiers"] = "/".join(map(str, self.rank_tiers))
        row["num_tokens"] = self.num_tokens
        return row


def sample_request_ranks(w: Workload, gen: torch.Generator) -> list[int]:
    tiers = w.rank_tiers
    n = len(w.request_lengths)
    if not is_known_dist(w.dist):
        raise ValueError(f"unknown rank mix {w.dist!r}")
    if w.dist.startswith(LOW_FRAC_PREFIX):
        n_low = min(round(int(w.dist.removeprefix(LOW_FRAC_PREFIX)) / 100 * n), n - 1)
        ranks = [tiers[0]] * n_low + [tiers[-1]] * (n - n_low)
        return [ranks[i] for i in torch.randperm(n, generator=gen).tolist()]
    probs = torch.tensor(DISTRIBUTIONS[w.dist](len(tiers)), dtype=torch.float64)
    idx = torch.multinomial(probs, n, replacement=True, generator=gen)
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
    lens = w.request_lengths
    if w.request_order == "sorted":
        order = sorted(range(len(ranks)), key=ranks.__getitem__)
        ranks, lens = [ranks[i] for i in order], [lens[i] for i in order]
    elif w.request_order != "random":
        raise ValueError(f"unknown request order {w.request_order!r}")
    x = torch.randn(sum(lens), w.h_in, generator=gen).to(device, DTYPES[w.dtype])
    return Batch.from_segments(x, lens, ranks)
