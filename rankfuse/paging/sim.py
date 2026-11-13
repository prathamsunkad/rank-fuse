"""Deterministic CPU simulator for adapter residency policies.

The simulator models adapter pages/slots rather than GPU tensors.  It is
deliberately independent of CUDA so traces can be generated and compared on
login or CPU nodes.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import random
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Request:
    adapter: int
    rank: int
    arrival: float = 0.0


@dataclass
class SimulationResult:
    requests: int
    hits: int
    misses: int
    evictions: int
    bytes_copied: int
    cold_starts: int
    resident_adapters: float

    @property
    def hit_rate(self) -> float:
        return self.hits / self.requests if self.requests else 1.0

    @property
    def cold_start_fraction(self) -> float:
        return self.cold_starts / self.requests if self.requests else 0.0


def poisson_trace(
    requests: int,
    adapters: int,
    *,
    seed: int = 0,
    rate: float = 1.0,
    popularity: str = "uniform",
    zipf_s: float = 1.2,
    ranks: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
    rank_weights: Sequence[float] | None = None,
) -> list[Request]:
    """Generate a reproducible request trace with Poisson inter-arrivals."""
    if requests < 0 or adapters <= 0 or rate <= 0:
        raise ValueError("requests must be non-negative, adapters and rate must be positive")
    if popularity not in {"uniform", "zipf"}:
        raise ValueError("popularity must be 'uniform' or 'zipf'")
    if not ranks:
        raise ValueError("ranks must not be empty")
    if popularity == "zipf" and zipf_s <= 0:
        raise ValueError("zipf_s must be positive")
    rng = random.Random(seed)
    weights = list(rank_weights) if rank_weights is not None else None
    if weights is not None and (len(weights) != len(ranks) or any(w < 0 for w in weights) or not any(weights)):
        raise ValueError("rank_weights must be non-negative and match ranks")
    out: list[Request] = []
    now = 0.0
    adapter_weights = (
        [1.0 / (i + 1) ** zipf_s for i in range(adapters)] if popularity == "zipf" else None
    )
    for _ in range(requests):
        now += rng.expovariate(rate)
        adapter = rng.choices(range(adapters), weights=adapter_weights, k=1)[0]
        rank = rng.choices(list(ranks), weights=weights, k=1)[0]
        out.append(Request(adapter, rank, now))
    return out


def _lru_touch(cache: OrderedDict[int, int], key: int, value: int) -> None:
    cache.pop(key, None)
    cache[key] = value


def simulate(
    trace: Iterable[Request],
    *,
    budget_bytes: int,
    r_max: int = 64,
    page_rows: int = 16,
    page_bytes: int = 81 * 1024**2,
    adapter_bytes: int = 323 * 1024**2,
    policy: str = "prefix",
) -> SimulationResult:
    """Replay a trace using one of the four documented residency policies."""
    if budget_bytes < 0 or r_max <= 0 or page_rows <= 0:
        raise ValueError("budget_bytes, r_max and page_rows must be positive")
    if policy not in {"vllm", "whole", "true_rank", "prefix"}:
        raise ValueError("unknown policy")
    trace = list(trace)
    slots = max(0, budget_bytes // adapter_bytes)
    pages = max(1, math.ceil(r_max / page_rows))
    page_capacity = max(0, budget_bytes // page_bytes)
    resident: OrderedDict[int, int] = OrderedDict()
    hits = misses = evictions = copied = cold = 0

    for req in trace:
        needed = min(r_max, max(1, req.rank))
        if policy == "prefix":
            capacity = page_capacity
            required = math.ceil(needed / page_rows)
            current = resident.get(req.adapter, 0)
            if current >= required:
                hits += 1
                _lru_touch(resident, req.adapter, current)
                continue
            misses += 1
            if current == 0:
                cold += 1
            additional = required - current
            while sum(resident.values()) + additional > capacity and resident:
                victim, victim_pages = next(iter(resident.items()))
                if victim_pages > 1:
                    resident[victim] = victim_pages - 1
                else:
                    del resident[victim]
                evictions += 1
            if sum(resident.values()) + additional > capacity:
                raise MemoryError("budget cannot hold one requested prefix")
            resident[req.adapter] = required
            copied += additional * page_bytes
            continue

        required = pages if policy in {"vllm", "whole"} else math.ceil(needed / page_rows)
        if policy == "true_rank":
            current = resident.get(req.adapter, 0)
            if current >= required:
                hits += 1
                _lru_touch(resident, req.adapter, current)
                continue
            misses += 1
            if current == 0:
                cold += 1
            additional = required - current
            capacity = max(0, budget_bytes // page_bytes)
            while sum(resident.values()) + additional > capacity and resident:
                resident.popitem(last=False)
                evictions += 1
            if sum(resident.values()) + additional > capacity:
                raise MemoryError("budget cannot hold one true-rank adapter")
            resident[req.adapter] = required
            copied += additional * page_bytes
            continue

        capacity = slots
        if req.adapter in resident and resident[req.adapter] >= required:
            hits += 1
            _lru_touch(resident, req.adapter, resident[req.adapter])
            continue
        misses += 1
        if req.adapter not in resident:
            cold += 1
        while len(resident) >= capacity and resident:
            resident.popitem(last=False)
            evictions += 1
        if capacity == 0:
            raise MemoryError("budget cannot hold one adapter slot")
        resident[req.adapter] = required
        copied += adapter_bytes if policy in {"vllm", "whole"} else required * page_bytes

    return SimulationResult(
        requests=len(trace),
        hits=hits,
        misses=misses,
        evictions=evictions,
        bytes_copied=copied,
        cold_starts=cold,
        resident_adapters=float(len(resident)),
    )
