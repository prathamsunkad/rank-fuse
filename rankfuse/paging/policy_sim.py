"""Residency-policy simulator with real page sizes, a copy-time model and load waits.

Extends rankfuse/paging/sim.py (whose `Request` it reuses) with:
  * byte sizes from a model (`SizeModel`: bytes per rank row over the whole model, layers);
  * a copy-time model `CostModel` (pinned H2D bandwidth plus a fixed cost per cudaMemcpyAsync,
    both fitted from the A100 measurements in results/a100_2026-11-2X_paging/);
  * one copy engine shared by every load (FIFO), so a request waits for its own load and for
    loads queued before it; a hit on a page that is still in flight waits for it too;
  * time-averaged resident adapters, steady-state metrics after a warm-up prefix of the trace.

Policies (each request = (adapter, rank tier); every policy gets the same byte budget):

  vllm       stock vLLM with Matryoshka tiers registered as separate LoRAs: fixed slots of
             max_lora_rank = r_max rows, one slot per (adapter, tier) copy, whole-slot LRU. A load
             copies the tier's rows (vLLM copies the true rank into the padded slot) as one copy per
             layer, module and matrix (2 x 7 x layers calls).
  slora      S-LoRA style: each (adapter, tier) stored at its true rank in a byte pool (no padding,
             row granularity), whole-adapter LRU, one copy per layer and matrix (2 x layers calls;
             S-LoRA fuses a layer's projections). The pool is the adapters' share only (S-LoRA
             shares it with the KV cache; that coupling is not modelled).
  rf_whole   RankFuse, whole adapters: one slot per adapter at r_max serves every tier; whole-slot
             LRU; a load copies all r_max rows as r_max/16 page copies.
  rf_prefix  RankFuse prefix paging: 16-row whole-model pages, an adapter's resident pages are a
             prefix, a request loads only its missing pages (one copy each), eviction takes the
             top page of the least-recently-used other adapter.

Not modelled: request duration (an item in use by a running request could still be evicted;
LRU makes that rare), CPU-side cache misses (every adapter is in pinned host memory), compute.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np

from rankfuse.paging.sim import Request

PAGE_ROWS = 16
POLICIES = ("vllm", "slora", "rf_whole", "rf_prefix")
MODULES = 7


@dataclass(frozen=True)
class SizeModel:
    name: str
    row_bytes: int  # one rank row of A and B over every layer and projection
    num_layers: int
    r_max: int

    @property
    def page_bytes(self) -> int:
        return PAGE_ROWS * self.row_bytes

    @staticmethod
    def qwen25(size: str, r_max: int, dtype_bytes: int = 2) -> "SizeModel":
        # Elements per rank row per layer (A + B, 7 projections), see model_pages.ModelSpec.qwen25.
        per_layer, layers = {"0.5B": (22912, 24), "7B": (90112, 28)}[size]
        return SizeModel(f"Qwen2.5-{size}", per_layer * layers * dtype_bytes, layers, r_max)


@dataclass(frozen=True)
class CostModel:
    bw_gbps: float = 25.0  # pinned H2D, GB/s (1e9 B/s)
    call_us: float = 10.0  # fixed cost of one cudaMemcpyAsync (launch + DMA setup)

    def seconds(self, nbytes: float, calls: int) -> float:
        return calls * self.call_us * 1e-6 + nbytes / (self.bw_gbps * 1e9)


@dataclass
class PolicyResult:
    policy: str
    feasible: bool
    requests: int  # counted (after warm-up)
    hits: int
    misses: int
    partial_hits: int  # misses whose adapter already had a usable prefix (rf_prefix only)
    evictions: int
    bytes_copied: int
    copy_calls: int
    copy_busy_s: float
    wait_s: float  # sum over counted requests of (load ready - arrival), >= 0
    wait_p99_ms: float
    resident_adapters: float  # mean over counted requests of adapters with any resident data
    resident_copies: float  # mean resident items (adapter-tier copies, adapters or adapters)
    duration_s: float

    @property
    def hit_rate(self) -> float:
        return self.hits / self.requests if self.requests else float("nan")

    def row(self) -> dict:
        d = asdict(self)
        d["hit_rate"] = self.hit_rate
        d["mean_wait_ms"] = 1e3 * self.wait_s / self.requests if self.requests else float("nan")
        d["gb_copied"] = self.bytes_copied / 1e9
        d["copy_util"] = self.copy_busy_s / self.duration_s if self.duration_s > 0 else float("nan")
        return d


def tiers(r_max: int) -> list[int]:
    t = [1]
    while t[-1] * 2 < r_max:
        t.append(t[-1] * 2)
    return t + [r_max]


RANK_MIXES = {
    "uniform": lambda n: np.full(n, 1.0 / n),
    "skew_low": lambda n: np.arange(n, 0, -1) / (n * (n + 1) / 2),
    "skew_high": lambda n: np.arange(1, n + 1) / (n * (n + 1) / 2),
    "all_max": lambda n: np.eye(n)[-1],
}


def make_trace(requests: int, adapters: int, *, r_max: int, zipf_alpha: float, rank_mix: str,
               rate: float, seed: int = 0) -> list[Request]:
    """Poisson arrivals (rate req/s), Zipf(alpha) adapter popularity (ids shuffled so popularity is
    not the registration order), per-request rank tier drawn from `rank_mix` over tiers(r_max)."""
    rng = np.random.default_rng(seed)
    t = tiers(r_max)
    p_ad = 1.0 / np.arange(1, adapters + 1) ** zipf_alpha
    p_ad /= p_ad.sum()
    perm = rng.permutation(adapters)
    ad = perm[rng.choice(adapters, size=requests, p=p_ad)]
    pr = RANK_MIXES[rank_mix](len(t))
    rk = np.asarray(t)[rng.choice(len(t), size=requests, p=pr / pr.sum())]
    arr = np.cumsum(rng.exponential(1.0 / rate, size=requests))
    return [Request(int(a), int(r), float(x)) for a, r, x in zip(ad, rk, arr)]


def simulate_policy(trace: Sequence[Request], policy: str, budget_bytes: int, size: SizeModel,
                    cost: CostModel, warmup: int = 0) -> PolicyResult:
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    rb, L, r_max = size.row_bytes, size.num_layers, size.r_max
    n_pages_max = math.ceil(r_max / PAGE_ROWS)
    max_rank = max(r.rank for r in trace)
    # Feasibility: one request's item must fit in the budget.
    need = {
        "vllm": r_max * rb,
        "slora": max_rank * rb,
        "rf_whole": n_pages_max * size.page_bytes,
        "rf_prefix": math.ceil(max_rank / PAGE_ROWS) * size.page_bytes,
    }[policy]
    duration = trace[-1].arrival - trace[warmup].arrival if len(trace) > warmup + 1 else 0.0
    if need > budget_bytes:
        nan = float("nan")
        return PolicyResult(policy, False, len(trace) - warmup, 0, 0, 0, 0, 0, 0, nan, nan, nan, nan, nan, duration)

    hits = misses = partial = evictions = copied = calls = 0
    busy = wait = 0.0
    waits: list[float] = []
    res_ad_sum = res_items_sum = 0.0
    engine_free = 0.0
    counted = 0

    if policy == "rf_prefix":
        cap = budget_bytes // size.page_bytes
        pages: OrderedDict[int, list[float]] = OrderedDict()  # adapter -> ready time of each page
        used = 0
        for i, req in enumerate(trace):
            c = i >= warmup
            t = req.arrival
            k = math.ceil(req.rank / PAGE_ROWS)
            have = pages.get(req.adapter)
            cur = len(have) if have else 0
            if cur >= k:
                ready = max(have[:k])
                if c:
                    hits += 1
                pages.move_to_end(req.adapter)
            else:
                add = k - cur
                if c:
                    misses += 1
                    partial += cur > 0
                while used + add > cap:
                    victim = next(a for a in pages if a != req.adapter)
                    vp = pages[victim]
                    vp.pop()
                    used -= 1
                    if c:
                        evictions += 1
                    if not vp:
                        del pages[victim]
                if have is None:
                    have = pages[req.adapter] = []
                pages.move_to_end(req.adapter)
                start = max(t, engine_free)
                for _ in range(add):
                    start += cost.seconds(size.page_bytes, 1)
                    have.append(start)
                dt = start - max(t, engine_free)
                engine_free = start
                used += add
                ready = max(have[:k])
                if c:
                    copied += add * size.page_bytes
                    calls += add
                    busy += dt
            if c:
                w = max(0.0, ready - t)
                wait += w
                waits.append(w)
                res_ad_sum += len(pages)
                res_items_sum += len(pages)
                counted += 1
    else:
        if policy == "vllm":
            item_bytes = lambda rank: r_max * rb  # noqa: E731  (slot size)
            load = lambda rank: (rank * rb, 2 * MODULES * L)  # noqa: E731
            key_of = lambda req: (req.adapter, req.rank)  # noqa: E731
        elif policy == "slora":
            item_bytes = lambda rank: rank * rb  # noqa: E731
            load = lambda rank: (rank * rb, 2 * L)  # noqa: E731
            key_of = lambda req: (req.adapter, req.rank)  # noqa: E731
        else:  # rf_whole
            item_bytes = lambda rank: n_pages_max * size.page_bytes  # noqa: E731
            load = lambda rank: (n_pages_max * size.page_bytes, n_pages_max)  # noqa: E731
            key_of = lambda req: req.adapter  # noqa: E731
        cache: OrderedDict = OrderedDict()  # key -> (bytes, ready)
        per_adapter: dict[int, int] = {}
        used = 0
        for i, req in enumerate(trace):
            c = i >= warmup
            t = req.arrival
            key = key_of(req)
            ent = cache.get(key)
            if ent is not None:
                ready = ent[1]
                cache.move_to_end(key)
                if c:
                    hits += 1
            else:
                if c:
                    misses += 1
                nb = item_bytes(req.rank)
                while used + nb > budget_bytes:
                    vk, (vb, _) = cache.popitem(last=False)
                    used -= vb
                    va = vk[0] if isinstance(vk, tuple) else vk
                    per_adapter[va] -= 1
                    if not per_adapter[va]:
                        del per_adapter[va]
                    if c:
                        evictions += 1
                lb, lc = load(req.rank)
                start = max(t, engine_free)
                dt = cost.seconds(lb, lc)
                ready = engine_free = start + dt
                cache[key] = (nb, ready)
                used += nb
                per_adapter[req.adapter] = per_adapter.get(req.adapter, 0) + 1
                if c:
                    copied += lb
                    calls += lc
                    busy += dt
            if c:
                w = max(0.0, ready - t)
                wait += w
                waits.append(w)
                res_ad_sum += len(per_adapter)
                res_items_sum += len(cache)
                counted += 1

    p99 = float(np.percentile(waits, 99)) * 1e3 if waits else float("nan")
    return PolicyResult(
        policy, True, counted, hits, misses, partial, evictions, copied, calls, busy, wait, p99,
        res_ad_sum / max(counted, 1), res_items_sum / max(counted, 1), duration,
    )
