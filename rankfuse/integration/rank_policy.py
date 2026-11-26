"""Per-request rank selection for RankFuse serving: the hook for an adaptive rank controller.

A request's Matryoshka rank is fixed once, at admission, when the OpenAI front end turns the
request into a MatryoshkaLoRARequest (rankfuse/integration/frontend.py). It never changes after
that: LoRA on q/k/v makes every K/V entry rank-dependent, so a request's KV cache is only valid at
the rank it was computed at (and the plugin puts the rank into the prefix-cache block hash).

Where the rank comes from, in order:

    1. the request:  `vllm_xargs: {"matryoshka_rank": 32}` (OpenAI extra body). Explicit ranks
                     always win; the policy is not consulted.
    2. the policy:   `choose_rank(state) -> int | None`, called for requests that name a
                     RankFuse adapter but no rank (or `"matryoshka_rank": "auto"`).
    3. None:         the whole adapter (vLLM's max_lora_rank).

The policy sees a RankState snapshot built from what the API server process already knows:
scheduler stats that the engine core sends every step (running / waiting requests, KV-cache
usage) and request timings the front end measures itself (TTFT and time per output token, as
exponential moving averages). Nothing in it needs a GPU sync or an extra RPC.

Select a policy with RANKFUSE_RANK_POLICY:

    unset / "none"            no policy: requests without a rank get the full adapter
    "fixed:<r>"               FixedRankPolicy(r): every request without a rank gets r
    "module.path:factory"     factory(**kwargs) -> object with choose_rank(state); kwargs come
                              from RANKFUSE_RANK_POLICY_ARGS (a JSON object)

Only the hook and the trivial fixed policy live here; the load-and-difficulty policy of part C
plugs in through "module.path:factory".
"""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class RankState:
    """What a rank policy can read when a request is admitted. Times in seconds."""

    now: float                      # time.monotonic() at admission
    adapter: str                    # lora_name of the requested adapter
    r_max: int                      # the adapter's full rank
    tiers: tuple[int, ...]          # ranks the policy may choose from (ascending, all <= r_max)
    prompt_tokens: int | None       # prompt length, when the front end knows it already
    max_tokens: int | None          # requested output budget
    # engine-core scheduler stats (latest step the front end has seen; None before the first)
    num_running: int | None = None
    num_waiting: int | None = None
    kv_cache_usage: float | None = None   # fraction of KV blocks in use, 0..1
    stats_age_s: float | None = None      # how old the scheduler stats are
    # front-end measurements
    in_flight: int = 0                    # admitted by this front end, not finished
    ttft_ewma_s: float | None = None      # EWMA of time to first token of finished requests
    tpot_ewma_s: float | None = None      # EWMA of time per output token after the first
    extra: dict = field(default_factory=dict)   # request's other vllm_xargs (difficulty hints...)


class RankPolicy(Protocol):
    def choose_rank(self, state: RankState) -> int | None:
        """Return a rank in 1..state.r_max, or None for the full adapter."""


class FixedRankPolicy:
    """Every request without an explicit rank gets the same rank (clipped to the adapter)."""

    def __init__(self, rank: int):
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        self.rank = int(rank)

    def choose_rank(self, state: RankState) -> int | None:
        return min(self.rank, state.r_max)

    def __repr__(self) -> str:
        return f"FixedRankPolicy({self.rank})"


def load_policy(spec: str | None = None, args: str | None = None) -> RankPolicy | None:
    """Build the policy named by RANKFUSE_RANK_POLICY (or `spec`); None for no policy."""
    spec = os.environ.get("RANKFUSE_RANK_POLICY", "") if spec is None else spec
    args = os.environ.get("RANKFUSE_RANK_POLICY_ARGS", "") if args is None else args
    spec = spec.strip()
    if spec in ("", "none"):
        return None
    if spec.startswith("fixed:"):
        return FixedRankPolicy(int(spec.split(":", 1)[1]))
    if ":" not in spec:
        raise ValueError(f"RANKFUSE_RANK_POLICY must be 'none', 'fixed:<r>' or 'module:factory', got {spec!r}")
    mod, name = spec.split(":", 1)
    factory = getattr(importlib.import_module(mod), name)
    kwargs = json.loads(args) if args else {}
    policy = factory(**kwargs)
    if not callable(getattr(policy, "choose_rank", None)):
        raise TypeError(f"{spec} did not return an object with choose_rank(state)")
    return policy


def clip_rank(rank: int | None, r_max: int) -> int | None:
    """None stays None (full adapter); otherwise 1 <= rank <= r_max."""
    if rank is None:
        return None
    return max(1, min(int(rank), r_max))
