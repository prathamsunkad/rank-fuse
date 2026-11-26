"""RankFuse in vLLM's OpenAI server: a per-request Matryoshka rank over HTTP.

The Stage 1 plugin (vllm_plugin.py) carries a rank from a MatryoshkaLoRARequest down to the
kernel, but the OpenAI server only ever builds plain LoRARequests from the model name. This
module closes that gap in the API server process:

    POST /v1/completions  {"model": "<adapter>", ...,
                           "vllm_xargs": {"matryoshka_rank": 32}}       # or "auto" / omitted

`vllm_xargs` is vLLM's own extension field on completion and chat requests (a dict of
str/int/float; with the OpenAI SDK pass it as extra_body={"vllm_xargs": {...}}), so no protocol
or validation patch is needed. The model name stays the adapter's identity, exactly as the plugin
keys identity on lora_name (one lora_int_id, one CPU cache entry, one GPU slot for all ranks).

Why a request field and not adapter aliases (`name@r32`):
  * the rank is a per-request knob like max_tokens, not a different model: aliases multiply the
    model namespace by the number of ranks (/v1/models, routers and per-model metrics would see
    N x R models) and would need _check_model and the model listing patched as well;
  * an absent or "auto" field is the natural place for the rank controller to decide;
  * the identity stays on the adapter name, so aliases could never split one adapter's slot.

What install() patches (only with RANKFUSE_VLLM=1; it is a no-op otherwise):

  BaseServing._maybe_get_adapters   the one call every OpenAI endpoint makes to turn
      (entrypoints/serve/engine/      request.model into a LoRARequest. Runs once per request,
       serving.py)                    before the request reaches the engine: this is where the
                                      rank is FIXED (admission). Returns a MatryoshkaLoRARequest
                                      with the same name / id / path and matryoshka_rank =
                                      explicit rank, else choose_rank(state), else None (full).
  StatLoggerManager.record          keeps the latest SchedulerStats (running, waiting, KV usage)
      (v1/metrics/loggers.py)         the engine core sends, for the rank policy. Needs log stats
                                      on (vLLM's default for `vllm serve`).
  AsyncLLM.generate                 only when a policy is configured: in-flight count and
      (v1/engine/async_llm.py)        TTFT / TPOT moving averages, measured in the front end.

The engine-core and worker side (decoding the rank, rank in the prefix-cache key, the kernel)
is the existing plugin. Call install() in the API server process before it serves; for now
bench/e2e/serve_vllm.py does that (a later one-line change can call it from
vllm_plugin.register()).

Optional log: RANKFUSE_RANK_LOG=<path> appends one JSON line per admitted request
(model, rank, source = request | policy | full).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from rankfuse.integration.rank_policy import RankState, clip_rank, load_policy

RANK_KEY = "matryoshka_rank"
_EWMA = 0.2
_LOCK = threading.Lock()
_INSTALLED = False


class _FrontendState:
    def __init__(self):
        self.policy = None
        self.sched = None            # latest SchedulerStats
        self.sched_t = None
        self.in_flight = 0
        self.ttft = None
        self.tpot = None
        self.log_path = os.environ.get("RANKFUSE_RANK_LOG") or None
        self.adapter_info: dict[str, tuple[int, tuple[int, ...]]] = {}
        self.requests: dict[tuple, object] = {}
        self.counts: dict[tuple, int] = {}


STATE = _FrontendState()


def adapter_info(path: str) -> tuple[int, tuple[int, ...]]:
    """(r_max, tiers) of a PEFT adapter directory; tiers from matryoshka.json when present."""
    info = STATE.adapter_info.get(path)
    if info is None:
        r_max = int(json.loads((Path(path) / "adapter_config.json").read_text())["r"])
        meta = Path(path) / "matryoshka.json"
        tiers = None
        if meta.exists():
            tiers = json.loads(meta.read_text()).get("tiers")
        if not tiers:
            tiers, t = [], 1
            while t <= r_max:
                tiers.append(t)
                t *= 2
        info = STATE.adapter_info[path] = (r_max, tuple(sorted(int(t) for t in tiers if int(t) <= r_max)))
    return info


def parse_rank(value) -> int | None | str:
    """vllm_xargs value -> int rank, "auto", or None (absent)."""
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("", "auto"):
            return "auto"
        if v in ("full", "max"):
            return "full"
        return int(v)
    if isinstance(value, (int, float)):
        return int(value)
    raise ValueError(f"{RANK_KEY} must be an int, 'auto' or 'full', got {value!r}")


def resolve_rank(request, lora_request) -> tuple[int | None, str]:
    """The rank this request will run at, and where it came from."""
    xargs = getattr(request, "vllm_xargs", None) or {}
    want = parse_rank(xargs.get(RANK_KEY))
    r_max, tiers = adapter_info(lora_request.lora_path)
    if isinstance(want, int):
        return clip_rank(want, r_max), "request"
    if want == "full":
        return None, "full"
    if STATE.policy is not None:
        prompt = getattr(request, "prompt", None)
        n_prompt = len(prompt) if isinstance(prompt, list) and prompt and isinstance(prompt[0], int) else None
        now = time.monotonic()
        s = STATE.sched
        state = RankState(
            now=now, adapter=lora_request.lora_name, r_max=r_max, tiers=tiers, prompt_tokens=n_prompt,
            max_tokens=getattr(request, "max_tokens", None) or getattr(request, "max_completion_tokens", None),
            num_running=None if s is None else s.num_running_reqs,
            num_waiting=None if s is None else s.num_waiting_reqs + getattr(s, "num_skipped_waiting_reqs", 0),
            kv_cache_usage=None if s is None else s.kv_cache_usage,
            stats_age_s=None if STATE.sched_t is None else now - STATE.sched_t,
            in_flight=STATE.in_flight, ttft_ewma_s=STATE.ttft, tpot_ewma_s=STATE.tpot,
            extra={k: v for k, v in xargs.items() if k != RANK_KEY},
        )
        return clip_rank(STATE.policy.choose_rank(state), r_max), "policy"
    return None, "full"


def matryoshka_request(lora_request, rank: int | None):
    """A MatryoshkaLoRARequest with lora_request's fields and the given rank (cached)."""
    import msgspec

    from rankfuse.integration.request import MatryoshkaLoRARequest

    key = (lora_request.lora_name, lora_request.lora_int_id, lora_request.lora_path, rank)
    req = STATE.requests.get(key)
    if req is None:
        fields = msgspec.structs.asdict(lora_request)
        fields.pop(RANK_KEY, None)
        req = STATE.requests[key] = MatryoshkaLoRARequest(**fields, matryoshka_rank=rank)
    return req


def install() -> bool:
    """Patch the API server process (see the module docstring). Returns True if installed."""
    global _INSTALLED
    if os.environ.get("RANKFUSE_VLLM", "0") != "1":
        return False
    with _LOCK:
        if _INSTALLED:
            return True
        _install()
        _INSTALLED = True
    return True


def _install() -> None:
    from vllm.entrypoints.serve.engine import serving as base_serving
    from vllm.logger import init_logger
    from vllm.v1.metrics import loggers as metric_loggers

    logger = init_logger("vllm.rankfuse_frontend")
    STATE.policy = load_policy()

    orig_get = base_serving.BaseServing._maybe_get_adapters

    def _maybe_get_adapters(self, request, *args, **kwargs):
        lora_request = orig_get(self, request, *args, **kwargs)
        if lora_request is None:
            return None
        rank, source = resolve_rank(request, lora_request)
        key = (lora_request.lora_name, rank, source)
        STATE.counts[key] = STATE.counts.get(key, 0) + 1
        if STATE.log_path:
            with open(STATE.log_path, "a") as f:
                f.write(json.dumps({"t": time.time(), "model": request.model, "rank": rank, "source": source}) + "\n")
        return matryoshka_request(lora_request, rank)

    base_serving.BaseServing._maybe_get_adapters = _maybe_get_adapters

    orig_record = metric_loggers.StatLoggerManager.record

    def record(self, scheduler_stats=None, iteration_stats=None, *args, **kwargs):
        if scheduler_stats is not None:
            STATE.sched, STATE.sched_t = scheduler_stats, time.monotonic()
        return orig_record(self, scheduler_stats, iteration_stats, *args, **kwargs)

    metric_loggers.StatLoggerManager.record = record

    if STATE.policy is not None:
        _install_timing()
    logger.info("RankFuse front end: per-request rank via vllm_xargs.%s, policy %s", RANK_KEY, STATE.policy)


def _install_timing() -> None:
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    orig_generate = AsyncLLM.generate

    async def generate(self, prompt, sampling_params, request_id, *args, **kwargs):
        t0 = time.monotonic()
        first = last = None
        n = 0
        delta = getattr(sampling_params, "output_kind", None) == RequestOutputKind.DELTA
        STATE.in_flight += 1
        try:
            async for out in orig_generate(self, prompt, sampling_params, request_id, *args, **kwargs):
                if out.outputs:
                    k = len(out.outputs[0].token_ids)
                    n = n + k if delta else k
                    if n and first is None:
                        first = time.monotonic()
                    last = time.monotonic()
                yield out
        finally:
            STATE.in_flight -= 1
            if first is not None:
                ttft = first - t0
                STATE.ttft = ttft if STATE.ttft is None else (1 - _EWMA) * STATE.ttft + _EWMA * ttft
                if n > 1:
                    tpot = (last - first) / (n - 1)
                    STATE.tpot = tpot if STATE.tpot is None else (1 - _EWMA) * STATE.tpot + _EWMA * tpot

    AsyncLLM.generate = generate
