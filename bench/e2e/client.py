"""Open-loop, streaming replay of a trace against any OpenAI-compatible server.

    python bench/e2e/client.py --base-url http://127.0.0.1:8000 --trace traces/mix.jsonl \
        --rank-mode xargs --adapter-name 'matryoshka-{a}' --rates 1 2 4 8 inf \
        --max-duration 120 --min-requests 200 --max-requests 2000 --out runs/rankfuse --tag rankfuse

Every request of the trace is sent at its scheduled time t_i / rate, whatever the state of the
earlier ones (open loop; `inf` sends everything at once). Responses are streamed
(`stream: true`, `stream_options: {include_usage, continuous_usage_stats}`), so each chunk's
arrival time and cumulative output-token count are recorded. Output lengths are forced with
`ignore_eos: true` and `max_tokens` = the trace's output_len, and prompts are sent as token ids.

How a request names its adapter and rank depends on the system (--rank-mode):

    base         model = --base-model-name, no adapter (rank ignored)
    adapter      model = adapter name; rank ignored (stock vLLM serving one copy for everyone)
    tier-copies  model = adapter name + "-r{rank}" (stock vLLM / SGLang / S-LoRA with one
                 registered adapter per rank tier)
    xargs        model = adapter name, "vllm_xargs": {"matryoshka_rank": rank} (RankFuse;
                 rank null -> "auto": the server's rank policy decides)

Per rate it writes <out>/<tag>_rate<R>.jsonl (header + one record per request) and appends a
summary line (metrics.summarize + /metrics gauges) to <out>/summary.jsonl. While a rate runs, the
server's Prometheus /metrics are sampled every --metrics-interval s (KV-cache usage, running /
waiting requests, preemptions, prefix-cache hits); vLLM and SGLang names are recognised.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
import time
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402
import gen_trace as T  # noqa: E402

GAUGES = {   # our key -> Prometheus names (vLLM 0.32, SGLang)
    "kv_cache_usage": ("vllm:kv_cache_usage_perc", "sglang:token_usage"),
    "running": ("vllm:num_requests_running", "sglang:num_running_reqs"),
    "waiting": ("vllm:num_requests_waiting", "sglang:num_queue_reqs"),
    "preemptions": ("vllm:num_preemptions_total",),
    "prefix_hits": ("vllm:prefix_cache_hits_total",),
    "prefix_queries": ("vllm:prefix_cache_queries_total",),
}
_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)")


# ----------------------------------------------------------------------------------- backends

class OpenAICompletions:
    """POST /v1/completions with streaming (vLLM, SGLang)."""

    def __init__(self, base_url, rank_mode, adapter_name, base_model_name, extra_body=None):
        self.url = base_url.rstrip("/") + "/v1/completions"
        self.rank_mode = rank_mode
        self.adapter_name = adapter_name
        self.base_model_name = base_model_name
        self.extra = extra_body or {}

    def body(self, req: dict) -> dict:
        a, r = req["adapter"], req.get("rank")
        name = self.adapter_name.format(a=a)
        b = dict(prompt=req["prompt"], max_tokens=req["output_len"], temperature=0.0, ignore_eos=True,
                 stream=True, stream_options=dict(include_usage=True, continuous_usage_stats=True))
        if self.rank_mode == "base":
            b["model"] = self.base_model_name
        elif self.rank_mode == "adapter":
            b["model"] = name
        elif self.rank_mode == "tier-copies":
            b["model"] = f"{name}-r{r}"
        elif self.rank_mode == "xargs":
            b["model"] = name
            b["vllm_xargs"] = {"matryoshka_rank": "auto" if r is None else int(r)}
        else:
            raise ValueError(self.rank_mode)
        b.update(self.extra)
        return b

    async def send(self, session: aiohttp.ClientSession, req: dict, rec: dict) -> None:
        rec["sent"] = time.perf_counter()
        n_prev, chunk_t, chunk_n = 0, [], []
        async with session.post(self.url, json=self.body(req)) as resp:
            rec["status"] = resp.status
            if resp.status != 200:
                rec["error"] = (await resp.text())[:500]
                return
            async for raw in resp.content:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                now = time.perf_counter()
                msg = json.loads(data)
                if "error" in msg:
                    rec["error"] = str(msg["error"])[:500]
                    return
                rec.setdefault("id", msg.get("id"))
                usage = msg.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    n = usage["completion_tokens"]
                else:  # no usage stats: count one token per chunk with a choice
                    n = n_prev + (1 if msg.get("choices") else 0)
                if n > n_prev:
                    if rec["first"] is None:
                        rec["first"] = now
                    chunk_t.append(round(now - rec["sent"], 6))
                    chunk_n.append(n - n_prev)
                    n_prev = n
        rec["end"] = time.perf_counter()
        rec["n_out"] = n_prev
        rec["chunk_t"], rec["chunk_n"] = chunk_t, chunk_n   # chunk_t: seconds after `sent`


# ------------------------------------------------------------------------------------ /metrics

def parse_prometheus(text: str) -> dict:
    vals: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = _LINE.match(line)
        if m:
            vals[m.group(1)] = vals.get(m.group(1), 0.0) + float(m.group(3))   # sum over label sets
    out = {}
    for key, names in GAUGES.items():
        out[key] = next((vals[n] for n in names if n in vals), None)
    return out


async def sample_metrics(session, base_url, interval, samples, stop: asyncio.Event):
    url = base_url.rstrip("/") + "/metrics"
    while not stop.is_set():
        try:
            async with session.get(url) as resp:
                text = await resp.text()
            samples.append({"t": time.perf_counter()} | parse_prometheus(text))
        except Exception as e:  # noqa: BLE001 - a missed sample is not fatal
            samples.append({"t": time.perf_counter(), "error": str(e)[:200]})
        try:
            await asyncio.wait_for(stop.wait(), interval)
        except asyncio.TimeoutError:
            pass


# ---------------------------------------------------------------------------------------- run

def select(reqs: list[dict], rate: float, max_duration: float, min_requests: int, max_requests: int) -> list[dict]:
    """The first requests of the trace that arrive within max_duration at this rate, clamped."""
    if math.isinf(rate):
        k = max_requests
    else:
        k = sum(1 for r in reqs if r["t"] / rate <= max_duration)
        k = max(min_requests, min(k, max_requests))
    if k > len(reqs):
        raise ValueError(f"trace has {len(reqs)} requests, run needs {k}")
    return reqs[:k]


async def replay(backend, reqs, rate, base_url, metrics_interval, drain_timeout, max_conn=0) -> tuple[list, list]:
    timeout = aiohttp.ClientTimeout(total=None, sock_read=None)
    conn = aiohttp.TCPConnector(limit=max_conn, force_close=False)
    records = [dict(i=r["i"], adapter=r["adapter"], rank=r.get("rank"), prompt_len=r["prompt_len"],
                    output_len=r["output_len"], scheduled=None, sent=None, first=None, end=None, n_out=0,
                    status=None, error=None) for r in reqs]
    samples: list[dict] = []
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session, \
            aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as msession:
        stop = asyncio.Event()
        sampler = asyncio.create_task(sample_metrics(msession, base_url, metrics_interval, samples, stop))
        start = time.perf_counter() + 0.05
        tasks = []

        async def one(req, rec):
            try:
                await backend.send(session, req, rec)
            except asyncio.CancelledError:
                rec["error"] = "unfinished (drain timeout)"
                raise
            except Exception as e:  # noqa: BLE001 - recorded as a failed request
                rec["error"] = f"{type(e).__name__}: {e}"[:500]

        for req, rec in zip(reqs, records):
            at = start + (0.0 if math.isinf(rate) else req["t"] / rate)
            rec["scheduled"] = at
            delay = at - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one(req, rec)))
        done, pending = await asyncio.wait(tasks, timeout=drain_timeout)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        stop.set()
        await sampler
    return records, samples


async def warmup(backend, reqs, base_url, n, concurrency):
    """n requests (discarded), `concurrency` at a time, cycling the trace's ranks and adapters."""
    if n <= 0:
        return
    pool = reqs[: max(n, 1)]
    timeout = aiohttp.ClientTimeout(total=None)
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def one(req):
            async with sem:
                rec = dict(first=None, error=None)
                await backend.send(session, req | {"output_len": min(req["output_len"], 32)}, rec)
                if rec.get("error"):
                    raise RuntimeError(f"warm-up request failed: {rec['error']}")
        await asyncio.gather(*(one(r) for r in pool))


async def warmup_grid(backend, ranks, lengths, concurrencies, rank_mode):
    """Exercise every kernel shape class before measuring: for each prompt length x burst size, one
    burst per rank (homogeneous batches -> every rank bucket / R_PAD at every token-count class) and
    one mixed-rank burst. RankFuse's rank-skip mode compiles one Triton variant per (token block,
    R_PAD, tiled expand) combination; a trace's own warm-up does not reach all of them."""
    import random

    rng = random.Random(4321)
    timeout = aiohttp.ClientTimeout(total=None)
    sets = [[r] for r in ranks] + ([list(ranks)] if len(ranks) > 1 and rank_mode != "base" else [])
    if rank_mode in ("base", "adapter"):
        sets = [[ranks[-1]]]
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for L in lengths:
            for k in concurrencies:
                for rs in sets:
                    reqs = [dict(adapter=0, rank=rs[j % len(rs)], output_len=4,
                                 prompt=[rng.randrange(1000, 30000) for _ in range(L)]) for j in range(k)]
                    recs = [dict(first=None, error=None) for _ in reqs]
                    await asyncio.gather(*(backend.send(session, q, r) for q, r in zip(reqs, recs)))
                    bad = [r["error"] for r in recs if r.get("error")]
                    if bad:
                        raise RuntimeError(f"warm-up grid request failed: {bad[0]}")


async def wait_ready(base_url, timeout_s):
    url = base_url.rstrip("/") + "/health"
    t_end = time.time() + timeout_s
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s:
        while time.time() < t_end:
            try:
                async with s.get(url) as r:
                    if r.status == 200:
                        return
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(2)
    raise TimeoutError(f"{url} not ready after {timeout_s} s")


def run(args) -> list[dict]:
    header, reqs = T.load(args.trace)
    backend = OpenAICompletions(args.base_url, args.rank_mode, args.adapter_name, args.base_model_name,
                                json.loads(args.extra_body) if args.extra_body else None)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ttft_slos = tuple(args.ttft_slos)
    tpot_slos = tuple(args.tpot_slos)
    summaries = []
    if args.wait_ready:
        asyncio.run(wait_ready(args.base_url, args.wait_ready))
    if args.warmup:
        # warm-up requests are drawn from the END of the trace (never measured: runs use a prefix)
        asyncio.run(warmup(backend, reqs[::-1], args.base_url, args.warmup, args.warmup_concurrency))
    if args.warmup_grid:
        ranks = sorted({r["rank"] for r in reqs if r.get("rank") is not None}) or [None]
        t = time.time()
        asyncio.run(warmup_grid(backend, ranks, [16, 120, 400, 900, 2000], [1, 4, 16, 64], args.rank_mode))
        print(f"[{args.tag}] warm-up grid done in {time.time() - t:.0f} s ({len(ranks)} ranks)", flush=True)
    for rate in args.rates:
        sel = select(reqs, rate, args.max_duration, args.min_requests, args.max_requests)
        t_wall = time.time()
        records, samples = asyncio.run(replay(backend, sel, rate, args.base_url, args.metrics_interval,
                                              args.drain_timeout))
        s = M.summarize(records, ttft_slos, tpot_slos)
        s.update(tag=args.tag, rate=rate, trace=str(args.trace), trace_args=header["args"], wall_start=t_wall,
                 rank_mode=args.rank_mode, server=M.server_timeseries_summary(samples), notes=args.notes)
        rname = "inf" if math.isinf(rate) else f"{rate:g}"
        with open(out / f"{args.tag}_rate{rname}.jsonl", "w") as f:
            f.write(json.dumps(dict(kind="rankfuse-e2e-run", tag=args.tag, rate=rate, args=vars(args),
                                    samples=samples)) + "\n")
            for r in records:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
        with open(out / "summary.jsonl", "a") as f:
            f.write(json.dumps(s) + "\n")
        summaries.append(s)
        print(f"[{args.tag}] rate {rname}: {s['completed']}/{s['n']} ok, {s['req_per_s']:.2f} req/s, "
              f"{s['out_tok_per_s']:.0f} out tok/s, TTFT p50/p99 {_ms(s['ttft']['p50'])}/{_ms(s['ttft']['p99'])} ms, "
              f"TPOT p50/p99 {_ms(s['tpot']['p50'])}/{_ms(s['tpot']['p99'])} ms, "
              f"KV mean {_pct(s['server'].get('kv_cache_usage'))}, lag p99 {_ms(s['client_lag']['p99'])} ms"
              + (" [CLIENT LAG]" if s["client_lag_flag"] else ""), flush=True)
        if args.stop_when_saturated and _saturated(s, rate):
            print(f"[{args.tag}] saturated at rate {rname}; stopping the sweep", flush=True)
            break
    return summaries


def _saturated(s, rate) -> bool:
    """Throughput fell well below the offered rate, or most requests missed the loosest SLO."""
    if math.isinf(rate):
        return False
    loosest = max(s["slo"].values(), key=lambda v: v["attain"])["attain"]
    return s["req_per_s"] < 0.8 * rate or s["failed"] > 0.05 * s["n"] or loosest < 0.2


def _ms(x):
    return "-" if x is None else f"{x * 1e3:.1f}"


def _pct(d):
    return "-" if not d else f"{d['mean'] * 100:.0f}%"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--trace", required=True)
    p.add_argument("--rank-mode", choices=["base", "adapter", "tier-copies", "xargs"], required=True)
    p.add_argument("--adapter-name", default="matryoshka-{a}")
    p.add_argument("--base-model-name", default="base")
    p.add_argument("--extra-body", default="", help="JSON merged into every request body")
    p.add_argument("--rates", type=float, nargs="+", required=True, help="req/s; inf = all at once")
    p.add_argument("--max-duration", type=float, default=120.0)
    p.add_argument("--min-requests", type=int, default=200)
    p.add_argument("--max-requests", type=int, default=2000)
    p.add_argument("--drain-timeout", type=float, default=600.0)
    p.add_argument("--warmup", type=int, default=64)
    p.add_argument("--warmup-concurrency", type=int, default=16)
    p.add_argument("--warmup-grid", action="store_true",
                   help="also warm up every (prompt length x burst size x rank) class, see warmup_grid")
    p.add_argument("--wait-ready", type=float, default=0.0, help="poll /health up to this many seconds first")
    p.add_argument("--metrics-interval", type=float, default=1.0)
    p.add_argument("--ttft-slos", type=float, nargs="+", default=list(M.TTFT_SLOS))
    p.add_argument("--tpot-slos", type=float, nargs="+", default=list(M.TPOT_SLOS))
    p.add_argument("--stop-when-saturated", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--notes", default="")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass
    run(args)


if __name__ == "__main__":
    main()
