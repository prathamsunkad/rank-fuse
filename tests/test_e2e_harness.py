"""CPU tests of the end-to-end harness (bench/e2e): traces, metrics, the client against a mock
OpenAI streaming server, the rank-policy hook and the front end's rank resolution."""

import asyncio
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

E2E = Path(__file__).resolve().parents[1] / "bench" / "e2e"
sys.path.insert(0, str(E2E))
import client as C  # noqa: E402
import gen_trace as T  # noqa: E402
import memory as MEM  # noqa: E402
import metrics as M  # noqa: E402


def _args(**kw):
    a = T.parser().parse_args(["--out", "x"])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


@pytest.mark.parametrize("arrival", ["poisson", "gamma:3", "onoff:5:15"])
def test_arrivals_unit_rate(arrival):
    t = T.arrivals(arrival, 20000, np.random.default_rng(0))
    assert t[0] == 0 and np.all(np.diff(t) >= 0)
    assert abs(len(t) / t[-1] - 1.0) < 0.1          # mean rate ~ 1 req/s
    if arrival.startswith("gamma"):
        g = np.diff(t)
        assert abs(g.std() / g.mean() - 3) < 0.3


def test_trace_roundtrip_and_determinism(tmp_path):
    a = _args(n=300, rank="skewed:1:8,16,32,64,128,256", adapters=4, zipf=1.2, shared_prefix=16)
    h, reqs = T.generate(a)
    h2, reqs2 = T.generate(a)
    assert reqs == reqs2
    T.write(tmp_path / "t.jsonl", h, reqs)
    h3, reqs3 = T.load(tmp_path / "t.jsonl")
    assert reqs3 == reqs and h3["stats"] == json.loads(json.dumps(h["stats"]))
    assert all(len(r["prompt"]) == r["prompt_len"] for r in reqs)
    ranks = [r["rank"] for r in reqs]
    assert ranks.count(8) > ranks.count(64) > 0
    by_a = {}
    for r in reqs:
        by_a.setdefault(r["adapter"], []).append(r)
    assert len(by_a[0]) > len(by_a[3])
    for rs in by_a.values():   # shared prefix per adapter
        k = min(16, *(r["prompt_len"] for r in rs))
        assert len({tuple(r["prompt"][:k]) for r in rs}) == 1


def test_select_scales_with_rate():
    reqs = [dict(t=float(i)) for i in range(1000)]
    assert len(C.select(reqs, 2.0, 100, 10, 1000)) == 201
    assert len(C.select(reqs, 0.1, 100, 50, 1000)) == 50
    assert len(C.select(reqs, math.inf, 100, 10, 300)) == 300


def test_metrics_definitions():
    rec = dict(i=0, rank=8, prompt_len=10, output_len=3, scheduled=0.0, sent=0.0, first=0.5, end=0.7, n_out=3,
               chunk_t=[0.5, 0.6, 0.7], chunk_n=[1, 1, 1], error=None)
    bad = dict(rec, i=1, error="boom", first=None, n_out=0)
    s = M.summarize([rec, bad], ttft_slos=(0.6,), tpot_slos=(0.2,))
    assert s["completed"] == 1 and s["failed"] == 1
    assert s["ttft"]["p50"] == pytest.approx(0.5)
    assert s["tpot"]["p50"] == pytest.approx(0.1)
    assert s["itl"]["p50"] == pytest.approx(0.1)
    assert s["slo"]["ttft0.6_tpot0.2"]["attain"] == pytest.approx(0.5)   # failed counts as a miss


def test_prometheus_parse():
    text = '# HELP x\nvllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.25\nvllm:num_requests_running{engine="0"} 7\n'
    g = C.parse_prometheus(text)
    assert g["kv_cache_usage"] == 0.25 and g["running"] == 7 and g["waiting"] is None


def test_memory_model_counts():
    d = MEM.dims(dict(hidden_size=896, intermediate_size=4864, num_hidden_layers=24, num_attention_heads=14,
                      num_key_value_heads=2))
    # Qwen2.5-0.5B at rank 64 is ~35M params (~70 MB in 16-bit), as the rank-64 adapter file
    assert MEM.lora_bytes(d, 64) == 64 * (10240 + 12672) * 24 * 2
    per = MEM.per_adapter_bytes(d, 256, [8, 16, 32, 64, 128, 256])
    assert per["stock vLLM (per-tier slots, padded)"] == 6 * per["RankFuse (one r_max copy)"]


# ------------------------------------------------------------------- client vs a mock server

async def _mock_server(port_box, seen):
    from aiohttp import web

    async def completions(request):
        body = await request.json()
        seen.append(body)
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for k in range(body["max_tokens"]):
            await asyncio.sleep(0.002)
            chunk = {"id": "cmpl-x", "choices": [{"text": "a", "index": 0}],
                     "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": k + 1}}
            await resp.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        await resp.write(b"data: [DONE]\n\n")
        return resp

    async def metrics(request):
        return web.Response(text='vllm:kv_cache_usage_perc{engine="0"} 0.5\n')

    async def health(request):
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_post("/v1/completions", completions)
    app.router.add_get("/metrics", metrics)
    app.router.add_get("/health", health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port_box.append(site._server.sockets[0].getsockname()[1])
    return runner


def test_client_against_mock(tmp_path):
    a = _args(n=40, rank="uniform:8,64", output_len="fixed:5", prompt_len="fixed:7")
    h, reqs = T.generate(a)
    T.write(tmp_path / "t.jsonl", h, reqs)
    seen = []

    async def go():
        box = []
        runner = await _mock_server(box, seen)
        base = f"http://127.0.0.1:{box[0]}"
        backend = C.OpenAICompletions(base, "xargs", "matryoshka-{a}", "base")
        await C.warmup(backend, reqs[::-1], base, 4, 2)
        recs, samples = await C.replay(backend, reqs, 200.0, base, 0.05, 30)
        await runner.cleanup()
        return recs, samples

    recs, samples = asyncio.run(go())
    assert all(r["error"] is None and r["n_out"] == 5 for r in recs)
    assert seen[-1]["vllm_xargs"]["matryoshka_rank"] in (8, 64) and seen[-1]["model"] == "matryoshka-0"
    assert seen[-1]["ignore_eos"] and seen[-1]["stream"]
    s = M.summarize(recs)
    assert s["completed"] == 40 and s["out_len_mismatch"] == 0 and s["tpot"]["p50"] > 0.001
    assert any(x.get("kv_cache_usage") == 0.5 for x in samples)


def test_rank_modes():
    req = dict(adapter=2, rank=32, prompt=[1, 2], output_len=4)
    mk = lambda m: C.OpenAICompletions("http://x", m, "matryoshka-{a}", "base").body(req)  # noqa: E731
    assert mk("base")["model"] == "base"
    assert mk("adapter")["model"] == "matryoshka-2"
    assert mk("tier-copies")["model"] == "matryoshka-2-r32"
    assert mk("xargs")["vllm_xargs"] == {"matryoshka_rank": 32}
    assert C.OpenAICompletions("http://x", "xargs", "m{a}", "b").body(dict(req, rank=None))["vllm_xargs"] == {
        "matryoshka_rank": "auto"}


# ----------------------------------------------------------------- rank policy and front end

def test_rank_policy_loading():
    from rankfuse.integration import rank_policy as RP

    assert RP.load_policy("", "") is None
    p = RP.load_policy("fixed:32", "")
    st = RP.RankState(now=0, adapter="a", r_max=16, tiers=(8, 16), prompt_tokens=None, max_tokens=None)
    assert p.choose_rank(st) == 16
    assert RP.clip_rank(0, 64) == 1 and RP.clip_rank(None, 64) is None


def test_frontend_resolve_rank(tmp_path):
    from rankfuse.integration import frontend as F
    from rankfuse.integration.rank_policy import FixedRankPolicy

    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 64}))
    lr = SimpleNamespace(lora_name="m", lora_path=str(tmp_path))
    req = lambda x: SimpleNamespace(vllm_xargs=x, prompt=[1, 2, 3], max_tokens=5)  # noqa: E731
    old = F.STATE.policy
    try:
        F.STATE.policy = None
        assert F.resolve_rank(req({"matryoshka_rank": 32}), lr) == (32, "request")
        assert F.resolve_rank(req({"matryoshka_rank": 512}), lr) == (64, "request")
        assert F.resolve_rank(req({"matryoshka_rank": "16"}), lr) == (16, "request")
        assert F.resolve_rank(req(None), lr) == (None, "full")
        F.STATE.policy = FixedRankPolicy(8)
        assert F.resolve_rank(req({"matryoshka_rank": "auto"}), lr) == (8, "policy")
        assert F.resolve_rank(req({}), lr) == (8, "policy")
        assert F.resolve_rank(req({"matryoshka_rank": 64}), lr) == (64, "request")
        assert F.resolve_rank(req({"matryoshka_rank": "full"}), lr) == (None, "full")
    finally:
        F.STATE.policy = old
    assert F.adapter_info(str(tmp_path)) == (64, (1, 2, 4, 8, 16, 32, 64))
