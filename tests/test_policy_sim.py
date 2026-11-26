"""Residency-policy simulator with sizes and copy times (rankfuse/paging/policy_sim.py). CPU only.

    python -m pytest -q tests/test_policy_sim.py
"""

import math

import pytest

from rankfuse.paging.policy_sim import (
    POLICIES,
    CostModel,
    SizeModel,
    make_trace,
    simulate_policy,
    tiers,
)
from rankfuse.paging.sim import Request

SZ = SizeModel("toy", row_bytes=1000, num_layers=2, r_max=64)  # page = 16 kB, adapter = 64 kB
FREE = CostModel(bw_gbps=1e9, call_us=0.0)  # copies take ~no time


def run(trace, policy, budget, cost=FREE, size=SZ, warmup=0):
    return simulate_policy(trace, policy, budget, size, cost, warmup)


def test_real_sizes():
    assert SizeModel.qwen25("0.5B", 64).row_bytes == 1_099_776
    assert SizeModel.qwen25("7B", 64).page_bytes == 16 * 5_046_272
    assert tiers(64) == [1, 2, 4, 8, 16, 32, 64] and tiers(256)[-1] == 256


def test_one_adapter_many_tiers():
    """Tiers of one adapter: vLLM/S-LoRA keep a copy per tier, RankFuse one copy for all."""
    trace = [Request(0, r, i) for i, r in enumerate([8, 64, 16, 8, 64, 16])]
    big = 10**9
    vllm, slora, whole, prefix = (run(trace, p, big) for p in POLICIES)
    assert (vllm.misses, slora.misses) == (3, 3)
    assert vllm.bytes_copied == slora.bytes_copied == (8 + 64 + 16) * 1000
    assert whole.misses == 1 and whole.bytes_copied == 64_000
    # prefix: r8 -> page 0, r64 -> pages 1..3, then hits
    assert prefix.misses == 2 and prefix.partial_hits == 1 and prefix.bytes_copied == 4 * 16_000
    assert vllm.resident_copies > 1 and whole.resident_copies == 1


def test_budget_slots_and_eviction():
    # budget of 2 r_max slots: vLLM holds 2 (adapter, tier) copies, rf_whole 2 adapters, prefix 8 pages
    trace = [Request(a, 16, i) for i, a in enumerate([0, 1, 2, 0, 1, 2])]
    b = 2 * 64_000
    assert run(trace, "vllm", b).hits == 0  # LRU with 2 slots on a cycle of 3
    assert run(trace, "rf_whole", b).hits == 0
    p = run(trace, "rf_prefix", b)
    assert p.hits == 3 and p.evictions == 0  # 3 one-page prefixes fit in 8 pages
    s = run(trace, "slora", b)
    assert s.hits == 3  # true rank 16 rows each


def test_top_page_eviction():
    b = 5 * 16_000  # 5 pages
    trace = [Request(0, 64, 0), Request(1, 16, 1), Request(2, 16, 2), Request(0, 32, 3)]
    r = run(trace, "rf_prefix", b)
    # A@64 (4) + B@16 (1) = 5; C@16 evicts A's page 3; A@32 still has pages 0..2 -> hit
    assert r.evictions == 1 and r.hits == 1 and r.misses == 3


def test_infeasible_budget():
    trace = [Request(0, 64, 0)]
    assert not run(trace, "rf_whole", 63_999).feasible
    assert not run(trace, "vllm", 63_999).feasible
    assert run(trace, "rf_prefix", 64_000).feasible
    small = [Request(0, 16, 0)]
    assert run(small, "rf_prefix", 16_000).feasible and not run(small, "rf_whole", 16_000).feasible


def test_waits_queue_on_one_copy_engine():
    cost = CostModel(bw_gbps=1e-3, call_us=0.0)  # 1 MB/s: a 16 kB page takes 16 ms
    trace = [Request(0, 16, 0.0), Request(1, 16, 0.001), Request(0, 16, 0.002)]
    r = run(trace, "rf_prefix", 10**9, cost=cost)
    # loads: A 0..16 ms; B queued 16..32 ms; A's hit at 2 ms waits for its in-flight page
    assert math.isclose(r.wait_s, 0.016 + (0.032 - 0.001) + (0.016 - 0.002), rel_tol=1e-9)
    assert r.hits == 1


def test_call_overhead_counts():
    trace = [Request(0, 64, 0)]
    cost = CostModel(bw_gbps=1e9, call_us=10.0)
    v = run(trace, "vllm", 10**9, cost=cost)
    assert v.copy_calls == 2 * 7 * SZ.num_layers and math.isclose(v.copy_busy_s, v.copy_calls * 1e-5, rel_tol=1e-6)
    assert run(trace, "rf_prefix", 10**9, cost=cost).copy_calls == 4
    assert run(trace, "slora", 10**9, cost=cost).copy_calls == 2 * SZ.num_layers


@pytest.mark.parametrize("policy", POLICIES)
def test_unpressured_all_hits_after_warmup(policy):
    trace = make_trace(4000, 20, r_max=64, zipf_alpha=1.0, rank_mix="uniform", rate=100.0, seed=1)
    r = run(trace, policy, 10**9, warmup=2000)
    assert r.feasible and r.evictions == 0 and r.hit_rate > 0.99


def test_prefix_never_copies_more_than_whole():
    trace = make_trace(5000, 200, r_max=64, zipf_alpha=1.0, rank_mix="skew_low", rate=50.0, seed=2)
    for b in (10 * 64_000, 50 * 64_000):
        p, w = run(trace, "rf_prefix", b), run(trace, "rf_whole", b)
        assert p.hit_rate >= w.hit_rate and p.bytes_copied <= w.bytes_copied
