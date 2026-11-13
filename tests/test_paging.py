import math
import random
import torch

from rankfuse.paging import AdapterPageManager, PagePool
from rankfuse.paging.sim import Request, poisson_trace, simulate


def test_prefix_pages_load_incrementally():
    trace = [Request(0, 8), Request(0, 64), Request(0, 16)]
    result = simulate(trace, budget_bytes=4 * 81 * 1024**2, policy="prefix")
    assert result.misses == 2
    assert result.hits == 1
    assert result.bytes_copied == 4 * 81 * 1024**2


def test_unpressured_policies_have_no_evictions():
    trace = [Request(adapter, rank) for adapter in range(4) for rank in (8, 64)]
    results = [
        simulate(trace, budget_bytes=16 * 323 * 1024**2, policy=policy)
        for policy in ("vllm", "whole", "true_rank", "prefix")
    ]
    assert all(result.evictions == 0 for result in results)
    assert [result.hit_rate for result in results] == [0.5, 0.5, 0.0, 0.0]


def test_poisson_trace_is_seeded():
    kwargs = dict(requests=20, adapters=8, popularity="zipf", zipf_s=0.8, seed=7)
    assert poisson_trace(**kwargs) == poisson_trace(**kwargs)


def test_page_manager_prefix_and_incremental_loads():
    pool = PagePool(num_frames=8, h_in=2, h_out=3, dtype=torch.float32, device="cpu")
    manager = AdapterPageManager(pool, page_rank=16, max_adapters=4, max_pages_per_adapter=4)
    for adapter in "AB":
        manager.register(adapter, torch.randn(64, 2), torch.randn(64, 3))
    first = manager.ensure_prefix("A", 16)
    second = manager.ensure_prefix("A", 64)
    assert len(first) == 1
    assert len(second) == 4
    assert tuple(manager.page_table[0, :4].tolist()) == second
    assert manager.pool.free_pages == 4


def test_page_manager_protects_in_use_adapters():
    pool = PagePool(num_frames=2, h_in=2, h_out=2, dtype=torch.float32, device="cpu")
    manager = AdapterPageManager(pool, page_rank=16, max_adapters=3, max_pages_per_adapter=2)
    for adapter in range(3):
        manager.register(adapter, torch.randn(32, 2), torch.randn(32, 2))
    manager.ensure_resident([0, 1], 16)
    try:
        manager.ensure_prefix(2, 16)
    except RuntimeError as exc:
        assert "in use" in str(exc)
    else:
        raise AssertionError("in-use pages were evicted")


def test_seven_step_worked_example():
    """7-step worked example from spec:
    Pool of 8 frames: A@64,B@8 -> C@16,D@8 -> E@32,B@8 evicts only A's page 3 -> A@16 loads nothing.
    """
    pool = PagePool(num_frames=8, page_rows=16, h_in=2, h_out=2, dtype=torch.float32, device="cpu")
    manager = AdapterPageManager(pool, page_rank=16, num_adapter_slots=6, r_max=64)
    for name in ["A", "B", "C", "D", "E"]:
        manager.register(name, torch.randn(64, 2), torch.randn(64, 2))

    # Step 1: A@64 (4 pages: 0..3), B@8 (1 page: 0) -> 5 frames used, 3 free
    manager.begin_step(["A", "B"])
    res1 = manager.ensure_resident(["A", "B"], {"A": 64, "B": 8})
    manager.end_step()
    assert len(res1[0]) == 4
    assert len(res1[1]) == 1
    assert manager._resident["A"] == 4
    assert manager._resident["B"] == 1
    assert pool.free_pages == 3
    a_pages_step1 = list(manager.cpu_page_table[manager._ids["A"], :4])

    # Step 2: C@16 (1 page), D@8 (1 page) -> 7 frames used, 1 free
    manager.begin_step(["C", "D"])
    res2 = manager.ensure_resident(["C", "D"], {"C": 16, "D": 8})
    manager.end_step()
    assert len(res2[0]) == 1
    assert len(res2[1]) == 1
    assert manager._resident["C"] == 1
    assert manager._resident["D"] == 1
    assert pool.free_pages == 1

    # Step 3: E@32 (2 pages), B@8 (1 page, already resident)
    # Available: 1 free frame. Needs 2 frames for E, so exactly 1 page must be evicted.
    # Active: E, B. Idle: A (used step 1), C (step 2), D (step 2).
    # LRU idle is A. Evicts only A's page 3!
    manager.begin_step(["E", "B"])
    res3 = manager.ensure_resident(["E", "B"], {"E": 32, "B": 8})
    manager.end_step()
    assert manager._resident["E"] == 2
    assert manager._resident["B"] == 1
    # Only A's top page (page 3) was evicted; pages 0, 1, 2 remain resident!
    assert manager._resident["A"] == 3
    assert list(manager.cpu_page_table[manager._ids["A"], :3]) == a_pages_step1[:3]
    assert manager.cpu_page_table[manager._ids["A"], 3] == -1
    assert manager._resident["C"] == 1
    assert manager._resident["D"] == 1
    assert pool.free_pages == 0

    # Step 4: A@16 -> loads nothing!
    # A needs 16 (1 page: page 0). A still has pages 0, 1, 2 resident.
    manager.begin_step(["A"])
    free_before = pool.free_pages
    res4 = manager.ensure_resident(["A"], {"A": 16})
    manager.end_step()
    assert len(res4[0]) == 1
    assert res4[0][0] == a_pages_step1[0]
    assert pool.free_pages == free_before == 0
    assert manager._resident["A"] == 3


def test_eviction_removes_only_top_page():
    pool = PagePool(num_frames=4, page_rows=16, h_in=2, h_out=2, dtype=torch.float32, device="cpu")
    manager = AdapterPageManager(pool, page_rank=16, num_adapter_slots=2, r_max=64)
    manager.register("A", torch.randn(64, 2), torch.randn(64, 2))
    manager.register("B", torch.randn(64, 2), torch.randn(64, 2))

    manager.begin_step(["A"])
    manager.ensure_resident(["A"], 64)
    manager.end_step()
    assert manager._resident["A"] == 4
    assert pool.free_pages == 0
    a_frames = list(manager.cpu_page_table[manager._ids["A"], :4])

    manager.begin_step(["B"])
    manager.ensure_resident(["B"], 16)
    manager.end_step()
    # Eviction removes only the top page (page 3) of A
    assert manager._resident["A"] == 3
    assert list(manager.cpu_page_table[manager._ids["A"], :3]) == a_frames[:3]
    assert manager.cpu_page_table[manager._ids["A"], 3] == -1
    assert manager._resident["B"] == 1
    assert pool.free_pages == 0


def test_page_manager_randomized_invariants(monkeypatch):
    pool = PagePool(num_frames=12, h_in=2, h_out=2, dtype=torch.float32, device="cpu")
    manager = AdapterPageManager(pool, page_rank=16, max_adapters=8, max_pages_per_adapter=4)
    for adapter in range(8):
        manager.register(adapter, torch.randn(64, 2), torch.randn(64, 2))

    # Assert no GPU readback is used: monkeypatch Tensor.item to raise during ensure_prefix / eviction
    def forbidden_item(self):
        raise AssertionError("Tensor.item was called during residency management!")
    monkeypatch.setattr(torch.Tensor, "item", forbidden_item)

    rng = random.Random(11)
    for _ in range(2000):
        active = rng.sample(range(8), rng.randrange(1, 4))
        ranks = {adapter: rng.choice((1, 8, 16, 32, 64)) for adapter in active}
        total_pages_needed = sum(math.ceil(ranks[a] / manager.page_rank) for a in active)

        manager.begin_step(active)
        try:
            manager.ensure_resident(active, ranks)
        except RuntimeError as exc:
            # Only the specific "step needs more pages than the pool" or all resident in use error is tolerated
            assert "all resident pages are in use" in str(exc) or "pool is full" in str(exc)
            assert total_pages_needed > pool.num_pages or len(active) == pool.num_pages
            manager.end_step()
            continue
        mapped = set()
        for adapter in range(8):
            row = manager.page_table[manager._ids[adapter]].tolist()
            resident = manager._resident[adapter]
            assert row[:resident] == [slot for slot in row[:resident] if slot >= 0]
            assert all(slot == -1 for slot in row[resident:])
            mapped.update(slot for slot in row if slot >= 0)
        assert len(mapped) + pool.free_pages == pool.num_pages
        manager.end_step()


def test_simulator_equal_when_budget_is_unpressured():
    trace = poisson_trace(100, 8, seed=4)
    budget = 8 * 323 * 1024**2
    results = [simulate(trace, budget_bytes=budget, policy=policy)
               for policy in ("vllm", "whole", "true_rank", "prefix")]
    assert all(result.evictions == 0 for result in results)
    assert all(result.cold_start_fraction == results[0].cold_start_fraction for result in results)
