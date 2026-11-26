"""Whole-model prefix pages (rankfuse/paging/model_pages.py): layout, manager, kernels on the pool.

    python -m pytest -q tests/test_model_pages.py
"""

import math
import random

import pytest
import torch

from rankfuse.multi import build_meta, tile_capacity
from rankfuse.paging.bridge import reference_delta
from rankfuse.paging.model_pages import ModelPageLayout, ModelPageManager, ModelSpec

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
if torch.cuda.is_available():
    from rankfuse.kernels.rankfuse_multi import multi_lora

R_MAX = 64
TINY = dict(num_layers=3, hidden=256, intermediate=512, n_heads=4, n_kv=2, head_dim=64)


# ------------------------------------------------------------------------------------------------
# Layout (CPU)
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("size,row_bytes", [("0.5B", 1_099_776), ("7B", 5_046_272)])
@pytest.mark.parametrize("merged", [True, False])
def test_qwen_page_sizes(size, row_bytes, merged):
    L = ModelPageLayout(ModelSpec.qwen25(size, merged=merged))
    assert L.row_bytes() == row_bytes
    assert L.page_bytes() == 16 * row_bytes
    assert L.adapter_bytes(64) == 4 * L.page_bytes() and L.adapter_bytes(256) == 16 * L.page_bytes()
    assert len(L.blocks) == L.spec.num_layers * (4 if merged else 7)


@pytest.mark.parametrize("merged", [True, False])
def test_blocks_tile_the_frame(merged):
    L = ModelPageLayout(ModelSpec.qwen25("0.5B", merged=merged))
    spans = []
    for fl in L.blocks.values():
        spans += [(o, 16 * fl.h_in) for o in fl.a_off] + [(o, 16 * n) for o, n in zip(fl.b_off, fl.slice_sizes)]
    spans.sort()
    pos = 0
    for o, n in spans:
        assert o == pos and o % 16 == 0
        pos += n
    assert pos == L.frame_elems


def test_pack_round_trip():
    L = ModelPageLayout(ModelSpec.transformer("tiny", **TINY))
    w = L.random_weights(40, seed=1)
    host = L.pack(w, 40, pin=False)
    assert host.shape == (3, L.frame_elems)
    for (layer, g), fl in L.blocks.items():
        A, Bs = w[(layer, g)]
        for s in range(fl.n_slices):
            for k in range(3):
                n = min(16, 40 - 16 * k)
                a = host[k, fl.a_off[s]:fl.a_off[s] + 16 * fl.h_in].view(16, fl.h_in)
                assert torch.equal(a[:n], A[s, 16 * k:16 * k + n]) and not a[n:].any()
                ns = fl.slice_sizes[s]
                b = host[k, fl.b_off[s]:fl.b_off[s] + 16 * ns].view(16, ns)
                assert torch.equal(b[:n], Bs[s][16 * k:16 * k + n]) and not b[n:].any()


def test_manager_randomized_invariants_cpu():
    L = ModelPageLayout(ModelSpec.transformer("tiny", **TINY))
    mgr = ModelPageManager(L, 12, num_adapter_slots=10, r_max=R_MAX, device="cpu")
    for a in range(10):
        r = random.Random(a).choice((16, 33, 64))
        mgr.register(a, torch.zeros(math.ceil(r / 16), L.frame_elems, dtype=torch.float16), rank=r)
    rng = random.Random(3)
    for _ in range(3000):
        active = rng.sample(range(10), rng.randrange(1, 4))
        ranks = {a: rng.randint(0, mgr._ranks[a]) for a in active}
        mgr.begin_step(active)
        need_new = sum(max(0, math.ceil(r / 16) - mgr._resident[a]) for a, r in ranks.items())
        evictable = sum(mgr._resident[a] for a in mgr._ids if a not in active)
        free = mgr.free_frames
        try:
            got = mgr.ensure(ranks)
        except RuntimeError:
            assert need_new > free + evictable, "raised although the step fits"
            mgr.end_step()
            mgr.check_invariants()
            continue
        assert got == ranks
        for a, r in ranks.items():
            assert mgr.resident_rank(a) >= r
        mgr.check_invariants()
        mgr.end_step()
    assert mgr.stats.copy_calls == mgr.stats.page_loads


def test_partial_mode_serves_resident_prefix():
    L = ModelPageLayout(ModelSpec.transformer("tiny", **TINY))
    mgr = ModelPageManager(L, 5, num_adapter_slots=3, r_max=R_MAX, device="cpu")
    for a in range(3):
        mgr.register(a, torch.zeros(4, L.frame_elems, dtype=torch.float16))
    mgr.begin_step([0, 1])
    assert mgr.ensure({0: 64, 1: 16}) == {0: 64, 1: 16}
    with pytest.raises(RuntimeError):
        mgr.ensure({2: 32})  # 0 and 1 are in this step: nothing evictable, 0 free
    mgr.end_step()
    mgr.check_invariants()
    # next step: 2 wants 64 but 1 is also in the step: it gets 0's pages (LRU) -> 4 frames
    mgr.begin_step([1, 2])
    assert mgr.ensure({1: 16, 2: 64}) == {1: 16, 2: 64}
    assert mgr.resident_rank(0) == 0
    mgr.end_step()
    mgr.begin_step([0, 2])
    got = mgr.ensure({0: 48, 2: 64}, allow_partial=True)  # only 1's page is evictable
    assert got == {0: 16, 2: 64} and mgr.serveable_rank(0, 48) == 16
    assert mgr.stats.partial == 1
    mgr.end_step()
    mgr.check_invariants()


# ------------------------------------------------------------------------------------------------
# Kernels on the whole-model pool (CUDA)
# ------------------------------------------------------------------------------------------------


def _model(merged=True, frames=10, ranks=(64, 48, 32, 17, 64, 5), spec=None):
    L = ModelPageLayout(spec or ModelSpec.transformer("tiny", merged=merged, **TINY))
    mgr = ModelPageManager(L, frames, num_adapter_slots=len(ranks), r_max=R_MAX)
    weights = {}
    for a, r in enumerate(ranks):
        weights[a] = L.random_weights(r, seed=a)
        mgr.register_weights(a, weights[a], r)
    return L, mgr, weights


def _batch(adapters, ranks, h_in, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(len(adapters), h_in, generator=gen).half().cuda()
    return x, torch.tensor(adapters, dtype=torch.int32).cuda(), torch.tensor(ranks, dtype=torch.int32).cuda()


def _check_all_blocks(L, mgr, weights, adapters, ranks, seed=0):
    """Run RF1-MA on every (layer, group) for one batch and compare with the fp32 reference."""
    slots = [mgr.slot(a) for a in adapters]
    for (layer, g), fl in L.blocks.items():
        x, ta, tr = _batch(slots, ranks, fl.h_in, seed + layer)
        W = mgr.weights(layer, g)
        meta = build_meta(ta.cpu(), tr.cpu(), num_adapters=mgr.num_adapter_slots)
        y = multi_lora(x, W, tr, meta)
        e = torch.zeros_like(y, dtype=torch.float32)
        for a in set(adapters):
            idx = [i for i, v in enumerate(adapters) if v == a]
            A, Bs = weights[a][(layer, g)]
            e[idx] = reference_delta(x[idx], A, Bs, [ranks[i] for i in idx])
        torch.testing.assert_close(y.float(), e, rtol=2e-2, atol=2e-2, msg=f"layer {layer} {g}")


def _step(mgr, ranks):
    mgr.begin_step(list(ranks))
    got = mgr.ensure(ranks)
    mgr.wait_for_copies()
    return got


@cuda
@pytest.mark.parametrize("merged", [True, False])
def test_every_layer_and_projection(merged):
    L, mgr, w = _model(merged)
    _step(mgr, {0: 64, 1: 48, 3: 17, 5: 5})  # 4 + 3 + 2 + 1 = 10 frames
    assert mgr.free_frames == 0 and mgr.stats.copy_calls == 10
    _check_all_blocks(L, mgr, w, [0, 1, 3, 5, 0, 1, 3, 0], [64, 48, 17, 5, 1, 30, 16, 33])
    mgr.end_step()
    mgr.check_invariants()


@cuda
def test_evict_then_reload_whole_model():
    L, mgr, w = _model(frames=8)
    _step(mgr, {0: 64, 1: 48})  # 7 frames
    mgr.end_step()
    f0 = mgr.page_table[mgr.slot(0)].tolist()
    _step(mgr, {2: 32})  # 1 free + adapter 0's top page
    assert mgr.resident_pages(0) == 3 and mgr.page_table[mgr.slot(0)].tolist() == f0[:3] + [-1]
    _check_all_blocks(L, mgr, w, [2, 0, 1, 2], [32, 48, 48, 1])
    mgr.end_step()
    _step(mgr, {0: 64, 4: 64})  # 0 needs 1 more, 4 needs 4: evicts 1 (LRU) top pages, then 2
    assert mgr.resident_pages(0) == 4 and mgr.page_table[mgr.slot(0), :3].tolist() == f0[:3]
    _check_all_blocks(L, mgr, w, [0, 4, 4, 0], [64, 64, 3, 20], seed=9)
    mgr.end_step()
    mgr.check_invariants()


@cuda
def test_real_qwen05b_pages():
    """Real Qwen2.5-0.5B page size (17.6 MB): first and last layer, merged groups."""
    spec = ModelSpec.qwen25("0.5B")
    L, mgr, w = _model(frames=5, ranks=(32, 64, 16), spec=spec)
    _step(mgr, {0: 32, 2: 16, 1: 32})
    assert mgr.stats.bytes_copied == 5 * L.page_bytes()
    slots = [mgr.slot(a) for a in (0, 2, 1, 0)]
    ranks = [32, 16, 32, 7]
    for layer in (0, 23):
        for g in ("qkv", "down"):
            fl = L.group(layer, g)
            x, ta, tr = _batch(slots, ranks, fl.h_in, layer)
            y = multi_lora(x, mgr.weights(layer, g), tr, build_meta(ta.cpu(), tr.cpu(), num_adapters=3))
            e = torch.cat([reference_delta(x[i:i + 1], *w[a][(layer, g)], [ranks[i]])
                           for i, a in enumerate((0, 2, 1, 0))])
            torch.testing.assert_close(y.float(), e, rtol=2e-2, atol=2e-2)
    mgr.end_step()


@cuda
def test_cuda_graph_whole_model():
    """One graph over every (layer, group) of the model, replayed while pages move between frames."""
    L, mgr, w = _model(frames=8, ranks=(64, 64, 32, 48, 64, 16))
    N, T, bm = mgr.num_adapter_slots, 12, 16
    cap = tile_capacity(T, bm, N)
    gen = torch.Generator().manual_seed(0)
    keys = list(L.blocks)

    def batch(active):
        ad = [list(active)[int(i)] for i in torch.randint(0, len(active), (T,), generator=gen)]
        rk = [int(torch.randint(1, active[a] + 1, (1,), generator=gen)) for a in ad]
        return ad, rk

    steps = [{0: 64, 2: 32, 5: 16}, {1: 64, 3: 48}, {0: 64, 4: 48}, {2: 32, 5: 16, 3: 32, 1: 16}]
    _step(mgr, steps[0])
    ad, rk = batch(steps[0])
    meta = build_meta(ad, rk, num_adapters=N, bm=bm, r_pad=R_MAX, capacity=cap)
    trs = torch.tensor(rk, dtype=torch.int32, device="cuda")
    xs = {k: torch.randn(T, L.blocks[k].h_in, generator=gen).half().cuda() for k in keys}
    outs = {k: torch.empty(T, sum(L.blocks[k].slice_sizes), dtype=torch.float16, device="cuda") for k in keys}

    def fn():
        for k in keys:
            multi_lora(xs[k], mgr.weights(*k), trs, meta, out=outs[k])

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()

    def verify(ad, rk):
        for k in keys:
            e = torch.cat([reference_delta(xs[k][i:i + 1], *w[a][k], [rk[i]]) for i, a in enumerate(ad)])
            torch.testing.assert_close(outs[k].float(), e, rtol=2e-2, atol=2e-2, msg=str(k))

    g.replay()
    verify(ad, rk)
    mgr.end_step()
    for active in steps[1:]:
        _step(mgr, active)
        ad, rk = batch(active)
        meta.load_(build_meta([mgr.slot(a) for a in ad], rk, num_adapters=N, bm=bm, r_pad=R_MAX, capacity=cap))
        trs.copy_(torch.tensor(rk, dtype=torch.int32))
        for k in keys:
            xs[k].normal_()
        g.replay()
        verify(ad, rk)
        mgr.end_step()
    mgr.check_invariants()
    assert mgr.stats.evictions > 0


@cuda
def test_eviction_waits_for_last_use():
    L, mgr, w = _model(frames=4, ranks=(64, 16))
    _step(mgr, {0: 64})
    torch.cuda.synchronize()
    frames = mgr.pool.frames
    top = int(mgr.page_table[0, 3])
    expect = frames[top].clone()
    torch.cuda._sleep(200_000_000)
    seen = frames[top].clone()
    mgr.end_step()
    _step(mgr, {1: 16})
    torch.cuda.synchronize()
    assert int(mgr.page_table[1, 0]) == top
    assert torch.equal(seen, expect)
    assert not torch.equal(frames[top], expect)


@cuda
def test_load_into_free_frame_does_not_wait():
    """A load into a free frame (no eviction) is not ordered after the running step."""
    L, mgr, w = _model(frames=8, ranks=(64, 16))
    _step(mgr, {0: 64})
    mgr.end_step()
    torch.cuda.synchronize()
    torch.cuda._sleep(400_000_000)  # a long step on the compute stream
    mgr.end_step()
    _step(mgr, {1: 16})
    mgr.copy_done.synchronize()
    done_first = not torch.cuda.current_stream().query()
    torch.cuda.synchronize()
    assert done_first, "the copy waited for the compute stream although no frame was reused"
