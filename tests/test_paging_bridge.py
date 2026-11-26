"""The multi-adapter kernels read an `AdapterPageManager`'s `PagePool` in place (rankfuse/paging/bridge.py).

    python -m pytest -q tests/test_paging_bridge.py
"""

import pytest
import torch

from rankfuse.multi import MULTI_REGISTRY, build_meta, tile_capacity
from rankfuse.paging.bridge import (
    BlockPageManager,
    FrameLayout,
    make_pool,
    reference_delta,
    weights_from_manager,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

if torch.cuda.is_available():
    from rankfuse.kernels import rankfuse_multi  # noqa: F401  (registers rf1_ma, rf2_ma_*)
    from rankfuse.kernels.rankfuse_multi import multi_lora

IMPLS = ["rf1_ma", "rf2_ma_splitk", "rf2_ma_segment"]
R_MAX = 64


def _manager(n_adapters=8, frames=12, h_in=896, slices=(896,), ranks=None, seed=0):
    layout = FrameLayout(h_in, tuple(slices))
    pool = make_pool(layout, frames, dtype=torch.float16)
    mgr = BlockPageManager(pool, layout, num_adapter_slots=n_adapters, r_max=R_MAX)
    gen = torch.Generator().manual_seed(seed)
    for a in range(n_adapters):
        r = ranks[a] if ranks else R_MAX
        A = (torch.randn(len(slices), r, h_in, generator=gen) / h_in**0.5).half()
        Bs = [(torch.randn(r, n, generator=gen) / r**0.5).half() for n in slices]
        mgr.register(a, A, Bs)
    return mgr


def _batch(adapters, ranks, h_in, tokens_per=1, seed=0):
    gen = torch.Generator().manual_seed(seed)
    ta = torch.tensor(adapters, dtype=torch.int32).repeat_interleave(tokens_per)
    tr = torch.tensor(ranks, dtype=torch.int32).repeat_interleave(tokens_per)
    x = torch.randn(ta.numel(), h_in, generator=gen).half()
    return x.cuda(), ta.cuda(), tr.cuda()


def _expected(mgr, x, ta, tr):
    out = torch.zeros(x.shape[0], sum(mgr.layout.slice_sizes), device=x.device)
    for a in ta.unique().tolist():
        idx = torch.nonzero(ta == a).flatten()
        A, Bs = mgr.host_weights(a)
        out[idx] = reference_delta(x[idx], A, Bs, tr[idx].cpu())
    return out


def _run(impl, x, W, ta, tr):
    m = MULTI_REGISTRY[impl]
    return m(x, W, ta, tr, **m.prepare(x, W, ta, tr))


def _close(y, e):
    torch.testing.assert_close(y.float(), e, rtol=2e-2, atol=2e-2)


def _step(mgr, ranks: dict):
    """One serving step's residency: pin the step's adapters, load their prefixes, fence."""
    mgr.begin_step(list(ranks))
    mgr.ensure_resident(list(ranks), ranks)
    mgr.wait_for_copies()


def test_layout_offsets_cover_frame():
    L = FrameLayout(896, (896, 128, 128))
    spans = [(o, 16 * 896) for o in L.a_off] + [(o, 16 * n) for o, n in zip(L.b_off, L.slice_sizes)]
    spans.sort()
    pos = 0
    for o, n in spans:
        assert o == pos
        pos += n
    assert pos == L.elems == 16 * L.row_width


def test_weights_read_pool_in_place():
    mgr = _manager(n_adapters=2, frames=4)
    W = weights_from_manager(mgr)
    assert W.A_pool.data_ptr() == W.B_pool.data_ptr() == mgr.pool.frames.data_ptr()
    assert W.page_table is mgr.page_table
    assert W.frame_stride_a == W.frame_stride_b == 16 * (896 + 896)
    assert W.a_off == (0,) and W.b_off == (16 * 896,) and W.b_ld == (896,)
    assert W.aligned()


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("tokens_per", [1, 37])
def test_partially_resident(impl, tokens_per):
    """12 frames for 8 adapters x 4 pages: each step's adapters hold only the pages their ranks need."""
    mgr = _manager(n_adapters=8, frames=12)
    step = {0: 64, 1: 8, 2: 33, 3: 16, 5: 17}  # 4 + 1 + 3 + 1 + 2 = 11 pages
    adapters = [0, 1, 2, 3, 5, 0, 2, 2, 5, 1, 3, 0]
    ranks = [64, 8, 33, 16, 17, 1, 20, 33, 5, 3, 16, 48]
    _step(mgr, step)
    W = weights_from_manager(mgr)
    pt = mgr.page_table.cpu()
    assert (pt == -1).any() and (pt[4] == -1).all()
    assert (pt[1, 1:] == -1).all() and (pt[2, 3] == -1)
    x, ta, tr = _batch(adapters, ranks, 896, tokens_per)
    y = _run(impl, x, W, ta, tr)
    mgr.end_step()
    _close(y, _expected(mgr, x, ta, tr))


@pytest.mark.parametrize("impl", IMPLS)
def test_evict_then_reload(impl):
    mgr = _manager(n_adapters=4, frames=6)
    W = weights_from_manager(mgr)
    # step 1: adapter 0 at 64 (4 pages), 1 at 16 (1 page): 5 of 6 frames
    _step(mgr, {0: 64, 1: 16})
    x, ta, tr = _batch([0, 1, 0], [64, 16, 40], 896)
    _close(_run(impl, x, W, ta, tr), _expected(mgr, x, ta, tr))
    mgr.end_step()
    frames0 = mgr.page_table[0].tolist()
    # step 2: adapter 2 at 48 needs 3 frames: 1 free + adapter 0's top two pages (LRU top-page eviction)
    _step(mgr, {2: 48})
    assert mgr._resident[0] == 2 and mgr._resident[1] == 1
    assert mgr.page_table[0].tolist() == frames0[:2] + [-1, -1]
    x, ta, tr = _batch([2, 2, 0, 1], [48, 7, 32, 16], 896)  # 0 still runs at <= 32 from its prefix
    _close(_run(impl, x, W, ta, tr), _expected(mgr, x, ta, tr))
    mgr.end_step()
    # step 3: adapter 0 back at 64: pages 2, 3 reload into frames freed from adapters 1 and 2
    _step(mgr, {0: 64})
    assert mgr._resident[0] == 4
    assert mgr.page_table[0, :2].tolist() == frames0[:2]
    x, ta, tr = _batch([0, 0, 2], [64, 50, 16], 896)
    _close(_run(impl, x, W, ta, tr), _expected(mgr, x, ta, tr))
    mgr.end_step()


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("h_in,slices", [(896, (896, 128, 128)), (896, (4864, 4864))])
def test_merged_slices(impl, h_in, slices):
    mgr = _manager(n_adapters=5, frames=10, h_in=h_in, slices=slices, ranks=[64, 32, 48, 20, 64])
    _step(mgr, {0: 64, 1: 32, 3: 20, 4: 16})  # 4 + 2 + 2 + 1 = 9 pages
    W = weights_from_manager(mgr)
    assert W.n_slices == len(slices) and W.aligned()
    x, ta, tr = _batch([0, 1, 3, 4, 0, 1], [64, 32, 20, 16, 9, 17], h_in, 5)
    _close(_run(impl, x, W, ta, tr), _expected(mgr, x, ta, tr))
    mgr.end_step()


def test_cuda_graph_replay_across_residency_changes():
    """One graph captured once; between replays pages are evicted, loaded into other frames and the
    page table rewritten in place. The replay reads the new contents."""
    N, T = 8, 24
    mgr = _manager(n_adapters=N, frames=10)
    W = weights_from_manager(mgr)
    gen = torch.Generator().manual_seed(5)

    def step_batch(active: dict):
        ad = torch.tensor(list(active))[torch.randint(0, len(active), (T,), generator=gen)]
        rk = torch.tensor([int(torch.randint(1, active[int(a)] + 1, (1,), generator=gen)) for a in ad])
        rk[torch.randint(0, T, (1,), generator=gen)] = 0  # a no-LoRA token
        x = torch.randn(T, 896, generator=gen).half()
        return x.cuda(), ad.int().cuda(), rk.int().cuda()

    bm = 16
    cap = tile_capacity(T, bm, N)
    steps = [{0: 64, 1: 32, 2: 16}, {3: 64, 4: 32, 0: 16}, {5: 32, 6: 64, 7: 32}, {0: 64, 2: 64, 4: 16}]
    _step(mgr, steps[0])
    x0, ta0, tr0 = step_batch(steps[0])
    meta = build_meta(ta0.cpu(), tr0.cpu(), num_adapters=N, bm=bm, r_pad=R_MAX, capacity=cap)
    xs, trs = x0.clone(), tr0.clone()
    out = torch.empty(T, 896, dtype=torch.float16, device="cuda")

    def fn():
        multi_lora(xs, W, trs, meta, out=out)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    _close(out, _expected(mgr, xs, ta0, trs))
    mgr.end_step()

    tables = {tuple(mgr.page_table.flatten().tolist())}
    for active in steps[1:]:
        _step(mgr, active)
        x, ta, tr = step_batch(active)
        meta.load_(build_meta(ta.cpu(), tr.cpu(), num_adapters=N, bm=bm, r_pad=R_MAX, capacity=cap))
        xs.copy_(x)
        trs.copy_(tr)
        g.replay()
        _close(out, _expected(mgr, x, ta, tr))
        mgr.end_step()
        tables.add(tuple(mgr.page_table.flatten().tolist()))
    assert len(tables) == len(steps), "every step should change the page table"
    assert mgr._resident[0] == 4 and mgr.pool.free_pages == 0


def test_eviction_waits_for_running_step():
    """A frame read by a still-running kernel is not overwritten by the next step's load."""
    mgr = _manager(n_adapters=2, frames=4)
    _step(mgr, {0: 64})
    torch.cuda.synchronize()
    frames = mgr.pool.frames
    top = int(mgr.page_table[0, 3])
    expect = frames[top].clone()
    torch.cuda._sleep(200_000_000)  # the "kernel" of this step is still running ...
    seen = frames[top].clone()  # ... and reads adapter 0's top page after the sleep
    mgr.end_step()
    _step(mgr, {1: 16})  # evicts adapter 0's top page and loads adapter 1 into that frame
    torch.cuda.synchronize()
    assert int(mgr.page_table[1, 0]) == top
    assert torch.equal(seen, expect), "the copy stream overwrote a frame the compute stream was reading"
    assert not torch.equal(frames[top], expect)
