"""Stage 2: per-step metadata, and CUDA-graph capture and replay of the multi-adapter implementations.

    python -m pytest -q tests/test_multi_graphs.py
"""

import pytest
import torch

from rankfuse.debug import TOL
from rankfuse.multi import (
    MULTI_REGISTRY,
    MultiWorkload,
    build_meta,
    load_all,
    make_batch,
    make_weights,
    tile_capacity,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

if torch.cuda.is_available():
    load_all(include_slora=False)
ALL = ["rf1_ma", "rf2_ma_splitk", "rf2_ma_segment", "rf2_ma_bucketed", "rf_ma_auto", "b1_multi", "b3_multi",
       "b4_multi_copies", "b4_multi_shared"]


@pytest.fixture(scope="module", autouse=True)
def b4_registered():
    from rankfuse.baselines import slora_bgmv_multi

    slora_bgmv_multi.enable()
    yield
    slora_bgmv_multi.disable()


def _expected(x, W, ta, tr, out=None, add=False):
    return MULTI_REGISTRY["b0_multi"](x, W, ta, tr, out=out, add=add)


def _close(y, e, dtype="fp16"):
    rtol, atol = TOL[dtype]
    torch.testing.assert_close(y.float(), e, rtol=rtol, atol=atol)


def _capture(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


# ------------------------------------------------------------------------------------------------
# Metadata
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["group", "rank_sorted", "segment"])
@pytest.mark.parametrize("T,num_adapters", [(1, 1), (37, 5), (300, 64), (2048, 256)])
def test_meta_tiles_cover_groups(mode, T, num_adapters):
    gen = torch.Generator().manual_seed(T)
    ta = torch.randint(-1, num_adapters, (T,), generator=gen)
    tr = torch.randint(0, 65, (T,), generator=gen)
    m = build_meta(ta, tr, num_adapters=num_adapters, mode=mode, device="cpu")
    order = m.order.long()
    assert sorted(order.tolist()) == list(range(T))
    none = (tr == 0) | (ta < 0)
    eff_ad = torch.where(none, -1, ta)
    seen = torch.zeros(T, dtype=torch.long)
    for a, start, length, limit, first in m.tiles[: m.num_tiles].tolist():
        assert 1 <= length <= m.bm
        toks = order[start:start + length]
        consecutive = bool((toks == toks[0] + torch.arange(length)).all())
        assert first == (int(toks[0]) if consecutive else -1)
        seen[toks] += 1
        assert (eff_ad[toks] == a).all(), "a tile holds tokens of one adapter"
        r = torch.where(none[toks], 0, tr[toks])
        assert int(r.max()) <= limit <= m.r_pad
        if mode != "group":
            assert int(r.max()) == limit, "per-tile limit is the tile's own maximum rank"
    assert (seen == 1).all(), "every token in exactly one tile"
    # groups
    G = m.num_groups
    assert int(m.group_len[:G].sum()) == T
    for g in range(G):
        s, n, a, lim = (int(t[g]) for t in (m.group_start, m.group_len, m.group_adapter, m.group_rank_limit))
        toks = order[s:s + n]
        assert (eff_ad[toks] == a).all()
        assert lim == int(torch.where(none[toks], 0, tr[toks]).max())
    if mode != "segment":
        real = m.group_adapter[:G][m.group_adapter[:G] >= 0]
        assert real.unique().numel() == real.numel(), "one group per adapter"


def test_meta_capacity_and_reload():
    T, N = 64, 16
    cap = tile_capacity(T, 16, N)
    gen = torch.Generator().manual_seed(1)
    m1 = build_meta(torch.randint(0, N, (T,), generator=gen), torch.randint(1, 65, (T,), generator=gen),
                    num_adapters=N, device="cpu", bm=16, r_pad=64, capacity=cap)
    assert m1.capacity == cap and (m1.tiles[m1.num_tiles:] == 0).all()
    m2 = build_meta(torch.zeros(T, dtype=torch.long), torch.full((T,), 3), num_adapters=N, device="cpu", bm=16,
                    r_pad=64, capacity=cap)
    m1.load_(m2)
    assert torch.equal(m1.tiles, m2.tiles) and torch.equal(m1.order, m2.order)
    with pytest.raises(ValueError):
        m1.load_(build_meta(torch.zeros(T, dtype=torch.long), torch.full((T,), 3), num_adapters=N, device="cpu"))


# ------------------------------------------------------------------------------------------------
# CUDA graphs
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("T", [17, 300])
@pytest.mark.parametrize("add", [False, True], ids=["write", "add"])
def test_capture_and_replay(impl, T, add):
    """The call with its prepared metadata is captured in a CUDA graph; replays give the result
    for new x values written into the captured input buffer."""
    w = MultiWorkload(T, 1, 9, h_in=896, h_out=896, zero_frac=0.2, seed=T)
    W = make_weights(w, "cuda")
    x, ta, tr = make_batch(w)
    m = MULTI_REGISTRY[impl]
    kw = m.prepare(x, W, ta, tr) if m.prepare else {}
    out = torch.randn(T, 896, device="cuda").half()
    base = out.clone()

    def body():
        if add:
            out.copy_(base)
        m(x, W, ta, tr, out=out, add=add, **kw)

    g = _capture(body)
    for seed in range(3):
        x.copy_(torch.randn(x.shape, generator=torch.Generator().manual_seed(seed)).to("cuda", x.dtype))
        g.replay()
        torch.cuda.synchronize()
        _close(out, _expected(x, W, ta, tr, out=base if add else None, add=add))


@pytest.mark.parametrize("impl,mode", [("rf1_ma", "group"), ("rf2_ma_splitk", "rank_sorted"),
                                       ("rf2_ma_segment", "segment")])
@pytest.mark.parametrize("T", [32, 256])
def test_one_graph_many_steps(impl, mode, T):
    """Fixed-capacity metadata: one graph captured once replays later steps with other adapters and
    ranks after meta.load_(new step), as a serving engine would reuse a decode graph."""
    N = 24
    W = make_weights(MultiWorkload(1, 1, N, h_in=896, h_out=896, seed=1), "cuda",
                     scale_per_rank={r: 16.0 / r for r in range(1, 65)})
    x = torch.randn(T, 896, device="cuda").half()
    ta = torch.zeros(T, dtype=torch.int32, device="cuda")
    tr = torch.zeros(T, dtype=torch.int32, device="cuda")
    bm = 16
    cap = tile_capacity(T, bm, N, mode)

    def step_meta(seed):
        gen = torch.Generator().manual_seed(seed)
        a = torch.randint(-1, N, (T,), generator=gen)
        r = torch.randint(0, 65, (T,), generator=gen)
        return a, r, build_meta(a, r, num_adapters=N, mode=mode, device="cuda", bm=bm, r_pad=64, capacity=cap)

    a0, r0, meta = step_meta(0)
    ta.copy_(a0)
    tr.copy_(r0)
    out = torch.empty(T, 896, device="cuda").half()
    g = _capture(lambda: MULTI_REGISTRY[impl](x, W, ta, tr, meta=meta, out=out))
    for seed in range(1, 5):
        a, r, new = step_meta(seed)
        ta.copy_(a)
        tr.copy_(r)
        meta.load_(new)
        x.normal_()
        g.replay()
        torch.cuda.synchronize()
        _close(out, _expected(x, W, ta, tr))


def test_auto_picks_both():
    """rf_ma_auto runs rf2_ma_bucketed for a mostly-low-rank batch on a long reduction, rf1_ma otherwise."""
    from rankfuse.kernels.rankfuse_multi import BucketPlan, use_bucketed

    low = BucketPlan([0, 1, 2, 3] * 8, [4] * 31 + [64], 4, "cpu")
    high = BucketPlan([0, 1, 2, 3] * 8, [64] * 32, 4, "cpu")
    assert use_bucketed(low, 4096) and not use_bucketed(low, 896) and not use_bucketed(high, 4096)
