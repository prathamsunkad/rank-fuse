"""RF2-MA on the shared split-K kernels (rankfuse_fused.py, MULTI mode).

    (i)  RANK_SKIP inside multi-adapter tiles: rf2_ma_splitk (sorted by adapter, rank), its
         full-width control rf2_ma_splitk_fw, rf2_ma_widths (groups of (adapter, width bucket),
         build_meta mode "adapter_bucket"), rf2_ma_segment (batch order)
    (ii) per-width group launches on concurrent streams: rf2_ma_bucketed, rf2_ma_bucketed_seq

Each against B0-multi, each against its *_legacy version (bitwise, default launch rules), the
adapter_bucket tile table's invariants, and CUDA-graph reuse over steps.
"""

import functools

import pytest
import torch

from rankfuse.debug import TOL
from rankfuse.kernels import rankfuse_fused as rf
from rankfuse.multi import MULTI_REGISTRY, MultiWorkload, build_meta, load_all, make_batch, make_weights, tile_capacity

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

if torch.cuda.is_available():
    load_all(include_slora=False)

NEW = ["rf2_ma_splitk", "rf2_ma_splitk_fw", "rf2_ma_widths", "rf2_ma_segment", "rf2_ma_bucketed", "rf2_ma_bucketed_seq"]
LEGACY = {"rf2_ma_splitk": "rf2_ma_splitk_legacy", "rf2_ma_segment": "rf2_ma_segment_legacy",
          "rf2_ma_bucketed": "rf2_ma_bucketed_legacy", "rf2_ma_bucketed_seq": "rf2_ma_bucketed_legacy"}
SHAPES = {"4096x4096": (4096, 4096, None), "qkv-896": (896, None, (896, 128, 128)), "gate_up-896": (896, None, (4864, 4864)),
          "down-4864x896": (4864, 896, None), "7b-3584x512": (3584, 512, None)}
TOKENS = {"decode-1": (1, 1), "decode-33": (33, 1), "decode-256": (256, 1), "prefill-2x150": (2, 150), "prefill-1x2048": (1, 2048)}


@functools.lru_cache(maxsize=16)
def _weights(shape, num_adapters, dtype="fp16", pages="dense", scaled=False, r_max=64):
    h_in, h_out, slices = SHAPES[shape]
    w = MultiWorkload(1, 1, num_adapters, h_in=h_in, h_out=h_out or max(slices), dtype=dtype, pages=pages, r_max=r_max,
                      seed=num_adapters)
    scales = {r: 16.0 / r for r in range(1, r_max + 1)} if scaled else None
    return make_weights(w, "cuda", slice_sizes=slices, scale_per_rank=scales)


def _batch(W, shape, tokens, num_adapters, dtype="fp16", dist="uniform", zero=0.2, popularity="uniform", seed=0, r_max=64):
    requests, tpr = TOKENS[tokens]
    w = MultiWorkload(requests, tpr, num_adapters, popularity=popularity, dist=dist, h_in=SHAPES[shape][0], h_out=W.h_out,
                      dtype=dtype, zero_frac=zero, seed=seed + requests + num_adapters, r_max=r_max)
    return make_batch(w)


def _call(impl, x, W, ta, tr, out=None, add=False):
    m = MULTI_REGISTRY[impl]
    return m(x, W, ta, tr, out=out, add=add, **m.prepare(x, W, ta, tr))


def _close(y, x, W, ta, tr, dtype, out=None, add=False):
    expected = MULTI_REGISTRY["b0_multi"](x, W, ta, tr, out=out, add=add)
    rtol, atol = TOL[dtype]
    bad = ~torch.isclose(y.float(), expected, rtol=rtol, atol=atol)
    assert not bad.any(), f"{int(bad.any(1).sum())}/{x.shape[0]} tokens wrong, max err {float((y.float() - expected).abs().max()):.3g}"


@pytest.mark.parametrize("impl", NEW)
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("num_adapters", [1, 7, 64, 256])
@pytest.mark.parametrize("tokens", list(TOKENS))
@pytest.mark.parametrize("dist", ["uniform", "skew_low"])
def test_matches_reference(impl, shape, num_adapters, tokens, dist):
    W = _weights(shape, num_adapters)
    x, ta, tr = _batch(W, shape, tokens, num_adapters, dist=dist, popularity="zipf" if num_adapters >= 64 else "uniform")
    y = _call(impl, x, W, ta, tr)
    _close(y, x, W, ta, tr, "fp16")
    assert (y[tr == 0] == 0).all()


@pytest.mark.parametrize("impl", NEW)
@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32"])
@pytest.mark.parametrize("r_max", [64, 128, 256])
@pytest.mark.parametrize("pages", ["dense", "shuffled"])
def test_dtypes_widths_scales_add_pages(impl, dtype, r_max, pages):
    W = _weights("qkv-896", 9, dtype=dtype, pages=pages, scaled=True, r_max=r_max)
    for tokens in ("decode-33", "prefill-2x150"):
        x, ta, tr = _batch(W, "qkv-896", tokens, 9, dtype=dtype, dist="skew_low", r_max=r_max)
        base = torch.randn(x.shape[0], W.h_out, device="cuda").to(x.dtype)
        _close(_call(impl, x, W, ta, tr, out=base.clone(), add=True), x, W, ta, tr, dtype, out=base.clone(), add=True)
        _close(_call(impl, x, W, ta, tr), x, W, ta, tr, dtype)


@pytest.mark.parametrize("impl", list(LEGACY))
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("num_adapters", [1, 7, 64, 256])
@pytest.mark.parametrize("tokens", list(TOKENS))
def test_bitwise_equal_to_legacy(monkeypatch, impl, shape, num_adapters, tokens):
    monkeypatch.setattr(rf, "TUNED", {})  # the legacy kernels never used the tuned table
    W = _weights(shape, num_adapters, pages="shuffled", scaled=True)
    x, ta, tr = _batch(W, shape, tokens, num_adapters, dist="skew_low", popularity="zipf")
    new, old = MULTI_REGISTRY[impl], MULTI_REGISTRY[LEGACY[impl]]
    kw = new.prepare(x, W, ta, tr)
    base = torch.randn(x.shape[0], W.h_out, device="cuda").half()
    for add in (False, True):
        a = new(x, W, ta, tr, out=base.clone(), add=add, **kw)
        b = old(x, W, ta, tr, out=base.clone(), add=add, **kw)
        torch.cuda.synchronize()
        assert torch.equal(a, b), f"max diff {float((a.float() - b.float()).abs().max()):.3g}"


def test_full_width_control_equals_rf1_width():
    # rf2_ma_splitk_fw runs every tile at the batch bucket: bitwise rf2_ma_splitk on an all-max batch
    W = _weights("qkv-896", 9)
    x, ta, tr = _batch(W, "qkv-896", "decode-256", 9, dist="all_max", zero=0.0)
    assert torch.equal(_call("rf2_ma_splitk_fw", x, W, ta, tr), _call("rf2_ma_splitk", x, W, ta, tr))


@pytest.mark.parametrize("T", [1, 50, 700])
@pytest.mark.parametrize("num_adapters", [1, 5, 40])
def test_adapter_bucket_tiles(T, num_adapters):
    gen = torch.Generator().manual_seed(T + num_adapters)
    ta = torch.randint(-1, num_adapters, (T,), generator=gen)
    tr = torch.randint(0, 257, (T,), generator=gen)
    m = build_meta(ta, tr, num_adapters=num_adapters, mode="adapter_bucket", device="cpu")
    order = m.order.long()
    assert sorted(order.tolist()) == list(range(T))
    none = (tr == 0) | (ta < 0)
    eff_ad = torch.where(none, -1, ta)
    bucket = torch.where(none, 0, torch.clamp(2 ** torch.ceil(torch.log2(tr.clamp(min=1).double())).long(), min=16))
    seen = torch.zeros(T, dtype=torch.long)
    for a, start, length, limit, first in m.tiles[: m.num_tiles].tolist():
        assert 1 <= length <= m.bm
        toks = order[start:start + length]
        seen[toks] += 1
        assert (eff_ad[toks] == a).all()
        assert bucket[toks].unique().numel() == 1, "a tile spans two width buckets"
        assert limit == (0 if a < 0 else int(tr[toks].max()))
        if first >= 0:
            assert toks.tolist() == list(range(first, first + length))
    assert (seen == 1).all()
    assert m.num_tiles <= tile_capacity(T, m.bm, num_adapters, "adapter_bucket")


@pytest.mark.parametrize("impl,mode", [("rf2_ma_splitk", "rank_sorted"), ("rf2_ma_widths", "adapter_bucket"),
                                       ("rf2_ma_splitk_fw", "rank_sorted"), ("rf2_ma_segment", "segment")])
@pytest.mark.parametrize("T", [32, 256])
def test_one_graph_many_steps(impl, mode, T):
    N = 24
    W = make_weights(MultiWorkload(1, 1, N, h_in=896, h_out=896, seed=1), "cuda", scale_per_rank={r: 16.0 / r for r in range(1, 65)})
    x = torch.randn(T, 896, device="cuda").half()
    ta = torch.zeros(T, dtype=torch.int32, device="cuda")
    tr = torch.zeros(T, dtype=torch.int32, device="cuda")
    bm = 16
    cap = tile_capacity(T, bm, N, mode)

    def step(seed):
        gen = torch.Generator().manual_seed(seed)
        a, r = torch.randint(-1, N, (T,), generator=gen), torch.randint(0, 65, (T,), generator=gen)
        return a, r, build_meta(a, r, num_adapters=N, mode=mode, device="cuda", bm=bm, r_pad=64, capacity=cap)

    a0, r0, meta = step(0)
    ta.copy_(a0)
    tr.copy_(r0)
    out = torch.empty(T, 896, device="cuda").half()
    fn = lambda: MULTI_REGISTRY[impl](x, W, ta, tr, meta=meta, out=out)  # noqa: E731
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for seed in range(1, 5):
        a, r, new = step(seed)
        ta.copy_(a)
        tr.copy_(r)
        meta.load_(new)
        x.normal_()
        g.replay()
        torch.cuda.synchronize()
        _close(out, x, W, ta, tr, "fp16")


def test_bucketed_capture_with_fixed_capacities():
    # rf2_ma_bucketed in a graph: concurrent per-width launches are captured (fork/join) and replay.
    from rankfuse.kernels.rankfuse_multi import BucketPlan
    W = _weights("qkv-896", 9, scaled=True)
    x, ta, tr = _batch(W, "qkv-896", "prefill-2x150", 9, dist="skew_low")
    plan = BucketPlan(ta, tr, W.num_adapters, x.device)
    out = torch.empty(x.shape[0], W.h_out, device="cuda").half()
    fn = lambda: MULTI_REGISTRY["rf2_ma_bucketed"](x, W, ta, tr, plan=plan, out=out)  # noqa: E731
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        x.normal_()
        g.replay()
        torch.cuda.synchronize()
        _close(out, x, W, ta, tr, "fp16")
