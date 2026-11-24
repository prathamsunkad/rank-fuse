"""RF1-MA on the shared split-K kernels (rankfuse_fused.py, MULTI mode): rf1_ma and its options.

    reference      rf1_ma (and the RANK_SKIP / full-width options of multi_lora_splitk) against
                   B0-multi: dense and paged layouts, merged slices, per-adapter scales, rank 0 and
                   adapter -1, r_max 64 / 128 / 256, fp16 / bf16 / fp32, 64-bit offsets
    legacy         rf1_ma against rf1_ma_legacy (the Stage 2 kernels) with the default launch rules
                   (no tuned table): bitwise equal wherever both pick the same launch settings
    graphs         one captured graph replayed over steps after MultiMeta.load_
    single         the single-adapter specializations still give splitk_pass's output after a multi
                   launch (the MULTI parameters default off)
"""

import functools

import pytest
import torch

from rankfuse.debug import TOL
from rankfuse.kernels import rankfuse_fused as rf
from rankfuse.kernels import rankfuse_multi as rm
from rankfuse.multi import MULTI_REGISTRY, MultiWorkload, build_meta, load_all, make_batch, make_weights, tile_capacity

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

if torch.cuda.is_available():
    load_all(include_slora=False)

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
    h_in, _, _ = SHAPES[shape]
    requests, tpr = TOKENS[tokens]
    w = MultiWorkload(requests, tpr, num_adapters, popularity=popularity, dist=dist, h_in=h_in, h_out=W.h_out, dtype=dtype,
                      zero_frac=zero, seed=seed + requests + num_adapters, r_max=r_max)
    return make_batch(w)


def _close(y, x, W, ta, tr, dtype, out=None, add=False):
    expected = MULTI_REGISTRY["b0_multi"](x, W, ta, tr, out=out, add=add)
    rtol, atol = TOL[dtype]
    bad = ~torch.isclose(y.float(), expected, rtol=rtol, atol=atol)
    assert not bad.any(), f"{int(bad.any(1).sum())}/{x.shape[0]} tokens wrong, max err {float((y.float() - expected).abs().max()):.3g}"


VARIANTS = {"rf1_ma": dict(), "rank_skip": dict(rank_skip=True), "full_width": dict(rank_skip=True, full_width=True)}


def _run(variant, x, W, ta, tr, mode=None, out=None, add=False, **meta_kw):
    mode = mode or ("rank_sorted" if variant != "rf1_ma" else "group")
    meta = build_meta(ta, tr, num_adapters=W.num_adapters, mode=mode, device=x.device, **meta_kw)
    return rm.multi_lora_splitk(x, W, tr, meta, out=out, add=add, **VARIANTS[variant])


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("num_adapters", [1, 7, 64, 256])
@pytest.mark.parametrize("tokens", list(TOKENS))
def test_matches_reference(variant, shape, num_adapters, tokens):
    W = _weights(shape, num_adapters)
    x, ta, tr = _batch(W, shape, tokens, num_adapters, popularity="zipf" if num_adapters >= 64 else "uniform")
    y = _run(variant, x, W, ta, tr)
    _close(y, x, W, ta, tr, "fp16")
    assert (y[tr == 0] == 0).all()


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32"])
@pytest.mark.parametrize("r_max", [64, 128, 256])
@pytest.mark.parametrize("tokens", ["decode-33", "prefill-2x150"])
def test_dtypes_widths_scales_add(variant, dtype, r_max, tokens):
    W = _weights("qkv-896", 9, dtype=dtype, scaled=True, r_max=r_max)
    x, ta, tr = _batch(W, "qkv-896", tokens, 9, dtype=dtype, dist="skew_low", r_max=r_max)
    base = torch.randn(x.shape[0], W.h_out, device="cuda").to(x.dtype)
    y = _run(variant, x, W, ta, tr, out=base.clone(), add=True)
    _close(y, x, W, ta, tr, dtype, out=base.clone(), add=True)
    _close(_run(variant, x, W, ta, tr), x, W, ta, tr, dtype)


@pytest.mark.parametrize("pages", ["contiguous", "shuffled"])
@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("mode", ["group", "rank_sorted", "segment"])
def test_paged_layouts_and_modes(pages, variant, mode):
    W = _weights("qkv-896", 12, pages=pages)
    for tokens in ("decode-33", "prefill-2x150"):
        x, ta, tr = _batch(W, "qkv-896", tokens, 12)
        _close(_run(variant, x, W, ta, tr, mode=mode), x, W, ta, tr, "fp16")


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_64bit_offsets(monkeypatch, variant):
    # Pools of 2**31 elements are too big for a quick test: force the OFF64 code path instead.
    W = _weights("qkv-896", 12, pages="shuffled", scaled=True)
    x, ta, tr = _batch(W, "qkv-896", "decode-33", 12)
    want = _run(variant, x, W, ta, tr)
    real = rm.multi_tiles
    monkeypatch.setattr(rm, "multi_tiles", lambda W, tiles, order: real(W, tiles, order)._replace(off64=True))
    got = _run(variant, x, W, ta, tr)
    assert torch.equal(got, want)


def test_partially_resident_pages():
    # Pages above an adapter's highest requested rank are not resident (page_table -1): never read.
    N = 8
    W = _weights("qkv-896", N, pages="shuffled")
    x, ta, tr = _batch(W, "qkv-896", "decode-33", N, zero=0.0)
    pt = W.page_table.clone()
    tops = torch.zeros(N, dtype=torch.long)
    for a, r in zip(ta.tolist(), tr.tolist()):
        tops[a] = max(tops[a], r)
    for a in range(N):
        pt[a, (int(tops[a]) + 15) // 16:] = -1
    W2 = rm.PagedWeights(**{**W.__dict__, "page_table": pt.contiguous(), "_cache": {}})
    for variant in VARIANTS:
        _close(_run(variant, x, W2, ta, tr), x, W, ta, tr, "fp16")


@pytest.fixture
def default_rules(monkeypatch):
    # The legacy kernels never used the tuned table: compare at the default launch rules.
    monkeypatch.setattr(rf, "TUNED", {})


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("num_adapters", [1, 7, 64, 256])
@pytest.mark.parametrize("tokens", list(TOKENS))
@pytest.mark.parametrize("pages", ["dense", "shuffled"])
def test_bitwise_equal_to_legacy(default_rules, shape, num_adapters, tokens, pages):
    W = _weights(shape, num_adapters, pages=pages, scaled=True)
    x, ta, tr = _batch(W, shape, tokens, num_adapters, popularity="zipf")
    meta = build_meta(ta, tr, num_adapters=num_adapters, mode="group", device="cuda")
    base = torch.randn(x.shape[0], W.h_out, device="cuda").half()
    for add in (False, True):
        new = MULTI_REGISTRY["rf1_ma"](x, W, ta, tr, meta=meta, out=base.clone(), add=add)
        old = MULTI_REGISTRY["rf1_ma_legacy"](x, W, ta, tr, meta=meta, out=base.clone(), add=add)
        assert torch.equal(new, old), f"max diff {float((new.float() - old.float()).abs().max()):.3g}"


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("mode", ["group", "rank_sorted"])
@pytest.mark.parametrize("T", [32, 256, 1024])
def test_one_graph_many_steps(variant, mode, T):
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
    fn = lambda: rm.multi_lora_splitk(x, W, tr, meta, out=out, **VARIANTS[variant])  # noqa: E731
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


def test_single_adapter_unchanged_after_multi_launch():
    # The same kernels in both modes: a multi launch must not disturb the single-adapter results.
    W = _weights("qkv-896", 7)
    x, ta, tr = _batch(W, "qkv-896", "decode-33", 7)
    g = torch.Generator(device="cuda").manual_seed(3)
    A = (torch.randn(3, 64, 896, generator=g, device="cuda") * 0.05).half()
    B = (torch.randn(3, 64, 896, generator=g, device="cuda") * 0.05).half()
    ranks = torch.randint(0, 65, (33,), generator=g, device="cuda").int()
    sizes = (896, 128, 128)
    want = [rf.rankfuse_lora(x, A, B, ranks, slice_sizes=sizes, r_limit=64, **kw).clone()
            for kw in (dict(), dict(rank_skip=True), dict(rank_skip=True, rank_sort=True))]
    _run("rf1_ma", x, W, ta, tr)
    _run("rank_skip", x, W, ta, tr)
    got = [rf.rankfuse_lora(x, A, B, ranks, slice_sizes=sizes, r_limit=64, **kw)
           for kw in (dict(), dict(rank_skip=True), dict(rank_skip=True, rank_sort=True))]
    for a, b in zip(got, want):
        assert torch.equal(a, b)


def test_registry_names():
    for name in ("rf1_ma", "rf1_ma_legacy", "rf2_ma_splitk", "rf2_ma_segment", "rf2_ma_bucketed", "rf_ma_auto"):
        assert name in MULTI_REGISTRY
