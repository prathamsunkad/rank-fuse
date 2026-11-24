"""The *_table variants: tile width and rank-row masking from the tile table's limit column (TOKEN_RANKS off).

rf1_ma_table, rf2_ma_splitk_table and rf2_ma_bucketed_table skip the per-token rank read at the start
of each program and follow rank changes only through MultiMeta.load_ (which a serving engine calls
every step). They must give the reference result, be bitwise equal to the legacy kernels (default
launch rules), and replay one captured graph across steps with load_.
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

TABLE = {"rf1_ma_table": "rf1_ma_legacy", "rf2_ma_splitk_table": "rf2_ma_splitk_legacy", "rf2_ma_bucketed_table": "rf2_ma_bucketed_legacy"}
SHAPES = {"4096x4096": (4096, 4096, None), "qkv-896": (896, None, (896, 128, 128)), "down-4864x896": (4864, 896, None)}
TOKENS = {"decode-1": (1, 1), "decode-33": (33, 1), "decode-256": (256, 1), "prefill-2x150": (2, 150), "prefill-1x2048": (1, 2048)}


@functools.lru_cache(maxsize=8)
def _weights(shape, num_adapters, pages="shuffled", r_max=64, dtype="fp16"):
    h_in, h_out, slices = SHAPES[shape]
    w = MultiWorkload(1, 1, num_adapters, h_in=h_in, h_out=h_out or max(slices), pages=pages, r_max=r_max, dtype=dtype,
                      seed=num_adapters)
    return make_weights(w, "cuda", slice_sizes=slices, scale_per_rank={r: 16.0 / r for r in range(1, r_max + 1)})


def _batch(W, shape, tokens, N, r_max=64, dtype="fp16"):
    requests, tpr = TOKENS[tokens]
    return make_batch(MultiWorkload(requests, tpr, N, popularity="zipf", dist="skew_low", h_in=SHAPES[shape][0],
                                    h_out=W.h_out, zero_frac=0.2, seed=requests + N, r_max=r_max, dtype=dtype))


def _close(y, x, W, ta, tr, dtype="fp16"):
    expected = MULTI_REGISTRY["b0_multi"](x, W, ta, tr)
    rtol, atol = TOL[dtype]
    bad = ~torch.isclose(y.float(), expected, rtol=rtol, atol=atol)
    assert not bad.any(), f"{int(bad.any(1).sum())} tokens wrong"


@pytest.mark.parametrize("impl", list(TABLE))
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("N", [1, 7, 64])
@pytest.mark.parametrize("tokens", list(TOKENS))
def test_reference_and_bitwise_legacy(monkeypatch, impl, shape, N, tokens):
    monkeypatch.setattr(rf, "TUNED", {})
    W = _weights(shape, N)
    x, ta, tr = _batch(W, shape, tokens, N)
    m, old = MULTI_REGISTRY[impl], MULTI_REGISTRY[TABLE[impl]]
    kw = m.prepare(x, W, ta, tr)
    y = m(x, W, ta, tr, **kw)
    torch.cuda.synchronize()
    _close(y, x, W, ta, tr)
    assert torch.equal(y, old(x, W, ta, tr, **kw))


@pytest.mark.parametrize("impl", list(TABLE))
@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
@pytest.mark.parametrize("r_max", [128, 256])
def test_dtypes_and_widths(impl, dtype, r_max):
    W = _weights("qkv-896", 9, r_max=r_max, dtype=dtype)
    x, ta, tr = _batch(W, "qkv-896", "prefill-2x150", 9, r_max=r_max, dtype=dtype)
    m = MULTI_REGISTRY[impl]
    _close(m(x, W, ta, tr, **m.prepare(x, W, ta, tr)), x, W, ta, tr, dtype)


@pytest.mark.parametrize("impl,mode", [("rf1_ma_table", "group"), ("rf2_ma_splitk_table", "rank_sorted")])
@pytest.mark.parametrize("T", [32, 256])
def test_one_graph_many_steps_with_load(impl, mode, T):
    N = 24
    W = make_weights(MultiWorkload(1, 1, N, h_in=896, h_out=896, seed=1), "cuda")
    x = torch.randn(T, 896, device="cuda").half()
    ta = torch.zeros(T, dtype=torch.int32, device="cuda")
    tr = torch.zeros(T, dtype=torch.int32, device="cuda")
    cap = tile_capacity(T, 16, N, mode)

    def step(seed):
        gen = torch.Generator().manual_seed(seed)
        a, r = torch.randint(-1, N, (T,), generator=gen), torch.randint(0, 65, (T,), generator=gen)
        return a, r, build_meta(a, r, num_adapters=N, mode=mode, device="cuda", bm=16, r_pad=64, capacity=cap)

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
        g.replay()
        torch.cuda.synchronize()
        _close(out, x, W, ta, tr)
