"""Stage 2 at r_max 128 and 256 (A100 sweeps): every multi-adapter implementation must match
B0-multi with tiers 1, 2, 4, ..., r_max, and survive CUDA-graph capture and replay.

tests/test_multi_correctness.py and tests/test_multi_graphs.py cover r_max 64; this file adds the
larger ranks only. B4 (S-LoRA bgmv) runs r_max 128 / 256 where the delta width is below both
hidden sizes (so not at 896 -> 128) and must report the rest as unsupported.

    python -m pytest -q tests/test_multi_rmax.py
"""

import functools

import pytest
import torch

from rankfuse.debug import TOL
from rankfuse.multi import MULTI_REGISTRY, MultiWorkload, load_all, make_batch, make_weights

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

if torch.cuda.is_available():
    load_all(include_slora=False)  # B4 is added by the fixture below
IMPLS = ["rf1_ma", "rf2_ma_splitk", "rf2_ma_segment", "rf2_ma_bucketed", "rf_ma_auto",
         "b1_multi", "b3_multi", "b4_multi_copies", "b4_multi_shared"]
SHAPES = {"4096x4096": (4096, 4096), "k_v-896x128": (896, 128), "down-4864x896": (4864, 896),
          "kv7-3584x512": (3584, 512), "gate_up7-3584x18944": (3584, 18944)}
TOKENS = {"decode-1": (1, 1), "decode-17": (17, 1), "decode-128": (128, 1), "prefill-2x300": (2, 300)}
MIXES = {"uniform": ("uniform", 0.0), "all_max": ("all_max", 0.0), "skew_low": ("skew_low", 0.0),
         "some_zero": ("uniform", 0.3)}


@pytest.fixture(scope="module", autouse=True)
def b4_registered():
    from rankfuse.baselines import slora_bgmv_multi

    slora_bgmv_multi.enable()
    yield
    slora_bgmv_multi.disable()


@functools.lru_cache(maxsize=4)
def _weights(h_in, h_out, num_adapters, r_max, pages="shuffled"):
    w = MultiWorkload(1, 1, num_adapters, h_in=h_in, h_out=h_out, r_max=r_max, dtype="fp16", pages=pages,
                      seed=num_adapters)
    return make_weights(w, "cuda")


def _skip_unsupported(impl, W):
    reason = MULTI_REGISTRY[impl].unsupported(W)
    if reason:
        pytest.skip(reason)


def _check(impl, x, W, ta, tr, out=None, add=False, **prepared):
    m = MULTI_REGISTRY[impl]
    kw = m.prepare(x, W, ta, tr) if (m.prepare and not prepared) else prepared
    base = None if out is None else out.clone()
    y = m(x, W, ta, tr, out=out, add=add, **kw)
    expected = MULTI_REGISTRY["b0_multi"](x, W, ta, tr, out=base, add=add)
    rtol, atol = TOL["fp16"]
    bad = ~torch.isclose(y.float(), expected, rtol=rtol, atol=atol)
    assert not bad.any(), (f"{impl}: {int(bad.any(1).sum())}/{x.shape[0]} tokens wrong, "
                           f"max err {float((y.float() - expected).abs().max()):.3g}")
    return y


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("r_max", [128, 256])
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("num_adapters", [1, 5, 64])
@pytest.mark.parametrize("tokens", list(TOKENS))
@pytest.mark.parametrize("mix", list(MIXES))
def test_matches_reference_large_rmax(impl, r_max, shape, num_adapters, tokens, mix):
    h_in, h_out = SHAPES[shape]
    W = _weights(h_in, h_out, num_adapters, r_max)
    _skip_unsupported(impl, W)
    requests, tpr = TOKENS[tokens]
    dist, zero = MIXES[mix]
    w = MultiWorkload(requests, tpr, num_adapters, popularity="zipf" if num_adapters == 64 else "uniform", dist=dist,
                      h_in=h_in, h_out=h_out, r_max=r_max, dtype="fp16", zero_frac=zero, seed=requests + r_max)
    x, ta, tr = make_batch(w)
    if mix == "all_max":
        assert int(tr.min()) == r_max
    y = _check(impl, x, W, ta, tr)
    assert (y[tr == 0] == 0).all(), "rank-0 rows must be exactly zero"


@pytest.mark.parametrize("r_max", [128, 256])
def test_b4_rank_width_limits(r_max):
    """B4 runs r_max 128 / 256 at hidden sizes above the delta width and reports 896 -> 128."""
    from rankfuse.baselines.slora_bgmv_multi import check

    assert check(_weights(4096, 4096, 1, r_max)) is None
    assert check(_weights(3584, 512, 1, r_max)) is None
    assert "below h_in and h_out" in check(_weights(896, 128, 1, r_max))


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("r_max", [128, 256])
@pytest.mark.parametrize("tokens", ["decode-128", "prefill-2x300"])
def test_capture_and_replay_large_rmax(impl, r_max, tokens):
    """The prepared call captured in a CUDA graph and replayed adds the same delta as eager."""
    W = _weights(4096, 4096, 16, r_max)
    _skip_unsupported(impl, W)
    requests, tpr = TOKENS[tokens]
    x, ta, tr = make_batch(MultiWorkload(requests, tpr, 16, h_in=4096, h_out=4096, r_max=r_max, seed=3))
    m = MULTI_REGISTRY[impl]
    kw = m.prepare(x, W, ta, tr) if m.prepare else {}
    base = torch.randn(x.shape[0], W.h_out, device="cuda").half()
    out = base.clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        m(x, W, ta, tr, out=out, add=True, **kw)
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        m(x, W, ta, tr, out=out, add=True, **kw)
    for _ in range(2):
        out.copy_(base)
        g.replay()
    torch.cuda.synchronize()
    expected = MULTI_REGISTRY["b0_multi"](x, W, ta, tr, out=base.clone(), add=True)
    rtol, atol = TOL["fp16"]
    assert torch.isclose(out.float(), expected, rtol=rtol, atol=atol).all()
