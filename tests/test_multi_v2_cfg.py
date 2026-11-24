"""Launch settings of the multi mode: the legacy settings by default, the tuned table only in *_tunedcfg.

rankfuse_fused.TUNED (tuned_a100.py) was tuned on single-adapter token blocks. With a tuned table in
force on this GPU (the A100 table, patched to this GPU's name), the default multi variants must still
launch with the legacy settings (bitwise equal to the legacy kernels), and the *_tunedcfg variants,
which use it, must still give the reference result.
"""

import functools

import pytest
import torch

from rankfuse.debug import TOL
from rankfuse.kernels import rankfuse_fused as rf
from rankfuse.kernels import tuned_a100
from rankfuse.multi import MULTI_REGISTRY, MultiWorkload, load_all, make_batch, make_weights

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

if torch.cuda.is_available():
    load_all(include_slora=False)

PAIRS = {"rf1_ma": "rf1_ma_legacy", "rf2_ma_splitk": "rf2_ma_splitk_legacy", "rf2_ma_bucketed": "rf2_ma_bucketed_legacy"}
TUNEDCFG = ["rf1_ma_tunedcfg", "rf2_ma_splitk_tunedcfg", "rf2_ma_bucketed_tunedcfg"]
TOKENS = {"decode-33": (33, 1), "decode-256": (256, 1), "prefill-4x300": (4, 300), "prefill-1x5000": (1, 5000)}


@pytest.fixture
def a100_table(monkeypatch):
    monkeypatch.setattr(rf, "TUNED", tuned_a100.TUNED)
    monkeypatch.setattr(rf, "TUNED_GPU", torch.cuda.get_device_name(0))


@functools.lru_cache(maxsize=8)
def _weights(r_max, N):
    w = MultiWorkload(1, 1, N, h_in=896, h_out=896, pages="shuffled", r_max=r_max, seed=N)
    return make_weights(w, "cuda", slice_sizes=(896, 128, 128), scale_per_rank={r: 16.0 / r for r in range(1, r_max + 1)})


def _batch(W, tokens, N, r_max):
    requests, tpr = TOKENS[tokens]
    return make_batch(MultiWorkload(requests, tpr, N, popularity="zipf", dist="skew_low", h_in=896, h_out=W.h_out,
                                    zero_frac=0.2, seed=requests + N, r_max=r_max))


@pytest.mark.parametrize("impl", list(PAIRS))
@pytest.mark.parametrize("r_max", [64, 128, 256])
@pytest.mark.parametrize("tokens", list(TOKENS))
def test_default_ignores_tuned_table(a100_table, impl, r_max, tokens):
    W = _weights(r_max, 12)
    x, ta, tr = _batch(W, tokens, 12, r_max)
    m = MULTI_REGISTRY[impl]
    kw = m.prepare(x, W, ta, tr)
    a = m(x, W, ta, tr, **kw)
    b = MULTI_REGISTRY[PAIRS[impl]](x, W, ta, tr, **kw)
    torch.cuda.synchronize()
    assert torch.equal(a, b)


@pytest.mark.parametrize("impl", TUNEDCFG)
@pytest.mark.parametrize("r_max", [64, 128, 256])
@pytest.mark.parametrize("tokens", list(TOKENS))
def test_tunedcfg_matches_reference(a100_table, impl, r_max, tokens):
    W = _weights(r_max, 12)
    x, ta, tr = _batch(W, tokens, 12, r_max)
    m = MULTI_REGISTRY[impl]
    y = m(x, W, ta, tr, **m.prepare(x, W, ta, tr))
    expected = MULTI_REGISTRY["b0_multi"](x, W, ta, tr)
    rtol, atol = TOL["fp16"]
    assert torch.isclose(y.float(), expected, rtol=rtol, atol=atol).all()
