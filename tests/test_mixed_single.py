"""Single-adapter mixed (chunked-prefill) steps at r_max 64 / 128 / 256: every implementation the
mixed sweeps time (bench/configs/rf2_mixed_a100_*.yaml) matches B0, on both request orders, adding
into an existing output as the sweeps do. B4 runs at widths 128 / 256 via slora_bgmv_wide.

    python -m pytest -q tests/test_mixed_single.py
"""

import pytest
import torch

from rankfuse.debug import TOL
from rankfuse.workloads import Workload, make_batch, make_lora

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

IMPLS = ["rf1_fused", "rf2_segment", "rf2_index", "rf2_ranges", "rf2_bucketed_tuned", "b1_padded", "b3_vllm",
         "b4_slora_copies", "b4_slora_shared"]
LENGTHS = [None, "chunked_512_512", "serving_2_512", "chunked_512_128"]


@pytest.fixture(scope="module", autouse=True)
def b4_wide():
    from rankfuse.baselines import slora_bgmv as sb
    from rankfuse.baselines import slora_bgmv_wide

    old = sb.NARROW_SIZES
    slora_bgmv_wide.enable()
    yield
    sb.NARROW_SIZES = old
    sb.disable()


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("r_max", [64, 128, 256])
@pytest.mark.parametrize("lengths", LENGTHS)
@pytest.mark.parametrize("order", ["random", "sorted"])
@pytest.mark.parametrize("h", [1024, 4096])
def test_matches_reference(impl, r_max, lengths, order, h):
    import rankfuse  # noqa: F401  (registers implementations)
    from rankfuse.interface import REGISTRY

    w = Workload(num_requests=48, tokens_per_request=1, h_in=h, h_out=h, r_max=r_max, dist="zipf_1", tiers="pow2",
                 request_order=order, seed=r_max, lengths=lengths, prefill_dist="all_max" if lengths else None)
    lora, batch = make_lora(w, "cuda"), make_batch(w, "cuda")
    x, tr = batch.x, batch.token_ranks
    m = REGISTRY[impl]
    base = torch.randn(x.shape[0], h, device="cuda").to(x.dtype)
    out = base.clone()
    kw = m.prepare(x, lora, tr) if m.prepare else {}
    y = m(x, lora, tr, out=out, add=True, **kw)
    expected = base.float() + REGISTRY["b0_reference"](x, lora, tr)
    rtol, atol = TOL["fp16"]
    bad = ~torch.isclose(y.float(), expected, rtol=rtol, atol=atol)
    assert not bad.any(), f"{impl}: {int(bad.any(1).sum())}/{x.shape[0]} tokens wrong, max err " \
                          f"{float((y.float() - expected).abs().max()):.3g}"


def test_b4_wide_shape_check():
    from rankfuse.baselines.slora_bgmv_wide import check_shape

    assert check_shape(256, 4096, 4096) is None and check_shape(64, 128, 128) is None
    assert check_shape(256, 896, 128) and check_shape(256, 256, 4096) and check_shape(512, 4096, 4096)
