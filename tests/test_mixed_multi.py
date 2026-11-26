"""Multi-adapter mixed (chunked-prefill) steps (rankfuse/mixed_multi.py): the workload, and every
multi-adapter implementation against B0-multi on mixed steps at r_max 64 / 128 / 256.

    python -m pytest -q tests/test_mixed_multi.py
"""

import functools

import pytest
import torch

from rankfuse.debug import TOL
from rankfuse.mixed_multi import MixedMultiWorkload, make_batch, sample_requests, step_type
from rankfuse.workloads import make_batch as single_make_batch

LENGTHS = [None, "chunked_512_512", "serving_2_512", "chunked_512_128"]


def test_step_types():
    assert [step_type(s) for s in LENGTHS] == ["decode_only", "one_prompt_fills", "short_prompts", "partial_prefills"]
    assert step_type("chunked_8192_2048") == "partial_prefills"


@pytest.mark.parametrize("lengths", LENGTHS)
@pytest.mark.parametrize("prefill_dist", [None, "all_max"])
def test_same_shape_and_ranks_as_single_adapter(lengths, prefill_dist):
    """Request lengths, order and ranks are the single-adapter step's with the same parameters."""
    w = MixedMultiWorkload(32, lengths, 16, r_max=128, seed=3, h_in=64, h_out=64,
                           prefill_dist=prefill_dist if lengths else None)
    lens, adapters, ranks, pre = sample_requests(w)
    sb = single_make_batch(w.shape_workload())
    assert torch.diff(sb.seg_starts.cpu()).tolist() == lens
    x, ta, tr = make_batch(w, "cpu")
    assert x.shape == (sum(lens), 64) and ta.shape == tr.shape == (sum(lens),)
    assert torch.equal(tr.long(), sb.token_ranks.long().cpu())
    assert sum(not p for p in pre) == 32 and all(n == 1 for n, p in zip(lens, pre) if not p)
    if lengths and lengths.startswith("chunked_"):
        assert sum(lens) == int(lengths.split("_")[1])
    if prefill_dist == "all_max":
        assert all(r == 128 for r, p in zip(ranks, pre) if p)
    # a request's tokens are consecutive and share one adapter and rank
    starts = torch.tensor([0] + lens[:-1]).cumsum(0).tolist()
    for s, n, a, r in zip(starts, lens, adapters, ranks):
        assert (ta[s:s + n] == a).all() and (tr[s:s + n] == r).all()
    row = w.as_row()
    assert row["decode_tokens"] + row["prefill_tokens"] == row["num_tokens"] == sum(lens)


def test_zipf_popularity_skews_to_adapter_zero():
    counts = {}
    for alpha in (0.8, 1.2):
        _, adapters, _, _ = sample_requests(MixedMultiWorkload(256, "chunked_2048_64", 64, zipf_alpha=alpha))
        counts[alpha] = adapters.count(0) / len(adapters)
    assert counts[1.2] > counts[0.8] > 1 / 64


IMPLS = ["rf1_ma", "rf1_ma_legacy", "rf2_ma_splitk", "rf2_ma_widths", "rf2_ma_segment", "rf2_ma_bucketed",
         "rf2_ma_splitk_legacy", "rf2_ma_segment_legacy", "rf2_ma_bucketed_legacy", "b1_multi", "b3_multi",
         "b4_multi_copies", "b4_multi_shared"]


@functools.lru_cache(maxsize=4)
def _weights(num_adapters, r_max):
    from rankfuse.multi import make_weights

    return make_weights(MixedMultiWorkload(1, None, num_adapters, h_in=1024, h_out=1024, r_max=r_max).weights_workload(),
                        "cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("r_max", [64, 128, 256])
@pytest.mark.parametrize("lengths", LENGTHS)
@pytest.mark.parametrize("num_adapters", [4, 64])
def test_matches_reference(impl, r_max, lengths, num_adapters):
    from rankfuse.multi import MULTI_REGISTRY, load_all

    load_all(include_slora=True)
    W = _weights(num_adapters, r_max)
    m = MULTI_REGISTRY[impl]
    reason = m.unsupported(W)
    if reason:
        pytest.skip(reason)
    w = MixedMultiWorkload(48, lengths, num_adapters, h_in=1024, h_out=1024, r_max=r_max, seed=r_max + num_adapters,
                           prefill_dist="all_max" if lengths else None)
    x, ta, tr = make_batch(w)
    base = torch.randn(x.shape[0], 1024, device="cuda").to(x.dtype)
    out = base.clone()
    kw = m.prepare(x, W, ta, tr) if m.prepare else {}
    y = m(x, W, ta, tr, out=out, add=True, **kw)
    expected = MULTI_REGISTRY["b0_multi"](x, W, ta, tr, out=base.clone(), add=True)
    rtol, atol = TOL["fp16"]
    bad = ~torch.isclose(y.float(), expected.float(), rtol=rtol, atol=atol)
    assert not bad.any(), f"{impl}: {int(bad.any(1).sum())}/{x.shape[0]} tokens wrong, max err " \
                          f"{float((y.float() - expected.float()).abs().max()):.3g}"
