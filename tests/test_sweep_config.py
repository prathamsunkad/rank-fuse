"""Sweep grid expansion and the rf2_mixed step types (CPU, plus a CUDA check against b0_reference)."""

from pathlib import Path

import pytest
import torch
import yaml

from bench.run_sweep import expand_grid
from rankfuse.interface import REGISTRY
from rankfuse.workloads import is_known_dist

CONFIGS = Path(__file__).resolve().parent.parent / "bench" / "configs"


def test_grid_list_joins_each_grid():
    one = {"num_requests": [1, 2], "tokens_per_request": 1}
    two = {"num_requests": 4, "tokens_per_request": [8, 16], "r_max": 32}
    joined = expand_grid([one, two])
    assert joined == expand_grid(one) + expand_grid(two)
    assert [(w.num_requests, w.tokens_per_request, w.r_max) for _, w in joined] == [(1, 1, 64), (2, 1, 64), (4, 8, 32), (4, 16, 32)]


def test_rf2_mixed_step_types():
    cfg = yaml.safe_load((CONFIGS / "rf2_mixed.yaml").read_text())
    assert all(name in REGISTRY for name in cfg["impls"])
    workloads = [w for _, w in expand_grid(cfg["grid"])]
    assert len({(s, w) for s, w in expand_grid(cfg["grid"])}) == len(workloads)  # no repeated points
    assert all(is_known_dist(d) for w in workloads for d in (w.dist, w.prefill_dist) if d is not None)
    by_lengths: dict[str | None, list] = {}
    for w in workloads:
        by_lengths.setdefault(w.lengths, []).append(w)
    # 1. decode only
    assert all(w.request_lengths == [1] * w.num_requests and w.prefill_dist is None for w in by_lengths[None])
    for w in workloads:
        if w.lengths is None:
            continue
        lens, prefill = w.request_lengths, w.request_is_prefill
        assert prefill.count(False) == w.num_requests
        budget = int(w.lengths.split("_")[-1] if w.lengths.startswith("serving_") else w.lengths.split("_")[1])
        if w.lengths.startswith("chunked_"):  # 2 and 4 fill the budget
            assert sum(lens) == budget
        else:  # 3: sampled prompts within the budget
            assert sum(lens) <= budget
        if w.lengths in ("chunked_2048_2048", "chunked_8192_8192"):  # 2: exactly one prompt
            assert prefill.count(True) == 1


MIXED_STEPS = [  # (lengths, prefill_dist): one or two per step type, at the sweep's token counts
    (None, None),                       # 1. decode only
    ("chunked_2048_2048", "all_max"),   # 2. one prompt fills the budget
    ("serving_1_2048", "skew_high"),    # 3. short prompts
    ("serving_2_8192", None),
    ("chunked_8192_512", "all_max"),    # 4. many partial prefills
    ("chunked_2048_512", None),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("lengths, prefill_dist", MIXED_STEPS)
@pytest.mark.parametrize("order", ["random", "sorted"])
@pytest.mark.parametrize("r_max", [128, 256])
def test_rf2_mixed_impls_match_reference(lengths, prefill_dist, order, r_max):
    """Every rf2_mixed implementation, called as run_sweep calls it (add mode, prepare's per-step
    arguments), matches b0_reference on each step type. Small h, the sweep's token counts."""
    from rankfuse.workloads import Workload, make_batch, make_lora

    cfg = yaml.safe_load((CONFIGS / "rf2_mixed.yaml").read_text())
    w = Workload(256, 1, h_in=512, h_out=384, r_max=r_max, tiers="pow2", dist="zipf_1", prefill_dist=prefill_dist,
                 lengths=lengths, request_order=order, dtype="fp16", seed=1)
    lora, batch = make_lora(w, "cuda"), make_batch(w, "cuda")
    args = (batch.x, lora, batch.token_ranks)
    expected = REGISTRY["b0_reference"](*args).float()
    for name in dict.fromkeys(cfg["impls"]):
        impl = REGISTRY[name]
        kwargs = impl.prepare(*args) if impl.prepare else {}
        base = torch.randn(batch.num_tokens, w.h_out, dtype=batch.x.dtype, device="cuda")
        out = base.clone()
        impl(*args, out=out, add=True, **kwargs)
        torch.testing.assert_close(out.float() - base.float(), expected, rtol=2e-2, atol=2e-2, msg=lambda m: f"{name}: {m}")


@pytest.mark.parametrize("seed", range(3))
def test_rf2_mixed_decode_share_spans_heavy_and_light(seed):
    """Short-prompt steps make decode-heavy mixed steps; the chunked ones stay prefill-heavy."""
    from rankfuse.workloads import Workload

    share = lambda lengths, n: n / Workload(n, 1, seed=seed, lengths=lengths).num_tokens  # noqa: E731
    assert share("chunked_8192_8192", 256) == 256 / 8192
    assert share("serving_1_2048", 256) >= 256 / 2048  # the prompt can take at most the rest of the budget
    assert share("serving_2_8192", 256) > share("chunked_8192_8192", 256)
