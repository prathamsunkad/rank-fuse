"""Every registered implementation must match the B0 reference.

New implementations (RankFuse kernels, B3, B4) are picked up automatically from the registry.
CUDA-only implementations and fp16/bf16 cases are skipped when no GPU is available.

Every check prints a report (see rankfuse/debug.py). pytest shows it for failing tests;
run with `-s` to see it for passing tests too:

    python -m pytest -q -k b2_grouped -s
"""

import pytest
import torch

import rankfuse
from rankfuse.debug import compare
from rankfuse.interface import REGISTRY, MatryoshkaLoRA
from rankfuse.workloads import DISTRIBUTIONS, Workload, make_batch, make_lora

HAS_CUDA = torch.cuda.is_available()
IMPLS = sorted(n for n in REGISTRY if n != "b0_reference")
SHAPES = {"decode-1": (1, 1), "decode-17": (17, 1), "decode-64": (64, 1), "prefill-5x13": (5, 13)}


def _device_and_skip(impl_name: str, dtype: str) -> str:
    if REGISTRY[impl_name].requires_cuda and not HAS_CUDA:
        pytest.skip("requires CUDA")
    if dtype != "fp32" and not HAS_CUDA:
        pytest.skip("half precision tested on GPU only")
    return "cuda" if HAS_CUDA else "cpu"


def _check(impl_name: str, lora: MatryoshkaLoRA, x: torch.Tensor, token_ranks: torch.Tensor, dtype: str,
           seg_starts: torch.Tensor | None = None):
    report = compare(impl_name, lora, x, token_ranks, dtype, seg_starts)
    print(f"\n{report}")
    assert report.ok, f"\n{report}"


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("dist", list(DISTRIBUTIONS))
@pytest.mark.parametrize("shape", list(SHAPES))
def test_matches_reference(impl, dtype, dist, shape):
    device = _device_and_skip(impl, dtype)
    requests, tokens_per_request = SHAPES[shape]
    w = Workload(requests, tokens_per_request, h_in=256, h_out=192, r_max=32, dist=dist, dtype=dtype, seed=requests)
    lora, batch = make_lora(w, device), make_batch(w, device)
    _check(impl, lora, batch.x, batch.token_ranks, dtype, batch.seg_starts)


@pytest.mark.parametrize("impl", IMPLS)
def test_per_rank_scale(impl):
    device = _device_and_skip(impl, "fp32")
    w = Workload(32, 1, h_in=128, h_out=128, r_max=16, dist="uniform", dtype="fp32")
    lora, batch = make_lora(w, device), make_batch(w, device)
    lora.scale_per_rank = {r: 16.0 / r for r in w.rank_tiers}
    _check(impl, lora, batch.x, batch.token_ranks, "fp32", batch.seg_starts)


@pytest.mark.parametrize("impl", IMPLS)
def test_ranks_outside_tiers(impl):
    """Any prefix rank 1..r_max is legal, not only the default tiers."""
    device = _device_and_skip(impl, "fp32")
    w = Workload(1, 1, h_in=64, h_out=64, r_max=16, dtype="fp32")
    lora = make_lora(w, device)
    token_ranks = torch.tensor([1, 3, 16, 7, 3, 1], dtype=torch.int32, device=device)
    x = torch.randn(len(token_ranks), 64, device=device)
    _check(impl, lora, x, token_ranks, "fp32")


def test_graphed_cache_handles_same_shape_different_distributions():
    device = "cuda" if HAS_CUDA else pytest.skip("requires CUDA")
    for dist in ("uniform", "all_min"):
        w = Workload(17, 1, h_in=256, h_out=192, r_max=32, dist=dist, dtype="fp32", seed=17)
        lora, batch = make_lora(w, device), make_batch(w, device)
        for impl in ("b1_padded_graphed", "b2_grouped_graphed"):
            _check(impl, lora, batch.x, batch.token_ranks, "fp32", batch.seg_starts)


def test_full_rank_equals_plain_lora():
    """Sanity check on the oracle: rank r_max is ordinary LoRA, x @ A @ B."""
    w = Workload(8, 1, h_in=64, h_out=48, r_max=16, dist="all_max", dtype="fp32")
    lora, batch = make_lora(w), make_batch(w)
    y = rankfuse.REGISTRY["b0_reference"](batch.x, lora, batch.token_ranks)
    torch.testing.assert_close(y, batch.x @ lora.A_t.T @ lora.B)
