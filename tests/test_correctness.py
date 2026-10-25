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
from rankfuse.workloads import DISTRIBUTIONS, DTYPES, Workload, make_batch, make_lora

HAS_CUDA = torch.cuda.is_available()
IMPLS = sorted(n for n in REGISTRY if n != "b0_reference")
SHAPES = {"decode-1": (1, 1), "decode-17": (17, 1), "decode-64": (64, 1), "prefill-5x13": (5, 13)}


def _device_and_skip(impl_name: str, dtype: str) -> str:
    if REGISTRY[impl_name].requires_cuda and not HAS_CUDA:
        pytest.skip("requires CUDA")
    if dtype != "fp32" and not HAS_CUDA:
        pytest.skip("half precision tested on GPU only")
    if REGISTRY[impl_name].dtypes is not None and DTYPES[dtype] not in REGISTRY[impl_name].dtypes:
        pytest.skip(f"{impl_name} does not support {dtype}")
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


def test_full_rank_equals_plain_lora():
    """Sanity check on the oracle: rank r_max is ordinary LoRA, x @ A @ B."""
    w = Workload(8, 1, h_in=64, h_out=48, r_max=16, dist="all_max", dtype="fp32")
    lora, batch = make_lora(w), make_batch(w)
    y = rankfuse.REGISTRY["b0_reference"](batch.x, lora, batch.token_ranks)
    torch.testing.assert_close(y, batch.x @ lora.A_t.T @ lora.B)


# ---------------------------------------------------------------------------------------------
# Edge cases the tests above do not reach: hidden sizes that are not a multiple of the kernel
# tiles, batches large enough to switch kernels to their prefill tiling, ranks above 32,
# rank-0 tokens, and rf1_fused's optional r_limit argument.
# ---------------------------------------------------------------------------------------------


def _check_workload(impl: str, w: Workload, device: str):
    # b2_grouped_graphed caches graphs by tensor address; a new tensor can reuse a freed
    # address, so start each case with an empty cache.
    if HAS_CUDA:
        from rankfuse.baselines.grouped import clear_graph_cache

        clear_graph_cache()
    lora, batch = make_lora(w, device), make_batch(w, device)
    _check(impl, lora, batch.x, batch.token_ranks, w.dtype, batch.seg_starts)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("dtype", ["fp32", "fp16"])
@pytest.mark.parametrize("shape", [(17, 1), (3, 13)], ids=["decode-17", "prefill-3x13"])
def test_uneven_hidden_sizes(impl, dtype, shape):
    """h_in = 200 and h_out = 100 leave a partial tile at the end of both reductions."""
    device = _device_and_skip(impl, dtype)
    w = Workload(*shape, h_in=200, h_out=100, r_max=32, dist="uniform", dtype=dtype, seed=3)
    _check_workload(impl, w, device)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("dtype", ["fp32", "fp16"])
def test_large_batch(impl, dtype):
    """520 prefill tokens: above 128 tokens, rf1_fused switches to 64-token blocks, and
    h_out = 320 (5 column tiles) leaves the last program's column range partly empty."""
    device = _device_and_skip(impl, dtype)
    w = Workload(4, 130, h_in=512, h_out=320, r_max=64, dist="uniform", dtype=dtype, seed=4)
    _check_workload(impl, w, device)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("dtype", ["fp32", "fp16"])
@pytest.mark.parametrize("shape", [(17, 1), (4, 50)], ids=["decode-17", "prefill-4x50"])
def test_large_rank(impl, dtype, shape):
    """r_max = 128 (tiers 1/2/.../128) compiles the 64- and 128-wide rank variants."""
    device = _device_and_skip(impl, dtype)
    w = Workload(*shape, h_in=256, h_out=192, r_max=128, dist="uniform", dtype=dtype, seed=5)
    _check_workload(impl, w, device)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize(
    "ranks", [[0, 4, 0, 16, 32, 0, 8], [0, 0, 0, 0, 0]], ids=["some-zero", "all-zero"]
)
def test_rank_zero(impl, ranks):
    """Rank 0 means no adapter: those rows must be exactly zero. All-zero is r_limit = 0."""
    device = _device_and_skip(impl, "fp32")
    if HAS_CUDA:
        from rankfuse.baselines.grouped import clear_graph_cache

        clear_graph_cache()
    w = Workload(1, 1, h_in=64, h_out=48, r_max=32, dtype="fp32")
    lora = make_lora(w, device)
    token_ranks = torch.tensor(ranks, dtype=torch.int32, device=device)
    x = torch.randn(len(ranks), 64, device=device)
    _check(impl, lora, x, token_ranks, "fp32")
    y = REGISTRY[impl](x, lora, token_ranks)
    assert (y[token_ranks == 0] == 0).all(), "rank-0 rows must be exactly zero"


@pytest.mark.skipif("rf1_fused" not in REGISTRY or not HAS_CUDA, reason="requires rf1_fused on CUDA")
def test_rf1_fused_r_limit():
    """A caller-supplied r_limit (the host-sync-free serving path) gives the same result,
    a larger-than-needed r_limit only adds padding, and out-of-range values are rejected."""
    w = Workload(9, 1, h_in=128, h_out=96, r_max=32, dist="skew_low", dtype="fp32", seed=6)
    lora, batch = make_lora(w, "cuda"), make_batch(w, "cuda")
    impl = REGISTRY["rf1_fused"]
    expected = REGISTRY["b0_reference"](batch.x, lora, batch.token_ranks)
    batch_max = int(batch.token_ranks.max())

    for r_limit in (batch_max, lora.r_max):
        y = impl(batch.x, lora, batch.token_ranks, r_limit=r_limit)
        torch.testing.assert_close(y, expected, rtol=1e-4, atol=1e-4)

    for bad in (-1, lora.r_max + 1):
        with pytest.raises(ValueError):
            impl(batch.x, lora, batch.token_ranks, r_limit=bad)
