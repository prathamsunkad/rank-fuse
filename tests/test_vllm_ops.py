"""B3 (vLLM's own LoRA kernels, rankfuse/baselines/vllm_ops.py) beyond the common tests.

tests/test_correctness.py runs b3_vllm in fp16/bf16 only (vLLM's shrink takes no fp32), so its
rank-0 and per-rank-scale checks, which run in fp32, skip it; they are repeated here in fp16,
with add mode, ranks outside the default tiers and CUDA-graph capture with prepared metadata.
"""

import pytest
import torch

import rankfuse  # noqa: F401  (registers implementations)
from rankfuse.debug import TOL
from rankfuse.interface import REGISTRY
from rankfuse.workloads import Workload, make_batch, make_lora

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and "b3_vllm" in REGISTRY), reason="requires vLLM on CUDA"
)


def _assert_close(got, expected, dtype="fp16"):
    rtol, atol = TOL[dtype]
    torch.testing.assert_close(got.float(), expected.float(), rtol=rtol, atol=atol)


def _setup(T=37, r_max=32, ranks=None, seed=0):
    w = Workload(T, 1, h_in=256, h_out=192, r_max=r_max, dist="uniform", dtype="fp16", seed=seed)
    lora, batch = make_lora(w, "cuda"), make_batch(w, "cuda")
    token_ranks = batch.token_ranks if ranks is None else torch.tensor(ranks, dtype=torch.int32, device="cuda")
    x = batch.x if ranks is None else torch.randn(len(ranks), 256, device="cuda", dtype=torch.float16)
    return x, lora, token_ranks


@pytest.mark.parametrize("ranks", [[0, 4, 0, 16, 32, 0, 8], [0, 0, 0, 0, 0]], ids=["some-zero", "all-zero"])
def test_rank_zero(ranks):
    x, lora, token_ranks = _setup(ranks=ranks)
    y = REGISTRY["b3_vllm"](x, lora, token_ranks)
    _assert_close(y, REGISTRY["b0_reference"](x, lora, token_ranks))
    assert (y[token_ranks == 0] == 0).all(), "rank-0 rows must be exactly zero"


def test_per_rank_scale_and_add():
    """Scales folded into the B copies; add=True adds into an existing output."""
    x, lora, token_ranks = _setup(seed=1)
    lora.scale_per_rank = {r: 16.0 / r for r in (1, 2, 4, 8, 16, 32)}
    base = torch.randn(x.shape[0], 192, device="cuda", dtype=torch.float16)
    out = base.clone()
    REGISTRY["b3_vllm"](x, lora, token_ranks, out=out, add=True)
    _assert_close(out, base.float() + REGISTRY["b0_reference"](x, lora, token_ranks))


def test_ranks_outside_default_tiers():
    """A rank that is no tier (3, 20) gets its own copy; earlier copies are kept."""
    x, lora, token_ranks = _setup(ranks=[3, 32, 20, 1, 3, 0])
    _assert_close(REGISTRY["b3_vllm"](x, lora, token_ranks), REGISTRY["b0_reference"](x, lora, token_ranks))
    assert {3, 20} <= set(lora._b3_copies[1].ranks)


def test_cuda_graph_capture_with_prepared_step():
    """With copies and step prepared outside, the call has no host sync: capture, then replay on new x."""
    impl = REGISTRY["b3_vllm"]
    x, lora, token_ranks = _setup(T=64, seed=2)
    meta = impl.prepare(x, lora, token_ranks)
    out = torch.empty(64, 192, device="cuda", dtype=torch.float16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        impl(x, lora, token_ranks, out=out, **meta)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        impl(x, lora, token_ranks, out=out, **meta)
    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.cuda.synchronize()
    _assert_close(out, REGISTRY["b0_reference"](x, lora, token_ranks))
