"""B4 (S-LoRA's bgmv kernel, rankfuse/baselines/slora_bgmv.py) must match the B0 reference.

B4 is registered only on request (its kernel is compiled for fixed hidden sizes, r_max <= 64 and
fp16/bf16, so tests/test_correctness.py's generic shapes do not apply). The fixture below
registers it for these tests and removes it again afterwards. The first test builds the CUDA
extension (a few minutes; cached afterwards in $TORCH_EXTENSIONS_DIR).

    python -m pytest -q tests/test_slora_bgmv.py
"""

import pytest
import torch

from rankfuse.debug import compare
from rankfuse.interface import REGISTRY
from rankfuse.workloads import DTYPES, Workload, make_batch, make_lora

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

MODES = ["b4_slora_copies", "b4_slora_shared"]
SHAPES = {"4096x4096": (4096, 4096), "qo-896x896": (896, 896), "kv-896x128": (896, 128),
          "gate_up-896x4864": (896, 4864), "down-4864x896": (4864, 896)}


@pytest.fixture(scope="module", autouse=True)
def b4_registered():
    from rankfuse.baselines import slora_bgmv

    slora_bgmv.enable()
    yield
    slora_bgmv.disable()


def _check(impl, lora, x, token_ranks, dtype, seg_starts=None):
    report = compare(impl, lora, x, token_ranks, dtype, seg_starts)
    print(f"\n{report}")
    assert report.ok, f"\n{report}"


def _scales(r_max: int) -> dict[int, float]:
    return {r: 16.0 / r for r in range(1, r_max + 1)}


@pytest.mark.parametrize("impl", MODES)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("T", [1, 17, 64, 128])
@pytest.mark.parametrize("dist", ["uniform", "skew_low", "all_max"])
@pytest.mark.parametrize("scaled", [False, True], ids=["unscaled", "scaled"])
def test_decode_matches_reference(impl, dtype, shape, T, dist, scaled):
    h_in, h_out = SHAPES[shape]
    w = Workload(T, 1, h_in=h_in, h_out=h_out, r_max=64, dist=dist, dtype=dtype, seed=T)
    lora, batch = make_lora(w, "cuda"), make_batch(w, "cuda")
    if scaled:
        lora.scale_per_rank = _scales(64)
    _check(impl, lora, batch.x, batch.token_ranks, dtype, batch.seg_starts)


@pytest.mark.parametrize("impl", MODES)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("shape", ["4096x4096", "kv-896x128", "down-4864x896"])
def test_rank_zero_and_odd_ranks(impl, dtype, shape):
    """Rank-0 tokens get exactly zero; ranks outside the tiers (3, 12, 40) and below 8 work."""
    h_in, h_out = SHAPES[shape]
    w = Workload(1, 1, h_in=h_in, h_out=h_out, r_max=64, dtype=dtype)
    lora = make_lora(w, "cuda")
    lora.scale_per_rank = _scales(64)
    token_ranks = torch.tensor([0, 1, 0, 2, 3, 4, 8, 12, 0, 40, 64, 0], dtype=torch.int32, device="cuda")
    x = torch.randn(len(token_ranks), h_in, device="cuda").to(DTYPES[dtype])
    _check(impl, lora, x, token_ranks, dtype)
    y = REGISTRY[impl](x, lora, token_ranks)
    assert (y[token_ranks == 0] == 0).all(), "rank-0 rows must be exactly zero"


@pytest.mark.parametrize("impl", MODES)
def test_all_rank_zero(impl):
    w = Workload(1, 1, h_in=896, h_out=896, r_max=64, dtype="fp16")
    lora = make_lora(w, "cuda")
    token_ranks = torch.zeros(5, dtype=torch.int32, device="cuda")
    x = torch.randn(5, 896, device="cuda", dtype=torch.float16)
    assert (REGISTRY[impl](x, lora, token_ranks) == 0).all()


@pytest.mark.parametrize("impl", MODES)
def test_add_mode_leaves_rank_zero_rows_untouched(impl):
    """add=True: out += delta (as S-LoRA adds into the projection output)."""
    w = Workload(33, 1, h_in=896, h_out=4864, r_max=64, dist="uniform", dtype="fp16", seed=7)
    lora, batch = make_lora(w, "cuda"), make_batch(w, "cuda")
    tr = batch.token_ranks.clone()
    tr[::4] = 0
    base = torch.randn(tr.numel(), 4864, device="cuda", dtype=torch.float16)
    out = base.clone()
    REGISTRY[impl](batch.x, lora, tr, out=out, add=True)
    expected = base.float() + REGISTRY["b0_reference"](batch.x, lora, tr).float()
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)
    assert torch.equal(out[tr == 0], base[tr == 0])


@pytest.mark.parametrize("impl", MODES)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("tokens", [(4, 512), (1, 2048)], ids=["prefill-4x512", "prefill-1x2048"])
def test_prefill_sized_batches(impl, dtype, tokens):
    """bgmv on prompt-sized batches (not how S-LoRA runs prompts, but B4's prefill rows use it)."""
    w = Workload(*tokens, h_in=4096, h_out=4096, r_max=64, dist="uniform", dtype=dtype, seed=8)
    lora, batch = make_lora(w, "cuda"), make_batch(w, "cuda")
    _check(impl, lora, batch.x, batch.token_ranks, dtype, batch.seg_starts)


@pytest.mark.parametrize("impl", MODES)
def test_cuda_graph_capture_and_replay(impl):
    """Captured with prepare()'s metadata (as run_sweep --timing graph does), replay recomputes
    from the current contents of x."""
    fn = REGISTRY[impl]
    w = Workload(32, 1, h_in=896, h_out=896, r_max=64, dist="uniform", dtype="fp16", seed=9)
    lora, batch = make_lora(w, "cuda"), make_batch(w, "cuda")
    lora.scale_per_rank = _scales(64)
    x, tr = batch.x.clone(), batch.token_ranks
    kwargs = fn.prepare(x, lora, tr)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn(x, lora, tr, **kwargs)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = fn(x, lora, tr, **kwargs)
    for _ in range(2):
        x.copy_(torch.randn_like(x))
        graph.replay()
        torch.cuda.synchronize()
        expected = REGISTRY["b0_reference"](x, lora, tr)
        torch.testing.assert_close(y.float(), expected.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("impl", MODES)
def test_runs_on_current_stream(impl):
    """The kernels follow torch.cuda.stream(...) (upstream S-LoRA launches on the default stream)."""
    fn = REGISTRY[impl]
    w = Workload(64, 1, h_in=4096, h_out=4096, r_max=64, dist="uniform", dtype="fp16", seed=10)
    lora, batch = make_lora(w, "cuda"), make_batch(w, "cuda")
    x = batch.x.clone()
    x_new = torch.randn_like(x)
    kwargs = fn.prepare(x, lora, batch.token_ranks)
    expected = REGISTRY["b0_reference"](x_new, lora, batch.token_ranks)
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        # x only gets its new values after a delay on the side stream; a kernel launched on
        # another stream would run first and read the old x.
        torch.cuda._sleep(50_000_000)
        x.copy_(x_new)
        y = fn(x, lora, batch.token_ranks, **kwargs)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    torch.testing.assert_close(y.float(), expected.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("impl", MODES)
def test_unsupported_inputs_raise(impl):
    fn = REGISTRY[impl]
    for kwargs in (dict(h_in=256, h_out=192, r_max=32, dtype="fp16"),   # hidden sizes not compiled in
                   dict(h_in=896, h_out=896, r_max=128, dtype="fp16"),  # r_max above 64
                   dict(h_in=896, h_out=896, r_max=64, dtype="fp32")):  # fp32
        w = Workload(4, 1, dist="uniform", **kwargs)
        lora, batch = make_lora(w, "cuda"), make_batch(w, "cuda")
        with pytest.raises(ValueError):
            fn(batch.x, lora, batch.token_ranks)


def test_registered_only_on_request():
    """`import rankfuse` alone does not register B4 (see the module docstring)."""
    import subprocess
    import sys

    code = "import rankfuse; from rankfuse.interface import REGISTRY; print(any(n.startswith('b4_') for n in REGISTRY))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"
