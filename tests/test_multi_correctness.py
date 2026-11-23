"""Stage 2 (many adapters, mixed ranks): every multi-adapter implementation must match B0-multi.

Implementations come from rankfuse.multi.MULTI_REGISTRY (a separate registry; the Stage 1
REGISTRY and tests/test_correctness.py are untouched). B4 (S-LoRA) is registered for this module
only. Cases an implementation cannot run (B3/B4 in fp32, B4 at hidden sizes it was not compiled
for or with merged slices) are skipped with the reason.

    python -m pytest -q tests/test_multi_correctness.py
"""

import functools

import pytest
import torch

from rankfuse.debug import TOL
from rankfuse.multi import MULTI_REGISTRY, MultiWorkload, PagedWeights, load_all, make_batch, make_weights

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

if torch.cuda.is_available():
    load_all(include_slora=False)  # B4 is added by the fixture below
FAST = ["rf1_ma", "rf2_ma_splitk", "rf2_ma_segment", "rf2_ma_bucketed", "rf_ma_auto"]
BASELINES = ["b1_multi", "b3_multi", "b4_multi_copies", "b4_multi_shared"]
IMPLS = FAST + BASELINES
SHAPES = {"4096x4096": (4096, 4096), "q_o-896x896": (896, 896), "k_v-896x128": (896, 128),
          "gate_up-896x4864": (896, 4864), "down-4864x896": (4864, 896)}
TOKENS = {"decode-1": (1, 1), "decode-17": (17, 1), "decode-64": (64, 1), "decode-128": (128, 1),
          "prefill-3x50": (3, 50), "prefill-2x300": (2, 300)}
MIXES = {"uniform": ("uniform", 0.0), "skew_low": ("skew_low", 0.0), "all_max": ("all_max", 0.0),
         "some_zero": ("uniform", 0.3)}


@pytest.fixture(scope="module", autouse=True)
def b4_registered():
    from rankfuse.baselines import slora_bgmv_multi

    slora_bgmv_multi.enable()
    yield
    slora_bgmv_multi.disable()


@functools.lru_cache(maxsize=8)
def _weights(h_in, h_out, num_adapters, dtype, pages="dense", scaled=False):
    w = MultiWorkload(1, 1, num_adapters, h_in=h_in, h_out=h_out, dtype=dtype, pages=pages, seed=num_adapters)
    scales = {r: 16.0 / r for r in range(1, 65)} if scaled else None
    return make_weights(w, "cuda", scale_per_rank=scales)


def _skip_unsupported(impl: str, W: PagedWeights):
    reason = MULTI_REGISTRY[impl].unsupported(W)
    if reason:
        pytest.skip(reason)


def check(impl, x, W, ta, tr, dtype, out=None, add=False, **prepared):
    """Run impl (with its prepared metadata) and compare with B0-multi; return impl's output."""
    m = MULTI_REGISTRY[impl]
    kw = m.prepare(x, W, ta, tr) if (m.prepare and not prepared) else prepared
    base = None if out is None else out.clone()
    y = m(x, W, ta, tr, out=out, add=add, **kw)
    expected = MULTI_REGISTRY["b0_multi"](x, W, ta, tr, out=base, add=add)
    rtol, atol = TOL[dtype]
    err = (y.float() - expected).abs()
    bad = ~torch.isclose(y.float(), expected, rtol=rtol, atol=atol)
    assert not bad.any(), (
        f"{impl}: {int(bad.any(1).sum())}/{x.shape[0]} tokens wrong, max err {float(err.max()):.3g}; "
        f"first bad tokens {torch.nonzero(bad.any(1)).flatten()[:5].tolist()} "
        f"(adapters {ta[bad.any(1)][:5].tolist()}, ranks {tr[bad.any(1)][:5].tolist()})")
    return y


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("num_adapters", [1, 5, 64])
@pytest.mark.parametrize("tokens", list(TOKENS))
@pytest.mark.parametrize("mix", list(MIXES))
def test_matches_reference(impl, dtype, shape, num_adapters, tokens, mix):
    h_in, h_out = SHAPES[shape]
    W = _weights(h_in, h_out, num_adapters, dtype)
    _skip_unsupported(impl, W)
    requests, tpr = TOKENS[tokens]
    dist, zero = MIXES[mix]
    w = MultiWorkload(requests, tpr, num_adapters, popularity="zipf" if num_adapters == 64 else "uniform",
                      dist=dist, h_in=h_in, h_out=h_out, dtype=dtype, zero_frac=zero, seed=requests + num_adapters)
    x, ta, tr = make_batch(w)
    y = check(impl, x, W, ta, tr, dtype)
    assert (y[tr == 0] == 0).all(), "rank-0 rows must be exactly zero"


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("pages", ["contiguous", "shuffled"])
@pytest.mark.parametrize("tokens", ["decode-17", "decode-128", "prefill-2x300"])
def test_paged_weights(impl, pages, tokens):
    """Weights in per-layer page pools (contiguous or shuffled frames) give the dense result."""
    W = _weights(896, 896, 12, "fp16", pages=pages)
    _skip_unsupported(impl, W)
    requests, tpr = TOKENS[tokens]
    x, ta, tr = make_batch(MultiWorkload(requests, tpr, 12, h_in=896, h_out=896, seed=requests))
    check(impl, x, W, ta, tr, "fp16")


def test_page_table_round_trip():
    """The shuffled pools hold every page where the page table says, and the table is a permutation."""
    W = _weights(896, 128, 12, "fp16", pages="shuffled")
    A, B = W.gather_dense()
    assert torch.equal(A, W.A_dense) and torch.equal(B, W.B_dense)
    frames = W.page_table.flatten()
    assert frames.unique().numel() == frames.numel()
    assert not torch.equal(frames.cpu(), torch.arange(frames.numel(), dtype=torch.int32)), "not shuffled"


@pytest.mark.parametrize("impl", FAST + ["b1_multi", "b3_multi", "b4_multi_shared"])
def test_partially_resident_pages(impl):
    """Pages above an adapter's highest requested rank need not be resident (page_table -1), as with
    prefix paging; the pool also has spare frames. Kernels must never read a non-resident page."""
    N, h = 8, 896
    w = MultiWorkload(40, 1, N, h_in=h, h_out=h, seed=11)
    gen = torch.Generator().manual_seed(0)
    A = (torch.randn(N, 1, 64, h, generator=gen) / h**0.5).half().cuda()
    B = (torch.randn(N, 1, 64, h, generator=gen) / 8).half().cuda()
    x, ta, tr = make_batch(w)
    # each adapter keeps only the pages its tokens need
    need = torch.zeros(N, 4, dtype=torch.bool)
    for a, r in zip(ta.tolist(), tr.tolist()):
        need[a, : max(1, -(-r // 16))] = True
    W = PagedWeights.paged(A, B, order="shuffled", resident=need, extra_frames=7, seed=3)
    assert (W.page_table == -1).any()
    # B0 and the baselines read the dense tensors; zero the non-resident rows there too.
    for a in range(N):
        for k in range(4):
            if not need[a, k]:
                W.A_dense[a, :, 16 * k:16 * k + 16] = 0
                W.B_dense[a, :, 16 * k:16 * k + 16] = 0
    check(impl, x, W, ta, tr, "fp16")


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("tokens", ["decode-17", "prefill-2x300"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_add_mode(impl, tokens, dtype):
    """out += delta: rows of rank-0 tokens keep their value exactly."""
    W = _weights(896, 4864, 6, dtype)
    _skip_unsupported(impl, W)
    requests, tpr = TOKENS[tokens]
    x, ta, tr = make_batch(MultiWorkload(requests, tpr, 6, h_in=896, h_out=4864, dtype=dtype, zero_frac=0.3,
                                         seed=5))
    out = torch.randn(x.shape[0], 4864, device="cuda").to(x.dtype)
    before = out.clone()
    y = check(impl, x, W, ta, tr, dtype, out=out, add=True)
    assert y.data_ptr() == out.data_ptr()
    assert torch.equal(y[tr == 0], before[tr == 0]), "add mode must not touch rank-0 rows"


@pytest.mark.parametrize("impl", IMPLS)
def test_overwrites_out(impl):
    """add=False into a buffer holding garbage: every row is overwritten (rank-0 rows with 0)."""
    W = _weights(896, 896, 6, "fp16")
    _skip_unsupported(impl, W)
    x, ta, tr = make_batch(MultiWorkload(33, 1, 6, h_in=896, h_out=896, zero_frac=0.3, seed=8))
    out = torch.full((33, 896), 7.0, device="cuda", dtype=torch.float16)
    y = check(impl, x, W, ta, tr, "fp16", out=out, add=False)
    assert (y[tr == 0] == 0).all()


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("slices", [(896, 128, 128), (4864, 4864)], ids=["qkv", "gate_up"])
@pytest.mark.parametrize("tokens", ["decode-17", "decode-128", "prefill-2x300"])
@pytest.mark.parametrize("add", [False, True], ids=["write", "add"])
def test_merged_slices(impl, slices, tokens, add):
    """Several output slices sharing x in one call (fused q/k/v and gate/up), slice widths differ."""
    W = make_weights(MultiWorkload(1, 1, 5, h_in=896, seed=2), "cuda", slice_sizes=slices,
                     scale_per_rank={r: 8.0 / r for r in range(1, 65)})
    _skip_unsupported(impl, W)
    requests, tpr = TOKENS[tokens]
    x, ta, tr = make_batch(MultiWorkload(requests, tpr, 5, h_in=896, zero_frac=0.2, seed=9))
    out = torch.randn(x.shape[0], sum(slices), device="cuda").half() if add else None
    check(impl, x, W, ta, tr, "fp16", out=out, add=add)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("pages", ["dense", "shuffled"])
def test_per_adapter_scales(impl, pages):
    """A different per-rank scale table per adapter."""
    N = 7
    scales = [{r: (a + 1) * 4.0 / r for r in range(1, 65)} for a in range(N)]
    W = make_weights(MultiWorkload(1, 1, N, h_in=896, h_out=896, pages=pages, seed=4), "cuda", scale_per_rank=scales)
    _skip_unsupported(impl, W)
    x, ta, tr = make_batch(MultiWorkload(50, 1, N, h_in=896, h_out=896, seed=6))
    check(impl, x, W, ta, tr, "fp16")


@pytest.mark.parametrize("impl", IMPLS)
def test_odd_ranks_and_unassigned(impl):
    """Ranks outside the tiers (3, 12, 40, 33) are legal; adapter -1 means no LoRA whatever the rank."""
    W = _weights(4096, 4096, 5, "fp16")
    _skip_unsupported(impl, W)
    tr = torch.tensor([3, 12, 40, 0, 64, 33, 1, 7, 12, 5, 2], dtype=torch.int32, device="cuda")
    ta = torch.tensor([0, 4, 2, 1, 2, 3, -1, 0, 0, 4, 1], dtype=torch.int32, device="cuda")
    x = torch.randn(11, 4096, device="cuda").half()
    y = check(impl, x, W, ta, tr, "fp16")
    assert (y[(tr == 0) | (ta < 0)] == 0).all()


@pytest.mark.parametrize("impl", IMPLS)
def test_all_rank_zero(impl):
    W = _weights(896, 896, 6, "fp16")
    _skip_unsupported(impl, W)
    x = torch.randn(9, 896, device="cuda").half()
    ta = torch.arange(9, device="cuda", dtype=torch.int32) % 6
    tr = torch.zeros(9, dtype=torch.int32, device="cuda")
    y = check(impl, x, W, ta, tr, "fp16", out=torch.ones(9, 896, device="cuda").half())
    assert (y == 0).all()


@pytest.mark.parametrize("impl", ["rf1_ma", "rf2_ma_splitk", "rf2_ma_segment", "rf2_ma_bucketed", "b1_multi"])
def test_fp32(impl):
    W = _weights(512, 192, 5, "fp32")
    x, ta, tr = make_batch(MultiWorkload(30, 1, 5, h_in=512, h_out=192, dtype="fp32", seed=2))
    check(impl, x, W, ta, tr, "fp32")


@pytest.mark.parametrize("impl", FAST + ["b1_multi"])
@pytest.mark.parametrize("tokens", ["decode-17", "prefill-2x300"])
@pytest.mark.parametrize("pages", ["dense", "shuffled"])
def test_unaligned_sizes(impl, tokens, pages):
    """h_in = 200 and h_out = 100 (not multiples of 16): the kernels' no-alignment-hint path and
    partial tiles at the end of both reductions."""
    W = make_weights(MultiWorkload(1, 1, 5, h_in=200, h_out=100, pages=pages, seed=3), "cuda")
    assert not W.aligned()
    requests, tpr = TOKENS[tokens]
    x, ta, tr = make_batch(MultiWorkload(requests, tpr, 5, h_in=200, h_out=100, zero_frac=0.2, seed=4))
    check(impl, x, W, ta, tr, "fp16")


@pytest.mark.parametrize("impl", FAST)
def test_large_prefill_many_adapters(impl):
    """8192 tokens over 64 adapters (Zipf): long and short groups, tiled expand, 64-token tiles."""
    W = _weights(4096, 4096, 64, "fp16")
    x, ta, tr = make_batch(MultiWorkload(32, 256, 64, popularity="zipf", h_in=4096, h_out=4096, zero_frac=0.1,
                                         seed=12))
    check(impl, x, W, ta, tr, "fp16")
