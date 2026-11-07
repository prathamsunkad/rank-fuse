"""Serving features of the fused kernels (rankfuse_lora) and the graphed B1 baseline.

tests/test_correctness.py checks every registered implementation through the common
(x, lora, token_ranks) interface. These tests cover what that interface cannot express:
adding into an existing output, several output slices in one launch, output column views,
CUDA-graph capture and replay, a precomputed rank order, and repeated graph replays. RF1 is
tested on both of its paths: split-K (the default) and the single kernel (split_k=0).
Every check compares against B0 (fp32 reference) on the same inputs.
"""

import pytest
import torch

import rankfuse  # noqa: F401  (registers implementations)
from rankfuse.interface import REGISTRY, MatryoshkaLoRA

HAS_FUSED = torch.cuda.is_available() and "rf1_fused" in REGISTRY
pytestmark = pytest.mark.skipif(not HAS_FUSED, reason="requires the fused Triton kernels on CUDA")

if HAS_FUSED:
    from rankfuse.kernels.rankfuse_fused import rankfuse_lora

TOL = {torch.float32: 1e-4, torch.float16: 2e-2}
KERNELS = pytest.mark.parametrize(
    "kernel",
    [
        dict(rank_sort=False),
        dict(rank_sort=False, split_k=0),
        dict(rank_sort=True),
        dict(rank_skip=True),
        dict(rank_skip=True, full_width=True),
        dict(rank_skip=True, rank_sort=True),
        dict(rank_skip=True, rank_sort=True, full_width=True),
        dict(rank_probe=1),
        dict(rank_probe=2),
    ],
    ids=["rf1", "rf1_single", "rf2_fused", "rf2_segment", "rf2_segment_fw", "rf2_index", "rf2_index_fw", "rf2_probe_read",
         "rf2_probe_branch"],
)
DTYPES = pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])


def _inputs(T, h_in, sizes, r_max, dtype, seed=0, ranks=None):
    """x [T, h_in], stacked A_t [S, r_max, h_in], B [S, r_max, max(sizes)], token_ranks [T].

    B columns past each slice's width hold large values, so a kernel that reads or writes past
    a slice boundary produces a visibly wrong result.
    """
    gen = torch.Generator().manual_seed(seed)
    n_max = max(sizes)
    x = torch.randn(T, h_in, generator=gen)
    A_t = torch.randn(len(sizes), r_max, h_in, generator=gen) / h_in**0.5
    B = torch.randn(len(sizes), r_max, n_max, generator=gen) / r_max**0.5
    for s, size in enumerate(sizes):
        B[s, :, size:] = 1e3
    if ranks is None:
        tiers = [r_max // 8, r_max // 4, r_max // 2, r_max]
        ranks = [tiers[i] for i in torch.randint(0, 4, (T,), generator=gen).tolist()]
    token_ranks = torch.tensor(ranks, dtype=torch.int32)
    to = dict(device="cuda", dtype=dtype)
    return x.to(**to), A_t.to(**to), B.to(**to), token_ranks.cuda()


def _reference(x, A_t, B, token_ranks, sizes, scale_per_rank=None):
    """B0 per slice, concatenated along columns; fp32."""
    parts = []
    for s, size in enumerate(sizes):
        lora = MatryoshkaLoRA(A_t=A_t[s], B=B[s, :, :size].contiguous(), scale_per_rank=scale_per_rank)
        parts.append(REGISTRY["b0_reference"](x, lora, token_ranks))
    return torch.cat(parts, dim=1)


def _assert_close(got, expected, dtype):
    tol = TOL[dtype]
    torch.testing.assert_close(got.float(), expected.float(), rtol=tol, atol=tol)


@KERNELS
@DTYPES
def test_add_into_output(kernel, dtype):
    """add=True: out += delta in the same pass, as when adding to the base projection."""
    x, A_t, B, ranks = _inputs(37, 192, [160], 32, dtype, seed=1)
    base = torch.randn(37, 160, device="cuda", dtype=dtype)
    out = base.clone()
    rankfuse_lora(x, A_t, B, ranks, out=out, add=True, **kernel)
    _assert_close(out, base.float() + _reference(x, A_t, B, ranks, [160]), dtype)


@KERNELS
@DTYPES
@pytest.mark.parametrize("add", [False, True], ids=["write", "add"])
def test_slices(kernel, dtype, add):
    """Fused q/k/v-style projection: three adapters share x and write consecutive columns."""
    sizes = [128, 32, 32]
    x, A_t, B, ranks = _inputs(53, 160, sizes, 32, dtype, seed=2)
    expected = _reference(x, A_t, B, ranks, sizes)
    if add:
        base = torch.randn(53, sum(sizes), device="cuda", dtype=dtype)
        out = base.clone()
        expected = expected + base.float()
    else:
        out = torch.full((53, sum(sizes)), float("nan"), device="cuda", dtype=dtype)
    rankfuse_lora(x, A_t, B, ranks, out=out, add=add, slice_sizes=sizes, **kernel)
    assert torch.isfinite(out).all(), "some output columns were never written"
    _assert_close(out, expected, dtype)


@KERNELS
def test_output_column_view(kernel):
    """out may be a column view of a larger buffer; columns outside it stay untouched."""
    x, A_t, B, ranks = _inputs(21, 96, [64, 16], 16, torch.float32, seed=3)
    buffer = torch.randn(21, 200, device="cuda")
    before = buffer.clone()
    view = buffer[:, 50:130]
    rankfuse_lora(x, A_t, B, ranks, out=view, add=True, slice_sizes=[64, 16], **kernel)
    _assert_close(view, before[:, 50:130] + _reference(x, A_t, B, ranks, [64, 16]), torch.float32)
    assert torch.equal(buffer[:, :50], before[:, :50]) and torch.equal(buffer[:, 130:], before[:, 130:])


@KERNELS
def test_per_rank_scale_with_slices(kernel):
    sizes = [96, 48]
    x, A_t, B, ranks = _inputs(30, 128, sizes, 32, torch.float32, seed=4)
    scale_per_rank = {r: 16.0 / r for r in (4, 8, 16, 32)}
    table = torch.zeros(33, device="cuda")
    for r, s in scale_per_rank.items():
        table[r] = s
    out = rankfuse_lora(x, A_t, B, ranks, slice_sizes=sizes, scale_table=table, **kernel)
    _assert_close(out, _reference(x, A_t, B, ranks, sizes, scale_per_rank), torch.float32)


@KERNELS
@pytest.mark.parametrize("add", [False, True], ids=["write", "add"])
def test_all_zero_blocks(kernel, add):
    """Whole token blocks of rank 0: their rows are zero (write) or unchanged (add)."""
    ranks = [0] * 20 + [4, 8, 16, 32] * 5
    x, A_t, B, token_ranks = _inputs(40, 64, [48], 32, torch.float32, seed=5, ranks=ranks)
    base = torch.randn(40, 48, device="cuda")
    out = base.clone() if add else torch.full((40, 48), float("nan"), device="cuda")
    rankfuse_lora(x, A_t, B, token_ranks, out=out, add=add, **kernel)
    expected = _reference(x, A_t, B, token_ranks, [48]) + (base if add else 0)
    _assert_close(out, expected, torch.float32)


@KERNELS
def test_cuda_graph_capture_and_replay(kernel):
    """With r_limit passed, the call has no host sync: capture once, then replay on new
    values of x and token_ranks written into the same tensors."""
    sizes = [96, 32]
    x, A_t, B, ranks = _inputs(29, 128, sizes, 32, torch.float32, seed=6)
    out = torch.empty(29, sum(sizes), device="cuda")

    def run():
        rankfuse_lora(x, A_t, B, ranks, out=out, slice_sizes=sizes, r_limit=32, **kernel)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()  # compiles the kernels and builds the cached slice tables before capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    x2, _, _, ranks2 = _inputs(29, 128, sizes, 32, torch.float32, seed=7)
    x.copy_(x2)
    ranks.copy_(ranks2)
    graph.replay()
    torch.cuda.synchronize()
    _assert_close(out, _reference(x, A_t, B, ranks, sizes), torch.float32)


def test_rf2_precomputed_order():
    """A scheduler-prepared rank order gives the same result as the internal sort."""
    x, A_t, B, ranks = _inputs(70, 128, [96], 64, torch.float16, seed=8)
    order = torch.argsort(ranks, stable=True)
    with_order = rankfuse_lora(x, A_t, B, ranks, rank_sort=True, order=order)
    without = rankfuse_lora(x, A_t, B, ranks, rank_sort=True)
    torch.testing.assert_close(with_order, without)
    with_order = rankfuse_lora(x, A_t, B, ranks, rank_sort=True, rank_skip=True, order=order)
    without = rankfuse_lora(x, A_t, B, ranks, rank_sort=True, rank_skip=True)
    torch.testing.assert_close(with_order, without)


def test_invalid_arguments():
    x, A_t, B, ranks = _inputs(8, 64, [32, 16], 16, torch.float32, seed=9)
    with pytest.raises(ValueError):
        rankfuse_lora(x, A_t, B, ranks, add=True, slice_sizes=[32, 16])  # add needs out
    with pytest.raises(ValueError):
        rankfuse_lora(x, A_t, B, ranks, slice_sizes=[32])  # one width for two slices
    with pytest.raises(ValueError):
        rankfuse_lora(x, A_t, B, ranks, slice_sizes=[32, 40])  # wider than B
    with pytest.raises(ValueError):
        rankfuse_lora(x, A_t, B, ranks, slice_sizes=[32, 16], out=torch.empty(8, 40, device="cuda"))
    with pytest.raises(ValueError):
        rankfuse_lora(x, A_t, B, ranks, split_k=-1)
    with pytest.raises(ValueError):
        rankfuse_lora(x, A_t, B, ranks, rank_skip=True, split_k=0)  # skipping runs on split-K
    with pytest.raises(ValueError):
        rankfuse_lora(x, A_t, B, ranks, full_width=True)  # the control for rank_skip
    with pytest.raises(ValueError):
        rankfuse_lora(x, A_t, B, ranks, rank_probe=3)
    with pytest.raises(ValueError):
        rankfuse_lora(x, A_t, B, ranks, rank_probe=1, rank_skip=True)


@pytest.mark.parametrize("split_k", [1, 2, 3, 7, 64], ids=lambda s: f"S{s}")
@pytest.mark.parametrize("T", [5, 300], ids=["decode-5", "prefill-300"])
def test_rf1_split_counts(split_k, T):
    """RF1's split-K path with explicit slice counts (64 is more than h_in has chunks), at a
    decode size and above 128 tokens (larger token blocks); fp16, two slices, add=True."""
    sizes = [192, 64]
    x, A_t, B, ranks = _inputs(T, 520, sizes, 64, torch.float16, seed=11)
    base = torch.randn(T, sum(sizes), device="cuda", dtype=torch.float16)
    out = base.clone()
    rankfuse_lora(x, A_t, B, ranks, out=out, add=True, slice_sizes=sizes, split_k=split_k)
    _assert_close(out, base.float() + _reference(x, A_t, B, ranks, sizes), torch.float16)


@pytest.mark.parametrize("T, r_max", [(64, 128), (300, 128), (300, 256)], ids=["decode-r128", "prefill-r128", "prefill-r256"])
def test_rf1_paths_agree(T, r_max):
    """RF1's split-K path and its single kernel give the same delta (fp32). Above 128 tokens
    the split-K expand runs one tile per program; r_max 256 in fp32 narrows its tiles."""
    x, A_t, B, ranks = _inputs(T, 384, [520], r_max, torch.float32, seed=12)
    splitk = rankfuse_lora(x, A_t, B, ranks)
    single = rankfuse_lora(x, A_t, B, ranks, split_k=0)
    torch.testing.assert_close(splitk, single, rtol=1e-4, atol=1e-4)
    _assert_close(splitk, _reference(x, A_t, B, ranks, [520]), torch.float32)


def test_b1_graphed_repeated_replays():
    """b1_padded_graphed replays the same graph on later calls; results must stay correct
    after unrelated allocations reuse freed memory (the bug fixed in b2_grouped_graphed)."""
    from rankfuse.workloads import Workload, make_batch, make_lora

    w = Workload(17, 1, h_in=128, h_out=96, r_max=32, dtype="fp32", seed=10)
    lora, batch = make_lora(w, "cuda"), make_batch(w, "cuda")
    expected = REGISTRY["b0_reference"](batch.x, lora, batch.token_ranks)
    impl = REGISTRY["b1_padded_graphed"]
    first = impl(batch.x, lora, batch.token_ranks).clone()
    junk = [torch.randint(0, 10**6, (256,), device="cuda") for _ in range(64)]  # noqa: F841
    second = impl(batch.x, lora, batch.token_ranks)
    torch.cuda.synchronize()
    _assert_close(first, expected, torch.float32)
    _assert_close(second, expected, torch.float32)


# RF2 variants (rankfuse_rf2_*.py). Their registered wrappers take one adapter; these tests cover
# add mode, an output column view and graph capture with per-step metadata built before capture.
BUCKETED = ("rf2_bucketed", "rf2_bucketed_tuned", "rf2_bucketed_seq")
RF2_VARIANTS = pytest.mark.parametrize(
    "name", ["rf2_segment", "rf2_segment_fw", "rf2_index", "rf2_index_fw", *BUCKETED, "rf2_ranges", "rf2_ranges_seq"]
)


def _rf2_lora(T, h_in, n, r_max, dtype, seed, ranks=None):
    x, A_t, B, token_ranks = _inputs(T, h_in, [n], r_max, dtype, seed=seed, ranks=ranks)
    return x, MatryoshkaLoRA(A_t=A_t[0], B=B[0].contiguous()), token_ranks


def _rf2_metadata(name, x, token_ranks):
    """What a serving engine would build once per step, outside the graph."""
    r_limit = int(token_ranks.max())
    if name.startswith("rf2_segment"):
        return dict(r_limit=r_limit)
    if name.startswith("rf2_index"):
        return dict(r_limit=r_limit, order=torch.argsort(token_ranks, stable=True).to(torch.int32))
    if name.startswith("rf2_ranges"):
        from rankfuse.kernels.rankfuse_rf2_ranges import RangePlan

        return dict(plan=RangePlan(token_ranks))
    from rankfuse.kernels.rankfuse_rf2_bucketed import Plan

    return dict(plan=Plan(token_ranks, x.device))


@RF2_VARIANTS
@DTYPES
@pytest.mark.parametrize("T", [37, 300], ids=["decode-37", "prefill-300"])
def test_rf2_add_into_column_view(name, dtype, T):
    """add=True into a column view of a larger buffer, with rank-0 rows; columns outside the
    view and rank-0 rows inside it stay unchanged."""
    ranks = [[0, 4, 8, 16, 32, 64][i % 6] for i in range(T)]
    x, lora, token_ranks = _rf2_lora(T, 192, 160, 64, dtype, seed=13, ranks=ranks)
    buffer = torch.randn(T, 260, device="cuda", dtype=dtype)
    before = buffer.clone()
    REGISTRY[name](x, lora, token_ranks, out=buffer[:, 40:200], add=True)
    expected = before[:, 40:200].float() + REGISTRY["b0_reference"](x, lora, token_ranks).float()
    _assert_close(buffer[:, 40:200], expected, dtype)
    assert torch.equal(buffer[:, :40], before[:, :40]) and torch.equal(buffer[:, 200:], before[:, 200:])


@RF2_VARIANTS
@pytest.mark.parametrize("T", [29, 300], ids=["decode-29", "prefill-300"])
def test_rf2_cuda_graph_capture_and_replay(name, T):
    """Metadata built before capture, no host sync inside: replay on new x. rf2_segment
    is also replayed on new ranks; rf2_bucketed's plan and rf2_index's order fix the ranks, so they
    are replayed on the same ones."""
    x, lora, token_ranks = _rf2_lora(T, 128, 96, 32, torch.float32, seed=14)
    meta = _rf2_metadata(name, x, token_ranks)
    out = torch.empty(T, 96, device="cuda")

    def run():
        REGISTRY[name](x, lora, token_ranks, out=out, **meta)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    x2, _, ranks2 = _rf2_lora(T, 128, 96, 32, torch.float32, seed=15)
    x.copy_(x2)
    if name.startswith("rf2_segment"):
        token_ranks.copy_(ranks2)
    graph.replay()
    torch.cuda.synchronize()
    _assert_close(out, REGISTRY["b0_reference"](x, lora, token_ranks), torch.float32)


def test_rf2_bucketed_sequential_matches_concurrent():
    x, lora, token_ranks = _rf2_lora(300, 256, 192, 64, torch.float16, seed=16)
    impl = REGISTRY["rf2_bucketed"]
    torch.testing.assert_close(impl(x, lora, token_ranks, concurrent=False), impl(x, lora, token_ranks))


def test_rf2_bucketed_slices():
    """Fused q/k/v-style projection on the bucketed launch path, above and below 128 tokens."""
    from rankfuse.kernels.rankfuse_rf2_bucketed import Plan, rf2_bucketed

    sizes = [128, 32, 32]
    for T, tuned in [(53, False), (300, False), (300, True)]:
        x, A_t, B, ranks = _inputs(T, 160, sizes, 32, torch.float16, seed=17)
        out = rf2_bucketed(x, A_t, B, ranks, Plan(ranks, x.device), tuned=tuned, slice_sizes=sizes)
        _assert_close(out, _reference(x, A_t, B, ranks, sizes), torch.float16)


@pytest.mark.parametrize("T", [29, 300], ids=["decode-29", "prefill-300"])
@pytest.mark.parametrize("r_max", [64, 128, 256], ids=["r64", "r128", "r256"])
def test_full_width_control_matches(T, r_max):
    """The control and rank skipping use different widths but must give the same delta."""
    x, A_t, B, ranks = _inputs(T, 384, [320], r_max, torch.float32, seed=18)
    skipped = rankfuse_lora(x, A_t, B, ranks, rank_skip=True)
    full = rankfuse_lora(x, A_t, B, ranks, rank_skip=True, full_width=True)
    torch.testing.assert_close(skipped, full, rtol=1e-4, atol=1e-4)
    _assert_close(skipped, _reference(x, A_t, B, ranks, [320]), torch.float32)


def test_sorted_batch_preserves_requests():
    """A physically rank-ordered batch gives each token the same delta as in its original order."""
    from rankfuse.workloads import Workload, make_batch, make_lora

    w = Workload(8, 1, h_in=128, h_out=96, r_max=32, dtype="fp32", lengths="serving_2", request_order="sorted", seed=3)
    lora, batch = make_lora(w, "cuda"), make_batch(w, "cuda")
    assert batch.token_ranks.tolist() == sorted(batch.token_ranks.tolist())
    got = REGISTRY["rf2_segment"](batch.x, lora, batch.token_ranks)
    _assert_close(got, REGISTRY["b0_reference"](batch.x, lora, batch.token_ranks), torch.float32)


# rf2_ranges: per-bucket launches over the contiguous rows of a rank-sorted batch.
def test_range_plan_runs():
    """Sorted ranks give one run per bucket (heaviest first) and rank-0 rows as zero runs;
    unsorted ranks fall back to one run at the batch's bucket."""
    from rankfuse.kernels.rankfuse_rf2_ranges import RangePlan

    plan = RangePlan([0, 0, 3, 16, 17, 30, 200])
    assert plan.sorted and plan.zero_runs == [(0, 2)]
    assert sorted(plan.runs) == [(16, 2, 4, 16), (32, 4, 6, 30), (256, 6, 7, 200)]
    assert plan.runs[0][0] == 256  # 1 row x 256 is heavier than 2 rows x 32 or x 16
    unsorted = RangePlan([4, 200, 4, 0])
    assert not unsorted.sorted and unsorted.zero_runs == [] and unsorted.runs == [(256, 0, 4, 200)]
    assert RangePlan([0, 0]).runs == [] and RangePlan([]).runs == []


@pytest.mark.parametrize("name", ["rf2_ranges", "rf2_ranges_seq"])
@pytest.mark.parametrize("add", [False, True], ids=["write", "add"])
@pytest.mark.parametrize("T", [37, 300, 1100], ids=["decode-37", "prefill-300", "prefill-1100"])
def test_rf2_ranges_sorted(name, add, T):
    """Sorted batch spanning every bucket, rank-0 rows first, r_max 200 (not a power of two, so the
    256 bucket must stop at row 200): matches B0 in write and add mode."""
    ranks = sorted([0, 1, 9, 16, 20, 40, 64, 100, 128, 150, 200][i % 11] for i in range(T))
    x, lora, token_ranks = _rf2_lora(T, 256, 192, 200, torch.float16, seed=21, ranks=ranks)
    out = torch.randn(T, 192, device="cuda", dtype=torch.float16)
    before = out.clone()
    REGISTRY[name](x, lora, token_ranks, out=out, add=add)
    expected = REGISTRY["b0_reference"](x, lora, token_ranks).float()
    if add:
        expected = expected + before.float()
    _assert_close(out, expected, torch.float16)


def test_rf2_ranges_sorted_graph_capture():
    """Concurrent per-bucket launches (forked streams) captured in one graph and replayed on new x."""
    from rankfuse.kernels.rankfuse_rf2_ranges import RangePlan

    T = 300
    ranks = sorted([0, 4, 20, 50, 100, 256][i % 6] for i in range(T))
    x, lora, token_ranks = _rf2_lora(T, 256, 128, 256, torch.float32, seed=22, ranks=ranks)
    plan = RangePlan(token_ranks)
    assert plan.sorted and len(plan.runs) == 5
    out = torch.empty(T, 128, device="cuda")

    def run():
        REGISTRY["rf2_ranges"](x, lora, token_ranks, out=out, plan=plan)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    x.copy_(_rf2_lora(T, 256, 128, 256, torch.float32, seed=23, ranks=ranks)[0])
    graph.replay()
    torch.cuda.synchronize()
    _assert_close(out, REGISTRY["b0_reference"](x, lora, token_ranks), torch.float32)
