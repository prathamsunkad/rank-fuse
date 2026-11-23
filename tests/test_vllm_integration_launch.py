"""The serving launch path of rf1_fused: cached plans / direct Triton launches and RF1Launcher.

The fast path must not change results: every case compares bitwise against the original
launch (Triton's own launcher through rankfuse_lora, forced by clearing the caches) and, as a
sanity check, against the fp32 reference.
"""

import pytest
import torch

from rankfuse.kernels import rankfuse_fused as rf

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

LAYOUTS = [(896, (896, 128, 128)), (896, (4864, 4864)), (4864, (896,)), (64, (40, 24))]


def make(h_in, sizes, t, r_max=64, dtype=torch.float16, seed=0, ranks_dtype=torch.int32):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n = len(sizes)
    A = (torch.randn(n, r_max, h_in, generator=g, device="cuda") * 0.05).to(dtype)
    B = (torch.randn(n, r_max, max(sizes), generator=g, device="cuda") * 0.05).to(dtype)
    x = torch.randn(t, h_in, generator=g, device="cuda").to(dtype)
    ranks = torch.randint(0, r_max + 1, (t,), generator=g, device="cuda").to(ranks_dtype)
    y = torch.randn(t, sum(sizes), generator=g, device="cuda").to(dtype)
    return x, A, B, ranks, y


def reference(x, A, B, ranks, sizes, scale_table=None):
    outs = []
    k = torch.arange(A.shape[1], device=x.device)
    for s, w in enumerate(sizes):
        v = x.float() @ A[s].float().T
        v = v * (k[None, :] < ranks[:, None].long())
        if scale_table is not None:
            v = v * scale_table[ranks.long()][:, None]
        outs.append(v @ B[s, :, :w].float())
    return torch.cat(outs, 1)


def slow_rankfuse_lora(*args, **kwargs):
    """rankfuse_lora through Triton's own launcher (the kernel caches emptied first)."""
    for plan in rf._SPLITK_PLANS.values():
        plan.kernels.clear()
    return rf.rankfuse_lora(*args, **kwargs)


@pytest.mark.parametrize("h_in,sizes", LAYOUTS)
@pytest.mark.parametrize("t", [1, 7, 32, 129, 1000])
@pytest.mark.parametrize("add", [True, False])
def test_fast_paths_bitwise_equal_to_triton_launch(h_in, sizes, t, add):
    x, A, B, ranks, y0 = make(h_in, sizes, t)
    r = int(ranks.max())
    want = y0.clone()
    slow_rankfuse_lora(x, A, B, ranks, out=want, add=add, slice_sizes=sizes, r_limit=r)
    got = y0.clone()  # second call of the same shape: cached plan, direct launch
    rf.rankfuse_lora(x, A, B, ranks, out=got, add=add, slice_sizes=sizes, r_limit=r)
    rf.rankfuse_lora(x, A, B, ranks, out=(got2 := y0.clone()), add=add, slice_sizes=sizes, r_limit=r)
    launcher = rf.RF1Launcher(A, B, sizes, max_tokens=1024)
    fast = y0.clone()
    launcher(x, ranks, fast, r_limit=r, add=add)
    launcher(x, ranks, (fast2 := y0.clone()), r_limit=r, add=add)
    for out in (got, got2, fast, fast2):
        assert torch.equal(out, want)
    ref = reference(x, A, B, ranks, sizes) + (y0.float() if add else 0)
    assert (want.float() - ref).abs().max().item() < 2e-2 * max(1.0, ref.abs().max().item())


@pytest.mark.parametrize("r_limit", [1, 3, 16, 17, 33, 64])
def test_r_limit_specializations(r_limit):
    # r_limit == 1 and multiples of 16 compile different kernels in Triton: the cache must keep them apart
    x, A, B, ranks, y0 = make(896, (896, 128, 128), 24)
    ranks = ranks.clamp(max=r_limit)
    launcher = rf.RF1Launcher(A, B, (896, 128, 128), max_tokens=64)
    for r in (r_limit, 64, r_limit):
        want = y0.clone()
        slow_rankfuse_lora(x, A, B, ranks, out=want, add=True, slice_sizes=(896, 128, 128), r_limit=r)
        got = y0.clone()
        launcher(x, ranks, got, r_limit=r)
        assert torch.equal(got, want)


def test_unaligned_views_and_int64_ranks():
    # out as a column view starting at an odd element (address not 16-byte aligned), x as a
    # contiguous slice at an unaligned offset, int64 ranks: different Triton specializations
    x_big, A, B, ranks, _ = make(896, (896,), 41, ranks_dtype=torch.int64)
    x = x_big.view(-1)[1:1 + 40 * 896].view(40, 896)  # 2 bytes past an aligned address
    ranks = ranks[1:]  # 8 bytes past an aligned address
    big = torch.randn(40, 1000, device="cuda", dtype=torch.float16)
    for out_view in (big[:, 1:897], big[:, 64:960]):
        y0 = out_view.clone()
        want = y0.clone()
        slow_rankfuse_lora(x, A, B, ranks, out=want, add=True, r_limit=64)
        rf.RF1Launcher(A, B, max_tokens=64)(x, ranks, out_view, r_limit=64)
        assert torch.equal(out_view, want)
        rf.RF1Launcher(A, B, max_tokens=64)(x, ranks.to(torch.int32), (o2 := y0.clone()), r_limit=64)
        assert torch.equal(o2, want)


def test_scale_table_and_bf16_fp32():
    for dtype in (torch.bfloat16, torch.float32):
        x, A, B, ranks, y0 = make(896, (896, 128, 128), 33, dtype=dtype)
        table = torch.rand(65, device="cuda")
        want = y0.clone()
        slow_rankfuse_lora(x, A, B, ranks, out=want, add=True, slice_sizes=(896, 128, 128), r_limit=64, scale_table=table)
        got = y0.clone()
        rf.RF1Launcher(A, B, (896, 128, 128), max_tokens=64, scale_table=table)(x, ranks, got, r_limit=64)
        assert torch.equal(got, want)


def test_launcher_reads_only_first_t_ranks_and_r_limit_zero():
    x, A, B, ranks, y0 = make(896, (896,), 9)
    buf = torch.full((4096,), 64, dtype=torch.int32, device="cuda")  # a static per-step buffer
    buf[:9] = ranks
    launcher = rf.RF1Launcher(A, B, max_tokens=4096)
    want = y0.clone()
    slow_rankfuse_lora(x, A, B, ranks, out=want, add=True, r_limit=64)
    got = y0.clone()
    launcher(x, buf, got, r_limit=64)
    assert torch.equal(got, want)
    # r_limit 0: nothing to add; with add=False the delta (all zero) is written
    same = y0.clone()
    launcher(x, torch.zeros_like(buf), same, r_limit=0)
    assert torch.equal(same, y0)
    zero = y0.clone()
    launcher(x, torch.zeros_like(buf), zero, r_limit=0, add=False)
    assert torch.count_nonzero(zero) == 0
    with pytest.raises(ValueError):
        launcher(torch.empty(5000, 896, device="cuda", dtype=torch.float16), buf, torch.empty(5000, 896, device="cuda", dtype=torch.float16), r_limit=8)


def test_launcher_cuda_graph_capture_and_replay():
    x, A, B, ranks, y0 = make(896, (4864, 4864), 16)
    launcher = rf.RF1Launcher(A, B, (4864, 4864), max_tokens=64)
    y = y0.clone()
    launcher(x, ranks, y, r_limit=64)  # warm-up (plan + compiled kernels)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s), torch.cuda.graph(g):
        launcher(x, ranks, y, r_limit=64)
    for seed in (1, 2):
        new_ranks = torch.randint(0, 65, (16,), device="cuda", dtype=torch.int32)
        ranks.copy_(new_ranks)
        y.copy_(y0)
        g.replay()
        want = y0.clone()
        slow_rankfuse_lora(x, A, B, ranks, out=want, add=True, slice_sizes=(4864, 4864), r_limit=64)
        torch.cuda.synchronize()
        assert torch.equal(y, want)


def test_workspace_reserved_for_every_token_count():
    A = torch.zeros(3, 64, 896, device="cuda", dtype=torch.float16)
    B = torch.zeros(3, 64, 896, device="cuda", dtype=torch.float16)
    launcher = rf.RF1Launcher(A, B, (896, 128, 128), max_tokens=3000)
    for t in (1, 128, 129, 1000, 2048, 3000):
        plan = rf._SplitKPlan(t, 896, 3, 64, 896, (896, 128, 128), 64, 2, True, False, 0, 0, None)
        assert plan.ws_numel <= launcher.ws.numel()


def test_launcher_rejects_bad_setup():
    A = torch.zeros(2, 64, 896, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError):
        rf.RF1Launcher(A, torch.zeros(2, 32, 896, device="cuda", dtype=torch.float16), max_tokens=8)
    with pytest.raises(ValueError):
        rf.RF1Launcher(A, torch.zeros(2, 64, 896, device="cuda", dtype=torch.bfloat16), max_tokens=8)
    with pytest.raises(ValueError):
        rf.RF1Launcher(A, torch.zeros(2, 64, 896, device="cuda", dtype=torch.float16), (896, 1000), max_tokens=8)
