"""GPU tests of the rank-skip serving launcher (rankfuse/integration/rank_skip.py).

RankSkipLauncher (rf2_index over a host-sorted order, and rf2_segment over the batch order) must
match RF1Launcher on mixed per-token ranks, eagerly and when replayed from a CUDA graph captured
at r_limit = r_max with DIFFERENT ranks than the replay (the ranks and order are refilled in
place, as the vLLM wrapper does every step)."""

import numpy as np
import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

TIERS = [0, 8, 16, 32, 64, 128, 256]
SHAPES = {  # name: (h_in, slice sizes)
    "qkv_7b": (3584, (3584, 512, 512)),
    "down_05b": (4864, (896,)),
}


def _weights(h_in, sizes, r_max, dtype, gen):
    n_max = max(sizes)
    A = (torch.randn(len(sizes), r_max, h_in, generator=gen) / h_in**0.5).to(dtype).cuda()
    B = (torch.randn(len(sizes), r_max, n_max, generator=gen) * 0.02).to(dtype).cuda()
    return A.contiguous(), B.contiguous()


def _ranks(T, gen, pattern):
    if pattern == "mixed":
        idx = torch.randint(0, len(TIERS), (T,), generator=gen)
        return torch.tensor([TIERS[i] for i in idx], dtype=torch.int32)
    return torch.full((T,), int(pattern), dtype=torch.int32)


def _order(ranks_cpu, max_tokens):
    from rankfuse.integration.rank_skip import fill_order

    host = np.arange(max_tokens, dtype=np.int32)
    fill_order(host, ranks_cpu.numpy(), ranks_cpu.numel(), 0)
    return torch.from_numpy(host).cuda()


@cuda
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("T", [1, 7, 64, 300])
@pytest.mark.parametrize("indexed", [True, False])
def test_eager_matches_rf1(shape, T, indexed):
    from rankfuse.integration.rank_skip import RankSkipLauncher
    from rankfuse.kernels.rankfuse_fused import RF1Launcher

    gen = torch.Generator().manual_seed(T)
    h_in, sizes = SHAPES[shape]
    A, B = _weights(h_in, sizes, 256, torch.bfloat16, gen)
    x = torch.randn(T, h_in, generator=gen).to(torch.bfloat16).cuda()
    ranks_cpu = _ranks(T, gen, "mixed")
    ranks = ranks_cpu.cuda()
    base = torch.randn(T, sum(sizes), generator=gen).to(torch.bfloat16).cuda()
    ref = base.clone()
    RF1Launcher(A, B, sizes, max_tokens=512)(x, ranks, ref, r_limit=int(ranks_cpu.max()), add=True)
    out = base.clone()
    RankSkipLauncher(A, B, sizes, max_tokens=512, indexed=indexed)(x, ranks, _order(ranks_cpu, 512), out,
                                                                   r_limit=256, add=True)
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)


@cuda
@pytest.mark.parametrize("indexed", [True, False])
def test_graph_replay_follows_ranks(indexed):
    """Capture at one rank pattern, refill ranks/order in place, replay: matches eager RF1."""
    from rankfuse.integration.rank_skip import RankSkipLauncher, fill_order
    from rankfuse.kernels.rankfuse_fused import RF1Launcher

    gen = torch.Generator().manual_seed(0)
    h_in, sizes = SHAPES["qkv_7b"]
    T, MAXT = 64, 512
    A, B = _weights(h_in, sizes, 256, torch.bfloat16, gen)
    x = torch.randn(T, h_in, generator=gen).to(torch.bfloat16).cuda()
    ranks = torch.zeros(MAXT, dtype=torch.int32, device="cuda")      # static buffers
    order = torch.arange(MAXT, dtype=torch.int32, device="cuda")
    host = np.arange(MAXT, dtype=np.int32)
    out = torch.zeros(T, sum(sizes), dtype=torch.bfloat16, device="cuda")
    launcher = RankSkipLauncher(A, B, sizes, max_tokens=MAXT, indexed=indexed)

    def fill(r):
        ranks[:T].copy_(r.cuda())
        fill_order(host, r.numpy(), T, T)
        order.copy_(torch.from_numpy(host).cuda())

    fill(_ranks(T, gen, "256"))
    launcher(x, ranks, order, out, r_limit=256, add=False)   # compile before capture
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        launcher(x, ranks, order, out, r_limit=256, add=False)
    for pattern in ("mixed", "8", "0", "mixed"):
        r = _ranks(T, gen, pattern)
        fill(r)
        g.replay()
        ref = torch.zeros_like(out)
        RF1Launcher(A, B, sizes, max_tokens=MAXT)(x, r.cuda(), ref, r_limit=max(int(r.max()), 1), add=False)
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
