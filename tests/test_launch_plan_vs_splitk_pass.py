"""The cached launch path of the split-K kernels (_SplitKPlan / _launch / RF1Launcher) against splitk_pass.

rankfuse_fused._splitk launches Tony's two shared kernels through a cached plan and, from the
second call of a shape on, directly through the compiled kernels. `reference_splitk` below is the
body _splitk had on origin/main (8264f8f): _config at R_PAD, _splitk_shrink_tiles, _split_count,
then splitk_pass through Triton's own launcher. Every case must match it bitwise, for RF1 and every
RF2 option on this path (rank_skip, full_width, rank_probe, index), with a tuned table in force,
and after module settings change at run time (bench/rf1_ablation.py, bench/tune_blocks.py).
"""

import pytest
import torch

from rankfuse.kernels import rankfuse_fused as rf

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def reference_splitk(x, A_t, B, token_ranks, out, add, sizes, r_limit, r_pad, scale_table, split_k, rank_skip, force_rank,
                     rank_probe, index=None):
    num_tokens, h_in = x.shape
    n_slices = A_t.shape[0]
    device_index = x.device.index or 0
    cfg = rf._config(r_pad, num_tokens, device_index)
    block_m = cfg["bm"]
    m_tiles = (num_tokens + block_m - 1) // block_m
    bh, stages = rf._splitk_shrink_tiles(block_m, r_pad, x.element_size(), device_index, cfg["bh"], cfg["stages"])
    tiled = cfg["tiled"]
    split_k, k_per = rf._split_count(h_in, tiled, m_tiles, n_slices, rf._num_sms(device_index), bh, split_k)
    ws = torch.empty((n_slices, split_k, num_tokens, r_pad), dtype=torch.float32, device=x.device)
    return rf.splitk_pass(x, A_t, B, token_ranks, out, add, sizes, scale_table, ws, r_limit, r_pad, split_k, k_per, block_m,
                          bh, stages, index=index, rank_skip=rank_skip, rank_probe=rank_probe, force_rank=force_rank,
                          block_n=cfg["bn"], shrink_warps=cfg["shrink_warps"], expand_warps=cfg["expand_warps"], tiled=tiled)


def via_reference(monkeypatch, fn):
    """Run fn with rankfuse_lora routed through reference_splitk (the origin/main code path)."""
    with monkeypatch.context() as m:
        m.setattr(rf, "_splitk", reference_splitk)
        return fn()


def make(h_in, sizes, t, r_max=64, dtype=torch.float16, seed=0, sort=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n = len(sizes)
    A = (torch.randn(n, r_max, h_in, generator=g, device="cuda") * 0.05).to(dtype)
    B = (torch.randn(n, r_max, max(sizes), generator=g, device="cuda") * 0.05).to(dtype)
    x = torch.randn(t, h_in, generator=g, device="cuda").to(dtype)
    ranks = torch.randint(0, r_max + 1, (t,), generator=g, device="cuda").to(torch.int32)
    ranks[torch.rand(t, generator=g, device="cuda") < 0.3] = 0
    if sort:
        ranks = ranks.sort().values.contiguous()
    y = torch.randn(t, sum(sizes), generator=g, device="cuda").to(dtype)
    return x, A, B, ranks, y


VARIANTS = {
    "rf1": dict(),
    "rf2_segment": dict(rank_skip=True),
    "rf2_segment_fw": dict(rank_skip=True, full_width=True),
    "probe_read": dict(rank_probe=1),
    "probe_branch": dict(rank_probe=2),
    "rf2_index": dict(rank_skip=True, rank_sort=True),
    "split_k_3": dict(split_k=3),
}
LAYOUTS = [(4096, (4096,)), (896, (896, 128, 128)), (896, (4864, 4864)), (3584, (512,)), (64, (40, 24))]


def check(monkeypatch, x, A, B, ranks, y0, sizes, add, kw, scale_table=None, r_limit=None):
    call = dict(add=add, slice_sizes=sizes, r_limit=r_limit, scale_table=scale_table, **kw)
    want = via_reference(monkeypatch, lambda: rf.rankfuse_lora(x, A, B, ranks, out=y0.clone(), **call))
    for _ in range(3):  # first call: Triton's launcher; later calls: the cached compiled kernels
        got = rf.rankfuse_lora(x, A, B, ranks, out=y0.clone(), **call)
        assert torch.equal(got, want)
    return want


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("h_in,sizes", LAYOUTS)
@pytest.mark.parametrize("t", [1, 5, 64, 129, 700, 2048])
@pytest.mark.parametrize("add", [True, False])
def test_bitwise_equal_to_splitk_pass(monkeypatch, variant, h_in, sizes, t, add):
    x, A, B, ranks, y0 = make(h_in, sizes, t, sort=variant.startswith("rf2_segment"))
    check(monkeypatch, x, A, B, ranks, y0, sizes, add, VARIANTS[variant])


@pytest.mark.parametrize("r_max", [16, 32, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("variant", ["rf1", "rf2_segment", "rf2_index"])
def test_widths_dtypes_and_scales(monkeypatch, r_max, dtype, variant):
    for t in (3, 300):
        x, A, B, ranks, y0 = make(1024, (1024, 256), t, r_max=r_max, dtype=dtype, sort=variant == "rf2_segment")
        table = torch.rand(r_max + 1, device="cuda")
        check(monkeypatch, x, A, B, ranks, y0, (1024, 256), True, VARIANTS[variant], scale_table=table, r_limit=r_max)
        check(monkeypatch, x, A, B, ranks, y0, (1024, 256), False, VARIANTS[variant])


def test_same_shape_fp16_then_bf16(monkeypatch):
    # The plan key holds the element size, not the dtype: the kernel cache must still keep them apart.
    for dtype in (torch.float16, torch.bfloat16, torch.float16):
        x, A, B, ranks, y0 = make(896, (896,), 40, dtype=dtype)
        check(monkeypatch, x, A, B, ranks, y0, (896,), True, {}, r_limit=64)


def test_tuned_table_entries(monkeypatch):
    # Every tuned key _config can set (bm, warps, bh, stages, bn, expand_warps, loop_expand), in force on this GPU.
    gname = torch.cuda.get_device_name(0)
    entries = [dict(bm=32, shrink_warps=8, bh=64, stages=2, bn=128, expand_warps=4),
               dict(bm=16, loop_expand=True), dict(bm=64, bn=64, expand_warps=8, stages=4)]
    for i, entry in enumerate(entries):
        table = {(w, r): entry for w in (16, 32, 64) for r in range(4)}
        monkeypatch.setattr(rf, "TUNED", table)
        monkeypatch.setattr(rf, "TUNED_GPU", gname)
        for t in (8, 200, 1500):
            for variant in ("rf1", "rf2_segment", "rf2_index"):
                x, A, B, ranks, y0 = make(896, (896, 128, 128), t, seed=i, sort=variant == "rf2_segment")
                check(monkeypatch, x, A, B, ranks, y0, (896, 128, 128), True, VARIANTS[variant])
        launcher = rf.RF1Launcher(*make(896, (896, 128, 128), 1)[1:3], (896, 128, 128), max_tokens=2048)
        for t in (8, 200, 1500):
            x, _, _, ranks, y0 = make(896, (896, 128, 128), t)
            x, A, B, ranks, y0 = x, launcher.A_t, launcher.B, ranks, y0
            want = via_reference(monkeypatch, lambda: rf.rankfuse_lora(x, A, B, ranks, out=y0.clone(), add=True,
                                                                         slice_sizes=(896, 128, 128), r_limit=64))
            got = y0.clone()
            launcher(x, ranks, got, r_limit=64)
            assert torch.equal(got, want)


def test_settings_changed_at_run_time(monkeypatch):
    # A plan built before a setting changes must not be reused after it (rf1_ablation's settings).
    x, A, B, ranks, y0 = make(896, (896, 128, 128), 600)
    check(monkeypatch, x, A, B, ranks, y0, (896, 128, 128), True, {})
    n_before = len(rf._SPLITK_PLANS)
    for name, value in [("MAX_SPLIT_K", 1), ("MIN_SPLIT_K_TILED", 1), ("SPLITK_STAGES", 2), ("SPLITK_BLOCK_H", 64),
                        ("EXPAND_WAVES", 1), ("EXPAND_TILED_MIN_TOKENS", 10**9), ("EXPAND_TILED_WARPS", 4),
                        ("USE_ALIGNED_HINT", False), ("_block_m", lambda num_tokens, r_pad: 16)]:
        with monkeypatch.context() as m:
            m.setattr(rf, name, value)
            check(monkeypatch, x, A, B, ranks, y0, (896, 128, 128), True, {})
    assert len(rf._SPLITK_PLANS) > n_before
    check(monkeypatch, x, A, B, ranks, y0, (896, 128, 128), True, {})


def test_rf2_ranges_and_bucketed_unchanged(monkeypatch):
    # rf2_ranges calls _splitk per run on row views (other alignments); rf2_bucketed calls splitk_pass itself.
    from rankfuse.interface import REGISTRY
    from rankfuse.workloads import Workload, make_batch, make_lora
    for name in ("rf2_ranges", "rf2_ranges_seq", "rf2_bucketed", "rf2_bucketed_tuned", "rf2_segment", "rf2_index", "rf1_fused"):
        for tokens, rpr in ((16, 1), (4, 512)):
            w = Workload(tokens, rpr, r_max=64, dist="uniform", seed=1)
            lora, batch = make_lora(w, "cuda"), make_batch(w, "cuda")
            ranks = batch.token_ranks.sort().values.contiguous() if name in ("rf2_ranges", "rf2_ranges_seq", "rf2_segment") else batch.token_ranks
            impl = REGISTRY[name]
            kw = impl.prepare(batch.x, lora, ranks) if impl.prepare else {}
            want = via_reference(monkeypatch, lambda: impl(batch.x, lora, ranks, **kw).clone())
            for _ in range(2):
                assert torch.equal(impl(batch.x, lora, ranks, **kw), want), name
