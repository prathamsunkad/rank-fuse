"""Rank skipping in the multi mode reads per-token ranks at CUDA-graph replay (rankfuse_fused.py, section 9).

Each tile takes its width from its tokens' ranks in token_ranks when it runs, as Tony's single-adapter
RANK_SKIP does; nothing about ranks is fixed when the graph is captured (the tile table's limit column
is unused). So, with the metadata left as captured (same adapters, MultiMeta not reloaded):

    lowered ranks  the replay is still exact and its GPU time drops (fewer rank rows loaded for every
                   variant; RF2-MA also runs narrower MMA)
    raised ranks   (up to the captured r_pad) the replay is still exact

and the controls that split the saving (rf2_ma_splitk_fw: no MMA saving; rf2_ma_splitk_full: no saving
at all) give the reference result.
"""

import statistics

import pytest
import torch

from rankfuse.debug import TOL
from rankfuse.multi import MULTI_REGISTRY, MultiWorkload, build_meta, load_all, make_weights, tile_capacity

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

if torch.cuda.is_available():
    load_all(include_slora=False)

MODE = {"rf1_ma": "group", "rf2_ma_splitk": "rank_sorted", "rf2_ma_splitk_fw": "rank_sorted", "rf2_ma_splitk_full": "rank_sorted",
        "rf2_ma_widths": "adapter_bucket", "rf2_ma_segment": "segment"}


def _close(y, x, W, ta, tr):
    expected = MULTI_REGISTRY["b0_multi"](x, W, ta, tr)
    rtol, atol = TOL["fp16"]
    bad = ~torch.isclose(y.float(), expected, rtol=rtol, atol=atol)
    assert not bad.any(), f"{int(bad.any(1).sum())} tokens wrong, max err {float((y.float() - expected).abs().max()):.3g}"


def _setup(impl, T, h_in, h_out, N, rank, tpr=1):
    W = make_weights(MultiWorkload(1, 1, N, h_in=h_in, h_out=h_out, seed=3), "cuda")
    gen = torch.Generator().manual_seed(T)
    reqs = T // tpr
    ta = torch.randint(0, N, (reqs,), generator=gen).repeat_interleave(tpr).to(torch.int32).cuda()
    tr = torch.full((T,), rank, dtype=torch.int32, device="cuda")
    x = torch.randn(T, h_in, device="cuda").half()
    mode = MODE[impl]
    meta = build_meta(ta, tr, num_adapters=N, mode=mode, device="cuda", bm=16 if T <= 128 else 64, r_pad=64,
                      capacity=tile_capacity(T, 16 if T <= 128 else 64, N, mode))
    out = torch.empty(T, W.h_out, device="cuda").half()
    return W, x, ta, tr, meta, out


def _graph(fn, calls=1):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            fn()
    return g


def _replay_us(g, calls, reps=9):
    g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3 / calls)
    return statistics.median(times)


@pytest.mark.parametrize("impl", list(MODE))
@pytest.mark.parametrize("T,tpr", [(64, 1), (2048, 128)])
def test_lowered_and_raised_ranks_after_capture_are_exact(impl, T, tpr):
    W, x, ta, tr, meta, out = _setup(impl, T, 896, 896, 12, 40, tpr)
    g = _graph(lambda: MULTI_REGISTRY[impl](x, W, ta, tr, meta=meta, out=out))
    gen = torch.Generator().manual_seed(1)
    for new in (torch.randint(0, 9, (T,), generator=gen), torch.full((T,), 64), torch.randint(1, 65, (T,), generator=gen)):
        tr.copy_(new)  # in place: same buffer the graph reads; metadata not reloaded
        g.replay()
        torch.cuda.synchronize()
        _close(out, x, W, ta, tr)


# Decode with many adapters on a long reduction: adapter weights dominate the bytes, so the drop is large.
# A 4096-token prompt batch: reading x and writing the output dominate (64 MB at 4096 x 4096 against 8 MB
# of weights), so the drop is real but small (H100: 6-18%).
CASES = {"decode-4096": (128, 1, 4096, 4096, 64), "prefill-4096": (4096, 512, 4096, 4096, 8)}
MAX_RATIO = {"decode-4096": 0.8, "prefill-4096": 0.97}


@pytest.mark.parametrize("impl", ["rf1_ma", "rf2_ma_splitk", "rf2_ma_widths", "rf2_ma_segment"])
@pytest.mark.parametrize("case", list(CASES))
def test_gpu_time_drops_when_ranks_drop_after_capture(impl, case):
    T, tpr, h_in, h_out, N = CASES[case]
    W, x, ta, tr, meta, out = _setup(impl, T, h_in, h_out, N, 64, tpr)
    calls = 10
    g = _graph(lambda: MULTI_REGISTRY[impl](x, W, ta, tr, meta=meta, out=out), calls)
    t_hi = _replay_us(g, calls)
    tr.fill_(8)  # every token drops from rank 64 to rank 8 after capture
    t_lo = _replay_us(g, calls)
    print(f"{impl} {case}: rank 64 {t_hi:.1f} us -> rank 8 {t_lo:.1f} us ({t_lo / t_hi:.2f}x)")
    _close(out, x, W, ta, tr)
    assert t_lo < MAX_RATIO[case] * t_hi, f"{impl} {case}: {t_hi:.1f} us at rank 64, {t_lo:.1f} us at rank 8"


@pytest.mark.parametrize("impl", ["rf2_ma_splitk_fw", "rf2_ma_splitk_full"])
@pytest.mark.parametrize("pages", ["dense", "shuffled"])
def test_controls_match_reference(impl, pages):
    W = make_weights(MultiWorkload(1, 1, 9, h_in=896, h_out=896, seed=2, pages=pages), "cuda", slice_sizes=None)
    gen = torch.Generator().manual_seed(0)
    for T in (33, 300):
        ta = torch.randint(-1, 9, (T,), generator=gen).to(torch.int32).cuda()
        tr = torch.randint(0, 65, (T,), generator=gen).to(torch.int32).cuda()
        x = torch.randn(T, 896, device="cuda").half()
        m = MULTI_REGISTRY[impl]
        _close(m(x, W, ta, tr, **m.prepare(x, W, ta, tr)), x, W, ta, tr)


def test_full_control_refuses_partially_resident_pages():
    W = make_weights(MultiWorkload(1, 1, 4, h_in=896, h_out=896, seed=2, pages="shuffled"), "cuda")
    W.page_table[0, -1] = -1
    assert MULTI_REGISTRY["rf2_ma_splitk_full"].unsupported(W)
