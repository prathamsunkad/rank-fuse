"""Multi-adapter RankFuse plugin pieces (rankfuse/integration/{multi_ops,multi_punica,paging}.py)
without a vLLM engine: the op over the wrapper's static buffers, CUDA-graph replay across steps
whose adapters / ranks / residency change, the plugin's page manager and adapter packing.

The end-to-end checks against stock vLLM are bench/vllm_multi_check.py and
bench/vllm_multi_layer_check.py (engines, minutes each; scripts/ice/vllm_multi.sh runs them).
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from rankfuse.multi import PAGE, PagedWeights, tile_capacity  # noqa: E402
from rankfuse.paging import model_pages  # noqa: E402
from rankfuse.integration import paging  # noqa: E402


def _ref(x, A, B, sizes, rows, ranks):
    """fp32 B0 from dense A [N, S, r, h_in], B [N, S, r, n_max]."""
    y = torch.zeros(x.shape[0], sum(sizes), dtype=torch.float32, device=x.device)
    for t in range(x.shape[0]):
        a, r = int(rows[t]), int(ranks[t])
        if a < 0 or r == 0:
            continue
        c = 0
        for s, n in enumerate(sizes):
            y[t, c:c + n] = (x[t].float() @ A[a, s, :r].float().T) @ B[a, s, :r, :n].float()
            c += n
    return y


class _Buffers:
    """The wrapper's static buffers and its per-step refill (MultiPunicaWrapper._update_multi_metadata)."""

    def __init__(self, max_tokens, N, r_max, dev):
        from rankfuse.multi import _r_pad

        self.N, self.r_pad_full = N, _r_pad(r_max)
        self.ranks = torch.zeros(max_tokens, dtype=torch.int32, device=dev)
        self.order = torch.zeros(max_tokens, dtype=torch.int32, device=dev)
        self.tiles = torch.zeros(tile_capacity(max_tokens, 16, N, "adapter_bucket"), 5, dtype=torch.int32, device=dev)
        self.meta_cpu = torch.zeros(6, dtype=torch.int64)

    def load(self, rows, rk, T, graph_max, kernel="rf1_ma"):
        from rankfuse.integration import multi_ops

        mode = multi_ops.KERNELS[kernel]
        from rankfuse.integration.multi_punica import step_meta

        rows, rk = np.asarray(rows, dtype=np.int64), np.asarray(rk, dtype=np.int64)
        m = step_meta(rows, rk, T, graph_max, self.N, self.r_pad_full, mode=mode)
        self.order[:T].copy_(m.order.pin_memory(), non_blocking=True)
        self.tiles[: m.capacity].copy_(m.tiles.pin_memory(), non_blocking=True)
        self.ranks[:T].copy_(torch.from_numpy(rk.astype(np.int32)).pin_memory(), non_blocking=True)
        self.meta_cpu.copy_(torch.tensor([T, m.capacity, m.bm, m.r_pad, int(rk.max() > 0),
                                          multi_ops.KERNEL_NAMES.index(kernel)]))

    def call(self, y, x, W, wid):
        from rankfuse.integration import multi_ops

        multi_ops.multi_lora_linear(y, x, W.A_pool, W.B_pool, W.page_table, self.ranks, self.order, self.tiles,
                                    self.meta_cpu, wid, True)


def _steps(rng, T, N, r_caps, n_steps):
    out = []
    for _ in range(n_steps):
        n = int(rng.integers(1, T + 1))  # real tokens; the rest is padding
        rows = rng.integers(-1, N, size=n)
        rk = np.array([0 if a < 0 else int(rng.choice([1, 3, 8, 16, 17, 31, 32, 48, 64][: 1 + 8]))
                       for a in rows], dtype=np.int64)
        rk = np.array([0 if a < 0 else min(r, r_caps[a]) for a, r in zip(rows, rk)], dtype=np.int64)
        rows = np.where(rk > 0, rows, -1)
        rows = np.concatenate([rows, -np.ones(T - n, dtype=np.int64)])
        rk = np.concatenate([rk, np.zeros(T - n, dtype=np.int64)])
        out.append((rows, rk))
    return out


@cuda
@pytest.mark.parametrize("kernel", ["rf1_ma", "rf1_ma_table", "rf2_ma_splitk", "rf2_ma_splitk_table", "rf2_ma_widths"])
@pytest.mark.parametrize("T", [1, 16, 48, 160])
def test_op_dense_graph_replay_matches_b0(T, kernel):
    """One graph captured at T replays steps with new adapters / ranks / padding (metadata loaded
    into the same buffers) and matches B0, as eager calls with the per-step metadata do."""
    from rankfuse.integration import multi_ops

    torch.manual_seed(0)
    dev = "cuda"
    N, S, r_max, h_in, sizes = 6, 3, 64, 256, (256, 64, 64)
    A = (torch.randn(N, S, r_max, h_in) / h_in**0.5).to(dev, torch.bfloat16)
    B = (torch.randn(N, S, r_max, max(sizes)) / 8).to(dev, torch.bfloat16)
    W = PagedWeights.dense(A, B, slice_sizes=sizes)
    wid = multi_ops.register_weights(W, "test")
    buf = _Buffers(512, N, r_max, dev)
    rng = np.random.default_rng(T)
    steps = _steps(rng, T, N, [64] * N, 6)
    x = torch.randn(T, h_in, device=dev, dtype=torch.bfloat16)
    y = torch.zeros(T, sum(sizes), device=dev, dtype=torch.bfloat16)
    # capture with the first step's metadata (graph_max >= T: fixed r_pad / bm / capacity)
    buf.load(*steps[0], T, graph_max=512, kernel=kernel)
    buf.call(y, x, W, wid)  # warm-up (compiles)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        buf.call(y, x, W, wid)
    for rows, rk in steps:
        buf.load(rows, rk, T, graph_max=512, kernel=kernel)
        x.copy_(torch.randn(T, h_in, device=dev, dtype=torch.bfloat16))
        y.zero_()
        g.replay()
        ref = _ref(x, A, B, sizes, rows, rk)
        torch.testing.assert_close(y.float(), ref, atol=3e-2, rtol=3e-2)
        # eager with the step's own metadata (T above any graph)
        y2 = torch.zeros_like(y)
        buf.load(rows, rk, T, graph_max=0, kernel=kernel)
        buf.call(y2, x, W, wid)
        torch.testing.assert_close(y2.float(), ref, atol=3e-2, rtol=3e-2)


def _layout(h_in=128, sizes=(128, 32, 32), layers=2):
    blocks = []
    for layer in range(layers):
        blocks.append(((2 * layer, f"l{layer}.qkv"), h_in, sizes))
        blocks.append(((2 * layer + 1, f"l{layer}.o"), sizes[0], (h_in,)))
    return paging.ModuleLayout(blocks)


def _adapter(layout, r, seed):
    gen = torch.Generator().manual_seed(seed)
    w = {}
    for key, fl in layout.blocks.items():
        A = (torch.randn(fl.n_slices, r, fl.h_in, generator=gen) / fl.h_in**0.5).to(torch.bfloat16)
        Bs = [(torch.randn(r, n, generator=gen) / 4).to(torch.bfloat16) for n in fl.slice_sizes]
        w[key] = (A, Bs)
    return w


@cuda
def test_paged_graph_replay_across_evictions_matches_b0():
    """The op over PluginPageManager.weights: one graph per layer captured once; steps move pages
    (ensure evicts other adapters' top pages and reloads), the graph is replayed after
    wait_for_copies and matches B0 from the original weights; end_step fences reuse."""
    from rankfuse.integration import multi_ops

    torch.manual_seed(0)
    dev = "cuda"
    layout = _layout()
    r_max = 64
    n_adapters = 8
    mgr = paging.PluginPageManager(layout, 10, num_adapter_slots=n_adapters, r_max=r_max, dtype=torch.bfloat16,
                                   device=dev)
    caps = [16, 32, 64, 64, 48, 64, 17, 64]
    W8 = {}
    for a in range(n_adapters):
        W8[a] = _adapter(layout, caps[a], 100 + a)
        mgr.register(a + 1, layout.pack(W8[a], caps[a], dtype=torch.bfloat16), caps[a])
    keys = list(layout.blocks)
    Ws = [mgr.weights(*k) for k in keys]
    wids = [multi_ops.register_weights(W, str(k)) for W, k in zip(Ws, keys)]
    T = 32
    buf = _Buffers(256, mgr.num_rows, r_max, dev)
    xs = {k: torch.randn(T, layout.blocks[k].h_in, device=dev, dtype=torch.bfloat16) for k in keys}
    ys = {k: torch.zeros(T, sum(layout.blocks[k].slice_sizes), device=dev, dtype=torch.bfloat16) for k in keys}

    def dense(k, a):
        A, Bs = W8[a][k]
        fl = layout.blocks[k]
        Ad = torch.zeros(1, fl.n_slices, r_max, fl.h_in, dtype=torch.bfloat16)
        Bd = torch.zeros(1, fl.n_slices, r_max, max(fl.slice_sizes), dtype=torch.bfloat16)
        Ad[0, :, : A.shape[1]] = A
        for s, b in enumerate(Bs):
            Bd[0, s, : b.shape[0], : b.shape[1]] = b
        return Ad.to(dev), Bd.to(dev)

    rng = np.random.default_rng(1)

    def step(adapters):
        """Tokens of up to 3 adapters, ranks <= cap, budget <= 10 frames - every pick fits."""
        ids = rng.choice(adapters, size=T)
        rk = np.array([int(rng.integers(1, caps[a - 1] + 1)) for a in ids])
        need = paging.max_rank_per_adapter(ids, rk)
        mgr.begin_step(need)
        mgr.ensure(need)
        rows = np.array([mgr.slot(a) for a in ids])
        buf.load(rows, rk, T, graph_max=256)
        mgr.wait_for_copies()
        return ids, rows, rk

    ids, rows, rk = step([1, 2])
    for k, W, wid in zip(keys, Ws, wids):
        buf.call(ys[k], xs[k], W, wid)
    mgr.end_step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for k, W, wid in zip(keys, Ws, wids):
            buf.call(ys[k], xs[k], W, wid)
    mgr.end_step()
    groups = [[1, 2], [3, 4], [5, 6], [7, 8], [3, 8], [2, 4, 6], [1, 5, 7], [4, 6], [8, 3]]
    for g in groups * 2:
        ids, rows, rk = step(g)
        for k in keys:
            ys[k].zero_()
        graph.replay()
        mgr.end_step()
        mgr.check_invariants()
        for k in keys:
            ref = torch.zeros(T, sum(layout.blocks[k].slice_sizes), device=dev)
            for a in np.unique(ids):
                sel = ids == a
                Ad, Bd = dense(k, int(a) - 1)
                ref[torch.from_numpy(sel).to(dev)] = _ref(xs[k][sel], Ad, Bd, layout.blocks[k].slice_sizes,
                                                          np.zeros(int(sel.sum()), dtype=np.int64), rk[sel])
            torch.testing.assert_close(ys[k].float(), ref, atol=3e-2, rtol=3e-2)
    assert mgr.stats.evictions > 0 and mgr.stats.page_loads > sum(math.ceil(c / PAGE) for c in caps)


def test_plugin_manager_unregister_reuses_slots_and_keeps_zero_row():
    layout = _layout(h_in=32, sizes=(32, 16, 16), layers=1)
    mgr = paging.PluginPageManager(layout, 6, num_adapter_slots=3, r_max=64, dtype=torch.float32, device="cpu")
    assert mgr.num_rows == 4 and mgr.dummy_row == 3
    assert (mgr.page_table[mgr.dummy_row] == mgr.zero_frame).all()
    assert mgr.pool.frames[mgr.zero_frame].abs().sum() == 0 and mgr.usable_frames == 6
    host = lambda r: layout.pack(_adapter(layout, r, r), r, dtype=torch.float32, pin=False)  # noqa: E731
    s1, s2, s3 = (mgr.register(a, host(32), 32) for a in (10, 11, 12))
    assert {s1, s2, s3} == {0, 1, 2}
    with pytest.raises(RuntimeError):
        mgr.register(13, host(16), 16)
    e = mgr.epoch
    mgr.begin_step({10: 32, 11: 32}); mgr.ensure({10: 32, 11: 32}); mgr.end_step()
    assert mgr.free_frames == 2
    assert mgr.unregister(10) and mgr.epoch == e + 1 and not mgr.is_registered(10)
    assert (mgr.page_table[s1] == -1).all() and mgr.free_frames == 4
    assert mgr.register(13, host(16), 16) == s1  # slot reused
    mgr.begin_step({13: 16}); mgr.ensure({13: 16}); mgr.end_step()
    mgr.check_invariants()
    assert (mgr.page_table[mgr.dummy_row] == mgr.zero_frame).all()
    assert not mgr.unregister(10)


def test_pack_lora_model_matches_checkpoint_layout():
    """vLLM LoRAModel (packed q|k|v with lora_b [out, r], one slice missing) -> host pages whose
    blocks hold A rows and B made rank-major; scaling applied when not folded."""
    layout = _layout(h_in=32, sizes=(32, 16, 16), layers=1)
    r = 20
    g = torch.Generator().manual_seed(0)
    qa, ka = torch.randn(r, 32, generator=g), torch.randn(r, 32, generator=g)
    qb, kb = torch.randn(32, r, generator=g), torch.randn(16, r, generator=g)
    oa, ob = torch.randn(r, 32, generator=g), torch.randn(32, r, generator=g)
    loras = {
        "l0.qkv": SimpleNamespace(lora_a=[qa, ka, None], lora_b=[qb, kb, None], scaling=[1, 1, None]),
        "l0.o": SimpleNamespace(lora_a=oa, lora_b=ob, scaling=0.5),
    }
    model = SimpleNamespace(rank=r, get_lora=lambda n: loras.get(n))
    keys = {"l0.qkv": (0, "l0.qkv"), "l0.o": (1, "l0.o")}
    host, rank = paging.pack_lora_model(layout, keys, model, lambda m, n: m.get_lora(n), dtype=torch.float32, pin=False)
    assert rank == r and host.shape == (2, layout.frame_elems)
    fl = layout.blocks[(0, "l0.qkv")]
    page1 = host[1]
    # rank row 17 = page 1 row 1
    a_q = page1[fl.a_off[0] + 1 * 32: fl.a_off[0] + 2 * 32]
    torch.testing.assert_close(a_q, qa[17])
    b_k = page1[fl.b_off[1] + 1 * 16: fl.b_off[1] + 2 * 16]
    torch.testing.assert_close(b_k, kb[:, 17])
    assert page1[fl.b_off[2]: fl.b_off[2] + 16 * 16].abs().sum() == 0  # missing v slice
    flo = layout.blocks[(1, "l0.o")]
    b_o = host[0][flo.b_off[0] + 3 * 32: flo.b_off[0] + 4 * 32]
    torch.testing.assert_close(b_o, ob[:, 3] * 0.5)
    # rows past the rank in the top page are zero
    assert host[1][fl.a_off[0] + 4 * 32: fl.a_off[0] + 16 * 32].abs().sum() == 0


def test_max_rank_per_adapter_and_request_pages(tmp_path):
    assert paging.max_rank_per_adapter(np.array([0, 3, 3, 5, 5, 5]), np.array([0, 8, 16, 4, 64, 0])) == {3: 16, 5: 64}
    d = tmp_path / "a"
    d.mkdir()
    (d / "adapter_config.json").write_text('{"r": 32}')
    lr = SimpleNamespace(lora_int_id=7, lora_path=str(d), matryoshka_rank=None)
    assert paging.request_pages(lr, 256, 1 << 30) == (7, 2)  # full rank -> the adapter's r = 32
    lr.matryoshka_rank = 17
    assert paging.request_pages(lr, 256, 1 << 30) == (7, 2)
    lr.matryoshka_rank = 8
    assert paging.request_pages(lr, 256, 1 << 30) == (7, 1)
    reqs = [SimpleNamespace(lora_request=lr), SimpleNamespace(lora_request=None)]
    assert paging.step_pages(reqs, 256, 1 << 30) == {7: 1}


def test_in_step_adapter_pages_above_its_need_are_evictable():
    """Regression (pagedsmall engine run): adapter B holds 4 pages from an earlier full-rank step;
    the next step uses B at rank 16 (1 page) and A at rank 64 (4 pages). The budget (5 <= 5 frames)
    allows it, so B's pages 1..3 must be trimmed instead of ensure() failing."""
    layout = _layout(h_in=32, sizes=(32, 16, 16), layers=1)
    mgr = paging.PluginPageManager(layout, 5, num_adapter_slots=2, r_max=64, dtype=torch.float32, device="cpu")
    host = lambda r, s: layout.pack(_adapter(layout, r, s), r, dtype=torch.float32, pin=False)  # noqa: E731
    mgr.register("A", host(64, 1), 64)
    mgr.register("B", host(64, 2), 64)
    mgr.begin_step({"B": 64}); mgr.ensure({"B": 64}); mgr.end_step()
    assert mgr.resident_pages("B") == 4 and mgr.free_frames == 1
    mgr.begin_step({"A": 64, "B": 16})
    got = mgr.ensure({"A": 64, "B": 16})
    mgr.end_step()
    assert got == {"A": 64, "B": 16}
    assert mgr.resident_pages("A") == 4 and mgr.resident_pages("B") == 1
    mgr.check_invariants()
    # the trimmed adapter still serves its prefix: page 0 of B is the same frame as before
    assert mgr.cpu_page_table[mgr.slot("B"), 0] >= 0


def test_prefetch_of_in_step_adapter_above_its_rank_terminates():
    """Regression (20-frame engine hang): B runs at rank 16 this step and a prefetch hint asks for
    B at rank 64 with the pool full of in-step pages. The prefetch must stop (partial), not load
    and trim B's page 1 forever."""
    layout = _layout(h_in=32, sizes=(32, 16, 16), layers=1)
    mgr = paging.PluginPageManager(layout, 5, num_adapter_slots=2, r_max=64, dtype=torch.float32, device="cpu")
    host = lambda r, s: layout.pack(_adapter(layout, r, s), r, dtype=torch.float32, pin=False)  # noqa: E731
    mgr.register("A", host(64, 1), 64)
    mgr.register("B", host(64, 2), 64)
    mgr.begin_step({"A": 64, "B": 16})
    mgr.ensure({"A": 64, "B": 16})
    assert mgr.prefetch("B", 64) == 16  # pool full of this step's needed pages: nothing to give
    mgr.end_step()
    mgr.check_invariants()
    assert mgr.prefetch("B", 64) == 16 + 0 or mgr.resident_pages("B") >= 1  # idle now: may evict A
    mgr.check_invariants()
