"""Does GPU time drop with rank under CUDA-graph replay? (the op path the vLLM plugin runs)

One CUDA graph is captured over every LoRA layer of a model (Qwen2.5-0.5B or -7B shapes: q|k|v, o,
gate|up, down per layer), all reading one PluginPageManager pool through torch.ops.rankfuse.
multi_lora_linear, exactly as the engine's layers do. It is captured at T tokens with the step's
metadata built the way the plugin builds it for replayable sizes (r_pad = max_lora_rank, fixed
block size and tile capacity). Then, without re-capturing, the tile table is reloaded in place for
steps where every token runs at rank r (r = 16 .. r_max), or at a mix, and the replay is timed.

    rf1_ma         (mode "group")        limits mask A/B loads only: bytes drop, MMA stays r_max wide
    rf2_ma_splitk  (mode "rank_sorted")  per-tile runtime rank buckets: work drops with rank

    python bench/vllm_multi_rank_time.py <0.5B|7B> <T> out.json

Kernels: every entry of multi_ops.KERNEL_NAMES. Pass criterion (printed): for rf2_ma_splitk, replay time at rank 16 <= 0.8 x replay time at r_max;
both modes match B0 on one layer at every rank.
"""

import json
import sys
import time

import numpy as np
import torch

from rankfuse.integration import multi_ops, paging
from rankfuse.integration.multi_punica import step_meta
from rankfuse.multi import _r_pad, tile_capacity
from rankfuse.paging.model_pages import ModelSpec

CFG = {"0.5B": dict(r_max=256, adapters=8), "7B": dict(r_max=128, adapters=6)}


def main(size, T, out_path):
    T = int(T)
    dev = "cuda"
    cfg = CFG[size]
    spec = ModelSpec.qwen25(size)
    blocks = [((layer * 4 + i, f"{layer}.{g.name}"), g.h_in, g.slice_sizes)
              for layer in range(spec.num_layers) for i, g in enumerate(spec.groups)]
    layout = paging.ModuleLayout(blocks)
    r_max, N = cfg["r_max"], cfg["adapters"]
    pages = N * r_max // 16
    mgr = paging.PluginPageManager(layout, pages, num_adapter_slots=N, r_max=r_max, dtype=torch.bfloat16, device=dev)
    for a in range(N):  # random pages straight into host buffers (packing is not what is timed)
        host = (torch.randn(r_max // 16, layout.frame_elems) * 0.02).to(torch.bfloat16).pin_memory()
        mgr.register(a, host, r_max)
    mgr.begin_step({a: r_max for a in range(N)})
    mgr.ensure({a: r_max for a in range(N)})
    mgr.wait_for_copies()
    mgr.end_step()
    torch.cuda.synchronize()
    keys = list(layout.blocks)
    Ws = [mgr.weights(*k) for k in keys]
    wids = [multi_ops.register_weights(W, str(k)) for W, k in zip(Ws, keys)]
    xs = {g.h_in: torch.randn(T, g.h_in, device=dev, dtype=torch.bfloat16) for g in spec.groups}
    ys = [torch.zeros(T, sum(layout.blocks[k].slice_sizes), device=dev, dtype=torch.bfloat16) for k in keys]
    rows_full = np.arange(T) % N
    R = mgr.num_rows
    res = {"size": size, "T": T, "r_max": r_max, "adapters": N, "layers": len(keys),
           "gpu": torch.cuda.get_device_name(), "modes": {}}
    ok = True
    for kernel in multi_ops.KERNEL_NAMES:
        mode = multi_ops.KERNELS[kernel]
        ranks_buf = torch.zeros(4096, dtype=torch.int32, device=dev)
        order = torch.zeros(4096, dtype=torch.int32, device=dev)
        tiles = torch.zeros(tile_capacity(4096, 16, R, "adapter_bucket"), 5, dtype=torch.int32, device=dev)
        meta_cpu = torch.zeros(6, dtype=torch.int64)

        def load(rk):
            rows = np.where(rk > 0, rows_full, -1).astype(np.int64)
            m = step_meta(rows, rk.astype(np.int64), T, 4096, R, _r_pad(r_max), mode=mode)
            order[:T].copy_(m.order)
            tiles[: m.capacity].copy_(m.tiles)
            ranks_buf[:T].copy_(torch.from_numpy(rk.astype(np.int32)))
            meta_cpu.copy_(torch.tensor([T, m.capacity, m.bm, m.r_pad, 1, multi_ops.KERNEL_NAMES.index(kernel)]))

        def run_all():
            for k, W, wid, y in zip(keys, Ws, wids, ys):
                multi_ops.multi_lora_linear(y, xs[W.h_in], W.A_pool, W.B_pool, W.page_table, ranks_buf, order, tiles,
                                            meta_cpu, wid, True)

        load(np.full(T, r_max))
        run_all()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run_all()
        cases = {f"r{r}": np.full(T, r) for r in (16, 32, 64, 128, 256) if r <= r_max}
        rng = np.random.default_rng(0)
        cases["mixed"] = rng.choice([r for r in (8, 16, 32, 64, 128, 256) if r <= r_max], size=T)
        times = {}
        A, B = Ws[1].gather_dense()  # o_proj of layer 0 (one slice), through the page table
        for name, rk in cases.items():
            load(rk)
            torch.cuda.synchronize()
            for _ in range(5):
                g.replay()
            st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            st.record()
            reps = 30
            for _ in range(reps):
                g.replay()
            en.record()
            torch.cuda.synchronize()
            times[name] = st.elapsed_time(en) / reps * 1000  # us per whole-model replay
            # correctness on one layer (o_proj of layer 0: single slice) against B0 at this rank mix
            k0 = 1
            W = Ws[k0]
            x = xs[W.h_in]
            ys[k0].zero_()
            g.replay()
            torch.cuda.synchronize()
            if True:
                ref = torch.zeros_like(ys[k0], dtype=torch.float32)
                for t in range(T):
                    a, r = int(rows_full[t]), int(rk[t])
                    ref[t] = (x[t].float() @ A[a, 0, :r].float().T) @ B[a, 0, :r, : W.slice_sizes[0]].float()
                err = float((ys[k0].float() - ref).abs().max() / ref.abs().max().clamp_min(1e-6))
                times[name + "_relerr"] = err
                ok &= err < 3e-2
        res["modes"][kernel] = times
        del g
    res["r16_over_rmax"] = {k: v["r16"] / v[f"r{r_max}"] for k, v in res["modes"].items()}
    res["ok"] = bool(ok and res["r16_over_rmax"]["rf2_ma_splitk"] <= 0.8)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    print("RANK TIME", "PASS" if res["ok"] else "FAIL")


if __name__ == "__main__":
    t0 = time.time()
    main(*sys.argv[1:4])
    print(f"{time.time() - t0:.0f} s")
