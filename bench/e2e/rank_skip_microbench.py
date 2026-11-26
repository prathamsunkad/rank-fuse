"""Graph-replayed GPU time of one LoRA layer call: RF1 (as the Stage 1 plugin runs it under CUDA
graphs, r_limit = r_max) vs the rank-skip launchers (rf2_index with a sorted order, rf2_segment).

    python bench/e2e/rank_skip_microbench.py <out.jsonl> [--model q7|q05]

For each layer of the model (q|k|v, o, gate|up, down), T tokens and a rank mix, it captures one
CUDA graph of 20 back-to-back calls per kernel and reports the median replay time per call over
50 replays (torch.cuda.Event). Rank mixes: all 256, uniform over {8..256}, skewed (alpha 1),
all 8, and "uniform + 0": a quarter of the tokens without the adapter. bf16, r_max 256, add=True.
Per-layer sums over one transformer layer are printed (the whole-step LoRA cost is that times
the layer count).
"""

import argparse
import json
import statistics

import numpy as np
import torch

from rankfuse.integration.rank_skip import RankSkipLauncher, fill_order
from rankfuse.kernels.rankfuse_fused import RF1Launcher

MODELS = {
    "q7": dict(hidden=3584, inter=18944, q=3584, kv=512),
    "q05": dict(hidden=896, inter=4864, q=896, kv=128),
}
TS = [1, 8, 32, 64, 128, 256, 512, 2048]
TIERS = [8, 16, 32, 64, 128, 256]


def layers(m):
    h, i, q, kv = m["hidden"], m["inter"], m["q"], m["kv"]
    return {"qkv": (h, (q, kv, kv)), "o": (q, (h,)), "gate_up": (h, (i, i)), "down": (i, (h,))}


def mix(name, T, rng):
    if name == "all256":
        return np.full(T, 256)
    if name == "all8":
        return np.full(T, 8)
    if name == "uniform":
        return rng.choice(TIERS, T)
    if name == "skewed":
        w = (np.asarray(TIERS) / 8.0) ** -1.0
        return rng.choice(TIERS, T, p=w / w.sum())
    if name == "uniform+0":
        r = rng.choice(TIERS, T)
        r[rng.random(T) < 0.25] = 0
        return r
    raise ValueError(name)


def time_graph(fn, reps=20, replays=50):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(replays):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / reps)
    return statistics.median(ts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--model", default="q7", choices=list(MODELS))
    a = p.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    MAXT = max(TS)
    dt = torch.bfloat16
    rows = []
    for lname, (h_in, sizes) in layers(MODELS[a.model]).items():
        n_max = max(sizes)
        A = (torch.randn(len(sizes), 256, h_in, device="cuda") / h_in**0.5).to(dt)
        B = (torch.randn(len(sizes), 256, n_max, device="cuda") * 0.02).to(dt)
        rf1 = RF1Launcher(A, B, sizes, max_tokens=MAXT)
        idx = RankSkipLauncher(A, B, sizes, max_tokens=MAXT, indexed=True)
        seg = RankSkipLauncher(A, B, sizes, max_tokens=MAXT, indexed=False)
        ranks = torch.zeros(MAXT, dtype=torch.int32, device="cuda")
        order = torch.arange(MAXT, dtype=torch.int32, device="cuda")
        host = np.arange(MAXT, dtype=np.int32)
        for T in TS:
            x = torch.randn(T, h_in, device="cuda").to(dt)
            y = torch.zeros(T, sum(sizes), device="cuda", dtype=dt)
            for mname in ("all256", "uniform", "skewed", "all8", "uniform+0"):
                r = mix(mname, T, rng).astype(np.int32)
                ranks[:T].copy_(torch.from_numpy(r))
                fill_order(host, r, T, MAXT)
                order.copy_(torch.from_numpy(host))
                row = dict(model=a.model, layer=lname, T=T, mix=mname,
                           rf1_rmax_us=time_graph(lambda: rf1(x, ranks, y, r_limit=256, add=True)),
                           rf1_batchmax_us=time_graph(lambda: rf1(x, ranks, y, r_limit=max(int(r.max()), 1), add=True)),
                           rf2_index_us=time_graph(lambda: idx(x, ranks, order, y, r_limit=256, add=True)),
                           rf2_segment_us=time_graph(lambda: seg(x, ranks, order, y, r_limit=256, add=True)))
                rows.append(row)
                print(json.dumps(row), flush=True)
    # Host cost per step that rf2_index adds in the plugin (_update_rank_metadata): argsort of the
    # step's token ranks + a pinned copy + the H2D copy_ (async; measured to completion here).
    import time
    for T in TS + [8192]:
        r = rng.choice(TIERS, T).astype(np.int64)
        buf = torch.arange(8192, dtype=torch.int32, device="cuda")
        host8 = np.arange(8192, dtype=np.int32)
        ts = []
        for _ in range(200):
            t0 = time.perf_counter()
            k = fill_order(host8, r, T, T)
            buf[:k].copy_(torch.from_numpy(host8[:k].copy()).pin_memory(), non_blocking=True)
            ts.append(time.perf_counter() - t0)
        torch.cuda.synchronize()
        row = dict(model=a.model, layer="host_order", T=T, host_order_us=statistics.median(ts) * 1e6)
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("\nper transformer layer (sum of 4 LoRA calls), us: rf1@r_max / rf1@batch-max / rf2_index / rf2_segment")
    for T in TS:
        for mname in ("all256", "uniform", "skewed", "all8", "uniform+0"):
            sel = [r for r in rows if r["T"] == T and r.get("mix") == mname]
            s = [sum(r[k] for r in sel) for k in ("rf1_rmax_us", "rf1_batchmax_us", "rf2_index_us", "rf2_segment_us")]
            print(f"T={T:5d} {mname:10s} " + " / ".join(f"{v:7.1f}" for v in s) + f"   index speedup vs rf1@r_max {s[0] / s[2]:.2f}x")


if __name__ == "__main__":
    main()
