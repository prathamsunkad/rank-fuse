"""Host-side (Python + launch) cost of one eager LoRA call: RF1 vs B1, before and after the fast path.

The serving fast path of rankfuse_lora (rankfuse/kernels/rankfuse_fused.py, `RF1Launcher`) moves
input checks to setup, takes r_limit from the caller (no host sync), caches the launch
configuration and the compiled Triton kernels, reuses its fp32 workspace and launches the
compiled kernels directly. This script measures where the time of one eager call goes, on the
four Qwen2.5-0.5B projection shapes vLLM uses (merged q|k|v, o, merged gate|up, down; r_max 64,
fp16, add into the base output):

    b1          B1's body (rankfuse.baselines.padded._padded_body): ~6 PyTorch ops, r given
    rf1         rankfuse_lora(..., out=y, add=True, r_limit=r)  (the whole-model harness call)
    rf1_sync    the same with r_limit=None (one aminmax().tolist() host sync per call)
    rf1_fast    RF1Launcher(...)(x, ranks, y, r_limit=r): the serving fast path

    cpu_us      wall time per call over N back-to-back calls with one synchronize at the end.
                At these sizes the GPU work is a few us, so this is the host cost per call.
    gpu_us      the same calls replayed from a CUDA graph (GPU time only)

    python bench/vllm_rf1_launch.py [out.json]      # repo root, inside a GPU job
    python bench/vllm_rf1_launch.py profile         # cProfile of 2000 rf1 calls (where the time goes)
"""

import cProfile
import json
import pstats
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rankfuse.baselines import padded  # noqa: E402
from rankfuse.interface import MatryoshkaLoRA  # noqa: E402
from rankfuse.kernels import rankfuse_fused as rf  # noqa: E402

# (name, h_in, slice widths) of the four LoRA layer types of Qwen2.5-0.5B (vLLM's merged layout)
LAYERS = [("qkv", 896, (896, 128, 128)), ("o", 896, (896,)), ("gate_up", 896, (4864, 4864)), ("down", 4864, (896,))]
TOKENS = [1, 8, 32, 128, 512, 2048]
R_MAX = 64


def make(h_in, sizes, t, dtype=torch.float16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n = len(sizes)
    n_max = max(sizes)
    A = (torch.randn(n, R_MAX, h_in, generator=g, device="cuda") * 0.02).to(dtype)
    B = (torch.randn(n, R_MAX, n_max, generator=g, device="cuda") * 0.02).to(dtype)
    x = torch.randn(t, h_in, generator=g, device="cuda").to(dtype)
    tiers = torch.tensor([1, 2, 4, 8, 16, 32, 64], device="cuda", dtype=torch.int32)
    ranks = tiers[torch.randint(0, 7, (t,), generator=g, device="cuda")]
    ranks[0] = R_MAX
    y = torch.randn(t, sum(sizes), generator=g, device="cuda").to(dtype)
    return x, A, B, ranks, y


def impls(h_in, sizes, x, A, B, ranks, y):
    r = R_MAX  # every workload has one rank-64 token
    # B1 has no merged-slice form: one call per slice, into column views of y (as a harness would)
    loras = [MatryoshkaLoRA(A[s].contiguous(), B[s, :, :w].contiguous()) for s, w in enumerate(sizes)]
    offs = [sum(sizes[:s]) for s in range(len(sizes))]

    def b1():
        for s, w in enumerate(sizes):
            y[:, offs[s]:offs[s] + w] += padded._padded_body(x, loras[s], ranks, r, None)

    launcher = rf.RF1Launcher(A, B, sizes, max_tokens=max(TOKENS))
    return {
        "b1": b1,
        "rf1": lambda: rf.rankfuse_lora(x, A, B, ranks, out=y, add=True, slice_sizes=sizes, r_limit=r),
        "rf1_sync": lambda: rf.rankfuse_lora(x, A, B, ranks, out=y, add=True, slice_sizes=sizes),
        "rf1_fast": lambda: launcher(x, ranks, y, r_limit=r),
    }


def cpu_us(fn, n=300):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    runs = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        runs.append((time.perf_counter() - t) / n * 1e6)
    return statistics.median(runs)


def gpu_us(fn, n=50):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    g.replay()
    torch.cuda.synchronize()
    runs = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        runs.append(a.elapsed_time(b) / n * 1e3)
    return statistics.median(runs)


def main(out):
    print(torch.cuda.get_device_name(), flush=True)
    rows = []
    for name, h_in, sizes in LAYERS:
        for t in TOKENS:
            x, A, B, ranks, y = make(h_in, sizes, t)
            fns = impls(h_in, sizes, x, A, B, ranks, y)
            # the fast path must give the same output bits as rankfuse_lora
            y0 = y.clone()
            fns["rf1"]()
            ref = y.clone()
            y.copy_(y0)
            fns["rf1_fast"]()
            same = torch.equal(y, ref)
            row = {"layer": name, "tokens": t, "bitwise_equal_fast_vs_rf1": same, "cpu_us": {}, "gpu_us": {}}
            for k, fn in fns.items():
                row["cpu_us"][k] = cpu_us(fn)
                if k != "rf1_sync":
                    row["gpu_us"][k] = gpu_us(fn)
            rows.append(row)
            c, g = row["cpu_us"], row["gpu_us"]
            print(f"{name:<8} T={t:<5} cpu/call us: " + " ".join(f"{k} {v:6.1f}" for k, v in c.items())
                  + " | gpu us: " + " ".join(f"{k} {v:6.1f}" for k, v in g.items()) + f" | fast==rf1 {same}", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {out}")


def profile():
    name, h_in, sizes = LAYERS[0]
    x, A, B, ranks, y = make(h_in, sizes, 8)
    fns = impls(h_in, sizes, x, A, B, ranks, y)
    for k in ("rf1", "rf1_fast"):
        fn = fns[k]
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(2000):
            fn()
        torch.cuda.synchronize()
        pr.disable()
        print(f"\n===== {k}: cProfile of 2000 calls (qkv, T=8); times are totals over 2000 calls =====")
        pstats.Stats(pr).sort_stats("tottime").print_stats(18)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "profile":
        profile()
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "results/raw/vllm_rf1_launch.json")
