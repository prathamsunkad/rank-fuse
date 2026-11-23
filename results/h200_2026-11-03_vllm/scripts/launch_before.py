"""Launch cost of rankfuse_lora BEFORE the fast path (run with rankfuse_fused.py as in git HEAD e9e10e8;
the RF1Launcher it imports via vllm_rf1_launch is not used). Prints per-call host time and a cProfile."""
import cProfile, pstats, time, statistics, sys
sys.path.insert(0, "bench")
import torch
import vllm_rf1_launch as L
from rankfuse.kernels import rankfuse_fused as rf
from rankfuse.baselines import padded
from rankfuse.interface import MatryoshkaLoRA
for name, h_in, sizes in L.LAYERS:
    for t in (1, 32, 512):
        x, A, B, ranks, y = L.make(h_in, sizes, t)
        loras = [MatryoshkaLoRA(A[s].contiguous(), B[s, :, :w].contiguous()) for s, w in enumerate(sizes)]
        offs = [sum(sizes[:s]) for s in range(len(sizes))]
        def b1():
            for s, w in enumerate(sizes):
                y[:, offs[s]:offs[s] + w] += padded._padded_body(x, loras[s], ranks, 64, None)
        def b1_one():  # single slice only (o/down) or first slice
            y[:, :sizes[0]] += padded._padded_body(x, loras[0], ranks, 64, None)
        r1 = lambda: rf.rankfuse_lora(x, A, B, ranks, out=y, add=True, slice_sizes=sizes, r_limit=64)
        r2 = lambda: rf.rankfuse_lora(x, A, B, ranks, out=y, add=True, slice_sizes=sizes)
        print(f"{name:8} T={t:5} b1 {L.cpu_us(b1):6.1f} b1_1slice {L.cpu_us(b1_one):6.1f} rf1 {L.cpu_us(r1):6.1f} rf1_sync {L.cpu_us(r2):6.1f}", flush=True)
x, A, B, ranks, y = L.make(896, (896,128,128), 8)
fn = lambda: rf.rankfuse_lora(x, A, B, ranks, out=y, add=True, slice_sizes=(896,128,128), r_limit=64)
for _ in range(20): fn()
torch.cuda.synchronize()
pr = cProfile.Profile(); pr.enable()
for _ in range(2000): fn()
torch.cuda.synchronize(); pr.disable()
pstats.Stats(pr).sort_stats("tottime").print_stats(22)
# component timings
def tm(f, n=3000):
    for _ in range(10): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e6
print("torch.empty ws", tm(lambda: torch.empty((3, 16, 8, 64), dtype=torch.float32, device=x.device)))
print("x.device", tm(lambda: x.device)); print("is_cuda", tm(lambda: x.is_cuda))
print("slice tables", tm(lambda: rf._slice_tables((896,128,128), x.device)))
