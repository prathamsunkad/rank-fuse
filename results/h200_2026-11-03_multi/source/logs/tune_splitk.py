"""Experiment: split-K slices and expand waves for RF1-MA when tiles >= SMs (many-adapter decode)."""
import sys, statistics, torch
sys.path.insert(0, "/storage/ice1/4/7/psunkad3/rankfuse/rank-fuse/bench")
sys.path.insert(0, "/storage/ice1/4/7/psunkad3/rankfuse/rank-fuse")
from run_multi_sweep import time_graph
from rankfuse.multi import MULTI_REGISTRY, MultiWorkload, load_all, make_batch, make_weights
from rankfuse.kernels import rankfuse_fused as rf
from rankfuse.kernels import rankfuse_multi as rm
load_all(include_slora=False)
for shape in [(4096, 4096), (4864, 896), (896, 4864), (896, 896)]:
    for N in (16, 64, 256):
        W = make_weights(MultiWorkload(1, 1, N, h_in=shape[0], h_out=shape[1], pages="shuffled"))
        for T in (32, 128, 256):
            w = MultiWorkload(T, 1, N, h_in=shape[0], h_out=shape[1], dist="uniform")
            x, ta, tr = make_batch(w)
            meta = MULTI_REGISTRY["rf1_ma"].prepare(x, W, ta, tr)["meta"]
            out = torch.zeros(T, W.h_out, device="cuda", dtype=x.dtype)
            res = {}
            for ew in (4, 8):
                rf.EXPAND_WAVES = ew
                for sk in (None, 2, 4, 8):
                    f = lambda: rm.multi_lora(x, W, tr, meta, out=out, add=True, split_k=sk)
                    res[f"e{ew}/s{sk}"] = statistics.median(time_graph(f, 5, 5, 20, True))
            rf.EXPAND_WAVES = 4
            best = min(res, key=res.get)
            print(shape, N, T, "tiles", meta.num_tiles, {k: round(v, 1) for k, v in res.items()}, "best", best, flush=True)
