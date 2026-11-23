"""B2 (group tokens by rank, one PyTorch matmul pair per rank) on every one-layer experiment.

B2 was registered and tested but never timed next to RankFuse. This runs it on exactly the
batches of the earlier experiments (same Workload seeds, same custom batches), with the same
timing: each call captured in a CUDA graph and replayed (median of 7 runs of 50 replays).

B2 here = b2_grouped's body with the groups built once per step outside the graph:
    write mode:  out[idx_r] = (x[idx_r] @ A_t[:r].T) @ B[:r]        for each rank r present
    add mode:    out.index_add_(0, idx_r, (x[idx_r] @ A_t[:r].T) @ B[:r])
Rank-0 rows stay zero (write) or untouched (add). Every B2 result is checked against B0 first.

In each experiment the reference kernels of that table (rf1_fused, and rf2_splitk /
rf2_segment / rf2_bucketed for the RF2 tables) are re-timed in the same run, from the repo,
so ratios against B2 are same-run.

    python b2_grouped_runs.py [out.json]          # repo root, inside a GPU job
    python b2_grouped_runs.py ncu <T>             # add mode, uniform, last call profiled
"""

import itertools
import json
import sys

import torch
import triton

import rankfuse  # noqa: F401
from rankfuse.debug import TOL
from rankfuse.interface import REGISTRY, Batch
from rankfuse.kernels import rankfuse_fused as rf
from rankfuse.kernels.rankfuse_rf2_bucketed import Plan
from rankfuse.kernels.rankfuse_rf2_segment import block_m_for, contig_blocks
from rankfuse.workloads import Workload, make_batch, make_lora


def bench(fn, replays=50, runs=7):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(replays):
            g.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) / replays * 1e3)
    return sorted(times)[len(times) // 2]


def groups_of(ranks):
    """[(rank, index tensor)] for every rank present except 0 (built on the host once per step)."""
    return [(r, torch.nonzero(ranks == r).squeeze(1)) for r in torch.unique(ranks).tolist() if r > 0]


def b2_write(x, lora, groups, out):
    for r, idx in groups:
        out[idx] = (x[idx] @ lora.A_t[:r].T) @ lora.B[:r]
    return out


def b2_add(x, lora, groups, out):
    for r, idx in groups:
        out.index_add_(0, idx, (x[idx] @ lora.A_t[:r].T) @ lora.B[:r])
    return out


def check(x, lora, ranks, groups):
    ref = REGISTRY["b0_reference"](x, lora, ranks)
    rtol, atol = TOL["fp16"]
    out = torch.zeros(x.shape[0], lora.h_out, device="cuda", dtype=x.dtype)
    ok = torch.allclose(b2_write(x, lora, groups, out).float(), ref, rtol=rtol, atol=atol)
    base = torch.randn_like(out)
    got = b2_add(x, lora, groups, base.clone())
    ok &= torch.allclose(got.float(), ref + base.float(), rtol=rtol, atol=atol)
    if not ok:
        sys.exit(f"B2 does not match B0 (T={x.shape[0]})")


def custom_batch(seg_lens, seg_ranks, r_max=64, h=4096, seed=0):
    """Identical to checks/rf2_segment.py's custom_batch (same seeds, same tensors)."""
    w = Workload(1, 1, h_in=h, h_out=h, r_max=r_max, dtype="fp16", seed=seed)
    lora = make_lora(w, "cuda")
    gen = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(sum(seg_lens), h, generator=gen).to("cuda", torch.float16)
    return lora, Batch.from_segments(x, seg_lens, seg_ranks)


def workload(req, tpr, dist):
    w = Workload(req, tpr, h_in=4096, h_out=4096, r_max=64, dist=dist, dtype="fp16", seed=0)
    return make_lora(w, "cuda"), make_batch(w, "cuda")


def write_row(lora, batch, rf2=False):
    x, ranks = batch.x, batch.token_ranks
    r = int(ranks.max())
    g = groups_of(ranks)
    check(x, lora, ranks, g)
    out = torch.zeros(x.shape[0], lora.h_out, device="cuda", dtype=x.dtype)
    row = {"max_rank": r, "mean_rank": float(ranks.float().mean()), "groups": len(g),
           "rf1": bench(lambda: REGISTRY["rf1_fused"](x, lora, ranks, r_limit=r))}
    if rf2:
        order = torch.argsort(ranks, stable=True)
        bm = block_m_for(x.shape[0], r)
        blocks = contig_blocks(x.shape[0], bm, x.device)
        plan = Plan(ranks, x.device)
        row["rf2_splitk"] = bench(lambda: REGISTRY["rf2_splitk"](x, lora, ranks, r_limit=r, order=order))
        row["rf2_segment"] = bench(lambda: REGISTRY["rf2_segment"](x, lora, ranks, r_limit=r, blocks=blocks, bm=bm))
        row["rf2_bucketed"] = bench(lambda: REGISTRY["rf2_bucketed"](x, lora, ranks, plan=plan))
    row["b2"] = bench(lambda: b2_write(x, lora, g, out))  # last: its gathers allocate large temporaries
    return row


def add_row(lora, batch):
    x, ranks = batch.x, batch.token_ranks
    r = int(ranks.max())
    g = groups_of(ranks)
    check(x, lora, ranks, g)
    out = torch.randn(x.shape[0], lora.h_out, device="cuda", dtype=x.dtype)
    return {"max_rank": r, "mean_rank": float(ranks.float().mean()), "groups": len(g),
            "b2_add": bench(lambda: b2_add(x, lora, g, out)),
            "rf1_add": bench(lambda: rf.rankfuse_lora(x, lora.A_t, lora.B, ranks, out=out, add=True, r_limit=r))}


MIXED = [
    ("15 x 512 @r4 + 1 x 512 @r64", [512] * 16, [4] * 15 + [64]),
    ("7 x 1024 @r8 + 1 x 1024 @r64", [1024] * 8, [8] * 7 + [64]),
    ("31 x 64 @r4 + 1 x 64 @r64", [64] * 32, [4] * 31 + [64]),
    ("3 x 2048 @r16 + 1 x 2048 @r64", [2048] * 4, [16] * 3 + [64]),
    ("mix 1/2/4/8/16/32/64 x 1024", [1024] * 7, [1, 2, 4, 8, 16, 32, 64]),
]
B3_WORKLOADS = (  # checks/b3_vllm.py
    [f"decode:{t}:1:{d}" for t in (1, 8, 32, 128) for d in ("uniform", "skew_low", "all_max")]
    + [f"decode:{t}:1:uniform" for t in (512, 2048, 8192)]
    + ["prompt:4:512:uniform", "prompt:16:512:skew_low", "prompt:4:2048:uniform", "mixed:16x512:r4+r64"]
)


def b3_setup(label):
    if label.startswith("mixed"):
        return custom_batch([512] * 16, [4] * 15 + [64])
    _, a, b, dist = label.split(":")
    return workload(int(a), int(b), dist)


def show(tag, row, keys):
    cells = "  ".join(f"{k} {row[k]:7.1f}" for k in keys if k in row)
    print(f"  {tag:<34} groups {row['groups']}  {cells}", flush=True)


def main(only=None):
    print(torch.cuda.get_device_name(), "| torch", torch.__version__, "| triton", triton.__version__, flush=True)
    res = {"one_layer_write": {}, "one_layer_add": {}, "vs_b3_add": [], "rf2_sorted": [], "rf2_segment_batches": []}

    if only == "seg":
        return seg_section(res)
    print("\n[table 01] one layer, write mode (rf1_final_sweep batches)")
    for dist in ("uniform", "skew_low"):
        for t in [1, 8, 32, 128, 256, 512, 1024, 2048, 4096, 8192]:
            row = write_row(*workload(t, 1, dist))
            res["one_layer_write"].setdefault(dist, {})[t] = row
            show(f"{dist} T={t}", row, ["b2", "rf1"])

    print("\n[table 02] one layer, add mode")
    for t in [512, 2048, 4096, 8192]:
        row = add_row(*workload(t, 1, "uniform"))
        res["one_layer_add"][t] = row
        show(f"uniform T={t}", row, ["b2_add", "rf1_add"])

    print("\n[table 03] the B3 comparison batches, add mode")
    for label in B3_WORKLOADS:
        row = add_row(*b3_setup(label))
        row["workload"] = label
        res["vs_b3_add"].append(row)
        show(label, row, ["b2_add", "rf1_add"])

    print("\n[table 07] the sorted RF2 batches (rf2_splitk_v2.py)")
    shapes = ([((t, 1), d) for t in (1, 8, 32, 128) for d in ("uniform", "skew_low", "skew_high")]
              + [((t, 1), d) for t in (512, 2048, 8192) for d in ("uniform", "skew_low")]
              + [(p, d) for p in [(1, 2048), (4, 512), (16, 128), (4, 2048), (8, 1024), (16, 512)]
                 for d in ("uniform", "skew_low", "skew_high")])
    for (req, tpr), dist in shapes:
        row = write_row(*workload(req, tpr, dist), rf2=True)
        row.update(requests=req, tokens_per_request=tpr, dist=dist)
        res["rf2_sorted"].append(row)
        show(f"{req} x {tpr} {dist}", row, ["b2", "rf1", "rf2_splitk"])

    seg_section(res)


def seg_section(res, out=None):
    print("\n[tables 08/09] the segment / bucketed RF2 batches")
    batches = [(label, *custom_batch(lens, rks)) for label, lens, rks in MIXED]
    batches += [(f"{req} x {tpr} {d}", *workload(req, tpr, d)) for (req, tpr), d in
                itertools.product([(16, 512), (32, 256), (64, 128), (16, 128), (8, 1024)], ["skew_low", "uniform"])]
    batches += [(f"{t} x 1 {d}", *workload(t, 1, d)) for t, d in
                itertools.product([32, 128, 2048, 8192], ["skew_low", "uniform"])]
    for label, lora, batch in batches:
        row = write_row(lora, batch, rf2=True)
        row["batch"] = label
        res["rf2_segment_batches"].append(row)
        show(label, row, ["b2", "rf1", "rf2_splitk", "rf2_segment", "rf2_bucketed"])

    out = sys.argv[-1] if len(sys.argv) > 1 and sys.argv[-1].endswith(".json") else "b2_grouped.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nwrote {out}")


def ncu_mode(t):
    lora, batch = workload(t, 1, "uniform")
    x, ranks = batch.x, batch.token_ranks
    g = groups_of(ranks)
    out = torch.randn(t, 4096, device="cuda", dtype=torch.float16)
    for _ in range(2):
        b2_add(x, lora, g, out)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    b2_add(x, lora, g, out)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ncu":
        ncu_mode(int(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "seg":
        main("seg")   # only the tables 08/09 batches:  python b2_grouped_runs.py seg out.json
    else:
        main()
