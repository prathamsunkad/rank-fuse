"""Whole-model forward time of Qwen2.5-0.5B for each LoRA backend (MatLoRA plan, step A3).

Uses bench/model_harness.py: all 168 projections wrapped, backends none / b1 / b3 / rf1 / rf2seg /
rf2bkt / rf2auto. Check every backend first with `python bench/model_harness.py check`.

Workloads (each request has one rank; per-token ranks are set once per forward, outside timing):
    decode   B requests, one new token each, on a KV cache pre-filled from a 128-token prompt
             (B = 1, 8, 32, 128)
    prefill  4 x 512 and 16 x 128 prompt tokens, no cache
Rank mixes: all64 (every request rank 64), uniform (over 1, 2, 4, ..., 64), mostlylow (skew_low),
mostlylow+64 (skew_low with request 0 at rank 64).

Timing:
    graph    the forward captured in a CUDA graph and replayed (GPU time only; median of 7 x 20)
    eager    the same forward run normally, wall time with a sync (median of 10), which includes
             Python and kernel-launch overhead (rf2auto never picks rf2bkt here)
A capture that fails is reported and skipped. The % after each time is the LoRA share of the
step: (time - none) / time.

    python bench/model_forward.py [out.json]     # repo root, inside a GPU job

Measured on an H200 (fp16): graph replay, rf1 is 1.38-1.47x faster than b1 and 1.12-1.31x faster
than b3 at decode, 1.18-1.30x at prefill; rf2bkt 0.95-1.01x of rf1 at this model size. Eager,
rf1 is 0.80-0.85x of b1: Python launch overhead per call (~108 us vs b1's ~71) dominates.
"""

import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch

import model_harness as mh  # also puts the repo root on sys.path
from rankfuse.workloads import DISTRIBUTIONS

BACKENDS = ["none", "b1", "b3", "rf1", "rf2seg", "rf2bkt", "rf2auto"]
MIXES = ["all64", "uniform", "mostlylow", "mostlylow+64"]
TIERS = mh.TIERS


def ranks_for(mix, n, seed=0):
    rng = random.Random(seed)
    if mix == "all64":
        return [64] * n
    probs = DISTRIBUTIONS["uniform" if mix == "uniform" else "skew_low"](len(TIERS))
    ranks = rng.choices(TIERS, weights=probs, k=n)
    if mix == "mostlylow+64":
        ranks[0] = 64
    return ranks


def graph_time(fn, replays=20, runs=7):
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
        times.append(a.elapsed_time(b) / replays)
    del g
    return statistics.median(times)


def eager_time(fn, runs=10):
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1e3)
    return statistics.median(times)


def run_workload(model, ctx, kind, batch, tokens, mix):
    gen = torch.Generator().manual_seed(1)
    ranks = ranks_for(mix, batch)
    row = {"kind": kind, "batch": batch, "tokens": tokens, "mix": mix, "max_rank": max(ranks),
           "mean_rank": sum(ranks) / len(ranks), "graph_ms": {}, "eager_ms": {}, "errors": {}}
    with torch.no_grad():
        if kind == "prefill":
            ids = torch.randint(1000, 30000, (batch, tokens), generator=gen).cuda()
        else:
            prompt = torch.randint(1000, 30000, (batch, 128), generator=gen).cuda()
            step = torch.randint(1000, 30000, (batch, 1), generator=gen).cuda()
        for backend in BACKENDS:
            ctx.backend = backend
            try:
                ctx.graphed = False
                if kind == "prefill":
                    ctx.set_batch(ranks, tokens)
                    fn = lambda: model(input_ids=ids, use_cache=False).logits
                else:
                    ctx.set_batch(ranks, 128)
                    cache = model(input_ids=prompt, use_cache=True).past_key_values
                    ctx.set_batch(ranks, 1)

                    def fn():
                        return model(input_ids=step, past_key_values=cache, use_cache=True).logits
                row["eager_ms"][backend] = eager_time(fn)
                ctx.graphed = True
                if kind == "prefill":
                    ctx.set_batch(ranks, tokens)
                else:  # eager runs grew the cache; rebuild it before capture
                    ctx.set_batch(ranks, 128)
                    cache = model(input_ids=prompt, use_cache=True).past_key_values
                    ctx.set_batch(ranks, 1)
                row["graph_ms"][backend] = graph_time(fn)
            except Exception as e:
                row["errors"][backend] = f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
            torch.cuda.empty_cache()
    return row


def fmt(row, key):
    t = row[key]
    base = t.get("none")
    cells = []
    for b in BACKENDS:
        if b not in t:
            cells.append(f"{'err':>13}")
        elif b == "none" or base is None:
            cells.append(f"{t[b]:>13.3f}")
        else:
            cells.append(f"{t[b]:>7.3f} ({(t[b] - base) / t[b] * 100:3.0f}%)")
    return "".join(cells)


def main():
    print(torch.cuda.get_device_name(), flush=True)
    model, ctx = mh.load_model()
    workloads = [("decode", b, 1) for b in (1, 8, 32, 128)] + [("prefill", 4, 512), ("prefill", 16, 128)]
    rows = []
    for kind, batch, tokens in workloads:
        for mix in MIXES:
            row = run_workload(model, ctx, kind, batch, tokens, mix)
            rows.append(row)
            label = f"{kind} {batch}x{tokens} {mix}"
            print(f"{label:<30} graph: {fmt(row, 'graph_ms')}", flush=True)
            for b, e in row["errors"].items():
                print(f"    {b}: {e}", flush=True)
    for key, title in (("graph_ms", "CUDA graph replay, GPU time per forward (ms); % = LoRA share of the step"),
                       ("eager_ms", "eager, wall time per forward (ms); % = LoRA share of the step")):
        print(f"\n=== {title} ===")
        print(f"{'workload':<30}" + "".join(f"{b:>13}" for b in BACKENDS)
              + f"{'rf1/b1':>9}{'rf1/b3':>9}{'bkt/rf1':>9}{'auto/rf1':>10}{'auto/b3':>9}")
        for row in rows:
            t = row[key]
            extra = ""
            if all(b in t for b in ("b1", "b3", "rf1", "rf2bkt", "rf2auto")):
                extra = (f"{t['b1'] / t['rf1']:>8.2f}x{t['b3'] / t['rf1']:>8.2f}x{t['rf1'] / t['rf2bkt']:>8.2f}x"
                         f"{t['rf1'] / t['rf2auto']:>9.2f}x{t['b3'] / t['rf2auto']:>8.2f}x")
            print(f"{row['kind'] + ' ' + str(row['batch']) + 'x' + str(row['tokens']) + ' ' + row['mix']:<30}{fmt(row, key)}{extra}")
    out = sys.argv[1] if len(sys.argv) > 1 else "results/raw/model_forward.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
