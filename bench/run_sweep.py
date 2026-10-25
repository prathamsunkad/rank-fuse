"""Benchmark every requested implementation over a grid of workloads; append rows to a CSV.

    python bench/run_sweep.py bench/configs/smoke.yaml
    python bench/run_sweep.py bench/configs/decode.yaml --out results/raw/decode.csv
    python bench/run_sweep.py bench/configs/decode.yaml --timing graph

Timing (`--timing`, or `timing:` in the config; default eager):
    eager  CUDA events around each call (CPU: perf_counter). Includes the Python and launch
           overhead the GPU waits through (~70-110 us per call on ICE nodes for a Triton
           wrapper), which dominates decode-sized calls.
    graph  GPU time only. The call is captured once in a CUDA graph with the per-step metadata
           the implementation's `prepare` builds outside it (batch maximum rank, rank order,
           block table, ...), as a serving engine would; each repetition replays it
           `graph_replays` times back to back. An implementation that cannot be captured (host
           sync inside, no `prepare`) falls back to eager timing, and *_graphed baselines,
           which replay their own graph, are timed eagerly; the `timing` column says which
           was used.
An optional L2 flush before each call keeps adapter weights cold, as in a real forward pass;
in graph mode the flush runs inside the graph and its own time is subtracted.
Each row also carries the roofline lower bound for the matching strategy.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rankfuse  # noqa: E402  (registers implementations)
from rankfuse import roofline  # noqa: E402
from rankfuse.interface import REGISTRY  # noqa: E402
from rankfuse.workloads import Workload, make_batch, make_lora  # noqa: E402

# Which roofline strategy each implementation should be compared against.
ROOFLINE_STRATEGY = {
    "b1_padded": "padded",
    "b2_grouped": "grouped",
    "b2_grouped_graphed": "grouped",
    "b3_vllm": "separate_padded",
    "b4_slora": "separate_true",
}
DEFAULT_STRATEGY = "fused_ideal"  # RankFuse kernels

L2_FLUSH_BYTES = 256 * 2**20


def gpu_name() -> str:
    return torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"


def roofline_gpu_key(name: str) -> str | None:
    n = name.upper()
    for key in roofline.GPUS:
        model, *variant = key.upper().split("-")
        if model in n and all(v.replace("GB", "") in n for v in variant):
            return key
    for key in roofline.GPUS:
        if key.split("-")[0].upper() in n:
            return key
    return None


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def time_impl(fn, args, device: str, warmup: int, reps: int, flush_l2: bool) -> list[float]:
    """Return per-repetition times in microseconds."""
    for _ in range(warmup):
        fn(*args)
    times = []
    if device == "cuda":
        flush = torch.empty(L2_FLUSH_BYTES, dtype=torch.uint8, device="cuda") if flush_l2 else None
        torch.cuda.synchronize()
        for _ in range(reps):
            if flush is not None:
                flush.zero_()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn(*args)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1e3)
    else:
        for _ in range(reps):
            t0 = time.perf_counter()
            fn(*args)
            times.append((time.perf_counter() - t0) * 1e6)
    return times


def time_graph(fn, args, kwargs, warmup: int, reps: int, replays: int, flush_l2: bool) -> list[float]:
    """Per-repetition GPU time in microseconds of fn(*args, **kwargs) by CUDA-graph replay.

    Raises if the call cannot be captured (e.g. it reads a value back to the host).
    """
    flush = torch.empty(L2_FLUSH_BYTES, dtype=torch.uint8, device="cuda") if flush_l2 else None

    def capture(body):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # warm up (compiles kernels, fills caches) off the capture
            for _ in range(warmup):
                body()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        return graph

    def per_replay_us(graph) -> float:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1e3 / replays

    def call():
        if flush is not None:
            flush.zero_()
        fn(*args, **kwargs)

    graph = capture(call)
    flush_graph = capture(flush.zero_) if flush is not None else None
    torch.cuda.synchronize()
    flush_us = statistics.median(per_replay_us(flush_graph) for _ in range(5)) if flush_graph else 0.0
    return [per_replay_us(graph) - flush_us for _ in range(reps)]


def expand_grid(grid: dict, shapes: dict | None = None) -> list[tuple[str, Workload]]:
    """Every combination of the grid values, as (shape name, workload).

    `shapes` maps a projection name to [h_in, h_out] (e.g. a model's q/kv/gate_up/down) and
    replaces the grid's h_in and h_out; without it, the shape name is "h_in x h_out".
    """
    named = [(name, {"h_in": h_in, "h_out": h_out}) for name, (h_in, h_out) in (shapes or {}).items()]
    keys = list(grid)
    values = [v if isinstance(v, list) else [v] for v in grid.values()]
    out = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        for name, dims in named or [(None, {})]:
            w = Workload(**{**params, **dims})
            out.append((name or f"{w.h_in}x{w.h_out}", w))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--out", default=None, help="CSV path (default: results/raw/<name>-<gpu>-<time>.csv)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--timing", choices=["eager", "graph"], default=None, help="see above (default: config, else eager)")
    a = p.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    timing = a.timing or cfg.get("timing", "eager")
    if timing == "graph" and a.device != "cuda":
        sys.exit("--timing graph needs CUDA")
    impls = sorted(n for n in REGISTRY if n != "b0_reference") if cfg.get("impls", "all") == "all" else cfg["impls"]
    impls = [n for n in impls if not (REGISTRY[n].requires_cuda and a.device != "cuda")]
    workloads = expand_grid(cfg["grid"], cfg.get("shapes"))

    gname = gpu_name()
    gkey = roofline_gpu_key(gname)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(a.out or f"results/raw/{cfg['name']}-{gname.replace(' ', '_')}-{stamp}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    sha = git_sha()
    print(f"{len(workloads)} workloads x {len(impls)} impls on {gname} (roofline: {gkey}, timing: {timing}) -> {out}")

    with out.open("w", newline="") as f:
        writer = None
        for shape, w in workloads:
            lora, batch = make_lora(w, a.device), make_batch(w, a.device)
            token_ranks = batch.token_ranks.tolist()
            for name in impls:
                impl = REGISTRY[name]
                if impl.dtypes is not None and batch.x.dtype not in impl.dtypes:
                    continue
                args = (batch.x, lora, batch.token_ranks)
                warmup, reps, flush_l2 = cfg.get("warmup", 10), cfg.get("reps", 50), cfg.get("flush_l2", True)
                used = timing
                if timing == "graph" and name.endswith("_graphed"):
                    used = "eager (graphs itself)"  # replays its own graph; cannot be nested in another
                elif timing == "graph":
                    try:
                        kwargs = impl.prepare(*args) if impl.prepare else {}
                        times = time_graph(impl, args, kwargs, warmup, reps, cfg.get("graph_replays", 20), flush_l2)
                    except Exception as e:  # not capturable: time it eagerly instead
                        torch.cuda.synchronize()
                        used = "eager (capture failed)"
                        print(f"  {name}: graph capture failed ({type(e).__name__}); timing eagerly")
                if used != "graph":
                    times = time_impl(impl, args, a.device, warmup, reps, flush_l2)
                q = statistics.quantiles(times, n=10)
                strategy = ROOFLINE_STRATEGY.get(name, DEFAULT_STRATEGY)
                c = roofline.cost(strategy, token_ranks, w.h_in, w.h_out, w.r_max, w.dtype)
                row = {
                    "impl": name, "gpu": gname, "git_sha": sha, "timing": used, "shape": shape, **w.as_row(),
                    "batch_max_rank": max(token_ranks), "mean_rank": sum(token_ranks) / len(token_ranks),
                    "median_us": statistics.median(times), "p20_us": q[1], "p80_us": q[7],
                    "roofline_strategy": strategy, "roofline_bytes": c.total_bytes, "roofline_flops": c.flops,
                    "roofline_us": c.time_s(gkey) * 1e6 if gkey else "",
                }
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
                print(f"  {name:<20} {shape:<11} T={w.num_tokens:<5} r_max={w.r_max:<4} {w.dist:<9} "
                      f"{row['median_us']:9.1f} us")


if __name__ == "__main__":
    main()
