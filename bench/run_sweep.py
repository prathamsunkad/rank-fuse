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

`mode: add` (config, default write) has each call add its delta into an existing output, as in a
model; the implementations listed must take out= and add=. `prepare_us` is the time for the
per-step metadata, kept out of median_us, so it can be charged once per step. An implementation
listed twice gets impl_pass 0 and 1 on the same workload, which measures noise between runs.
--resume (with --out) keeps finished workloads in the CSV and reruns the rest, so a preempted job
can continue; an unfinished workload starts over, so one workload's rows share a GPU.
`prepared: true` with eager timing passes the per-step metadata in, so eager time minus graph time
on the same workload is launch overhead and not also a host sync or sort.
"""

from __future__ import annotations

import argparse
import csv
import functools
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
from rankfuse.workloads import Workload, is_known_dist, make_batch, make_lora  # noqa: E402

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
    if flush_graph is None:
        return [per_replay_us(graph) for _ in range(reps)]
    # The flush is timed right before each repetition and subtracted from it: repeats differed by
    # up to 6.4% with one flush median taken up front, 2.5% this way (bench/flush_timing_check.py).
    times = []
    for _ in range(reps):
        flush_us = per_replay_us(flush_graph)
        times.append(per_replay_us(graph) - flush_us)
    return times


def expand_grid(grid: dict, shapes: dict | None = None, batches: list | None = None) -> list[tuple[str, Workload]]:
    """Every combination of the grid values, as (shape name, workload).

    `shapes` maps a projection name to [h_in, h_out] (e.g. a model's q/kv/gate_up/down) and
    replaces the grid's h_in and h_out; without it, the shape name is "h_in x h_out".
    `batches` lists [num_requests, tokens_per_request] pairs and replaces those two grid keys,
    for batch shapes that are not a cross product (e.g. 4 x 512 and 16 x 128 but not 4 x 128).
    """
    named = [(name, {"h_in": h_in, "h_out": h_out}) for name, (h_in, h_out) in (shapes or {}).items()]
    pairs = [{"num_requests": n, "tokens_per_request": t} for n, t in batches] if batches else [{}]
    if batches and {"num_requests", "tokens_per_request"} & set(grid):
        raise ValueError("give the batch shape either as `batches` or in the grid, not both")
    keys = list(grid)
    values = [v if isinstance(v, list) else [v] for v in grid.values()]
    out = []
    for pair, combo in itertools.product(pairs, itertools.product(*values)):
        params = {**pair, **dict(zip(keys, combo))}
        for name, dims in named or [(None, {})]:
            w = Workload(**{**params, **dims})
            out.append((name or f"{w.h_in}x{w.h_out}", w))
    return out


def workload_key(shape: str, w: Workload) -> tuple[str, ...]:
    return (shape, *("" if v is None else str(v) for v in w.as_row().values()))  # as csv writes them


def read_finished(path: Path, num_rows: int, keys: set[tuple[str, ...]]) -> tuple[list[dict], set[tuple[str, ...]]]:
    """Rows of workloads that have all num_rows rows in the CSV, and those workloads' keys."""
    if not path.exists() or path.stat().st_size == 0:
        return [], set()
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return [], set()
    columns = list(Workload(1, 1).as_row())
    key = lambda r: (r["shape"], *(r[c] for c in columns))  # noqa: E731
    count: dict[tuple[str, ...], int] = {}
    for r in rows:
        count[key(r)] = count.get(key(r), 0) + 1
    finished = {k for k, n in count.items() if n >= num_rows and k in keys}
    return [r for r in rows if key(r) in finished], finished


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--out", default=None, help="CSV path (default: results/raw/<name>-<gpu>-<time>.csv)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--timing", choices=["eager", "graph"], default=None, help="see above (default: config, else eager)")
    p.add_argument("--resume", action="store_true", help="see above; needs --out")
    a = p.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    timing = a.timing or cfg.get("timing", "eager")
    mode = cfg.get("mode", "write")
    if a.resume and not a.out:
        sys.exit("--resume needs --out")
    if timing == "graph" and a.device != "cuda":
        sys.exit("--timing graph needs CUDA")
    impls = sorted(n for n in REGISTRY if n != "b0_reference") if cfg.get("impls", "all") == "all" else cfg["impls"]
    impls = [n for n in impls if not (REGISTRY[n].requires_cuda and a.device != "cuda")]
    workloads = expand_grid(cfg["grid"], cfg.get("shapes"), cfg.get("batches"))
    unknown = sorted({w.dist for _, w in workloads if not is_known_dist(w.dist)})
    if unknown:
        sys.exit(f"unknown rank mixes in {a.config}: {unknown}")

    gname = gpu_name()
    gkey = roofline_gpu_key(gname)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(a.out or f"results/raw/{cfg['name']}-{gname.replace(' ', '_')}-{stamp}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    sha = git_sha()
    print(f"{len(workloads)} workloads x {len(impls)} impls on {gname} (roofline: {gkey}, timing: {timing}, mode: {mode}) -> {out}")

    done_rows, done = [], set()
    if a.resume:
        done_rows, done = read_finished(out, len(impls), {workload_key(s, w) for s, w in workloads})
        print(f"resuming: {len(done)} workloads already finished")
    with out.open("w", newline="") as f:
        writer = None
        if done_rows:
            writer = csv.DictWriter(f, fieldnames=list(done_rows[0]))
            writer.writeheader()
            writer.writerows(done_rows)
            f.flush()
        for shape, w in workloads:
            if workload_key(shape, w) in done:
                continue
            lora, batch = make_lora(w, a.device), make_batch(w, a.device)
            token_ranks = batch.token_ranks.tolist()
            passes: dict[str, int] = {}
            for name in impls:
                impl = REGISTRY[name]
                passes[name] = passes.get(name, -1) + 1
                fn = impl
                if mode == "add":
                    target = torch.zeros(batch.x.shape[0], w.h_out, dtype=batch.x.dtype, device=a.device)
                    fn = functools.partial(impl, out=target, add=True)
                if impl.dtypes is not None and batch.x.dtype not in impl.dtypes:
                    continue
                args = (batch.x, lora, batch.token_ranks)
                warmup, reps, flush_l2 = cfg.get("warmup", 10), cfg.get("reps", 50), cfg.get("flush_l2", True)
                used = timing
                prepare_us = ""
                if timing == "graph" and name.endswith("_graphed"):
                    used = "eager (graphs itself)"  # replays its own graph; cannot be nested in another
                elif timing == "graph":
                    try:
                        if a.device == "cuda":
                            torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        kwargs = impl.prepare(*args) if impl.prepare else {}
                        if a.device == "cuda":
                            torch.cuda.synchronize()
                        prepare_us = (time.perf_counter() - t0) * 1e6
                        times = time_graph(fn, args, kwargs, warmup, reps, cfg.get("graph_replays", 20), flush_l2)
                    except Exception as e:  # not capturable: time it eagerly instead
                        torch.cuda.synchronize()
                        used = "eager (capture failed)"
                        print(f"  {name}: graph capture failed ({type(e).__name__}); timing eagerly")
                if used != "graph":
                    if cfg.get("prepared") and impl.prepare:
                        fn = functools.partial(fn, **impl.prepare(*args))
                    times = time_impl(fn, args, a.device, warmup, reps, flush_l2)
                q = statistics.quantiles(times, n=10)
                strategy = ROOFLINE_STRATEGY.get(name, DEFAULT_STRATEGY)
                c = roofline.cost(strategy, token_ranks, w.h_in, w.h_out, w.r_max, w.dtype)
                row = {
                    "impl": name, "impl_pass": passes[name], "gpu": gname, "git_sha": sha, "timing": used, "mode": mode,
                    "shape": shape, **w.as_row(),
                    "batch_max_rank": max(token_ranks), "mean_rank": sum(token_ranks) / len(token_ranks),
                    "median_us": statistics.median(times), "p20_us": q[1], "p80_us": q[7], "prepare_us": prepare_us,
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
