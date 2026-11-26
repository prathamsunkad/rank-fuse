"""Prefix-paging timings on one GPU (target: ICE A100 80GB PCIe).

    python bench/paging_timing.py --out results/a100_2026-11-23_paging/source [--quick]

Three experiments, each written as JSON + CSV under --out:

  h2d       pinned host -> device copy time and bandwidth:
              * a size sweep (4 KiB .. 1 GiB, one cudaMemcpyAsync) to fit t(n) = t0 + n / BW;
              * real sizes: one whole-model page (Qwen2.5-0.5B 17.6 MB, 7B 80.7 MB) and whole
                adapters at r 64 / 128 / 256 as one contiguous copy, as r/16 page copies through
                ModelPageManager into scattered frames, and as vLLM-style per-layer, per-module,
                per-matrix copies (2 x 7 x layers calls of r rows each).
  overlap   does a concurrent H2D copy slow the RF1-MA kernels? One decode step's LoRA work over
            every (layer, group) of the model (one CUDA graph: 4 merged groups x layers RF1-MA
            calls on the whole-model pool) timed alone and while a copy stream streams pages; and
            the copy bandwidth while the graph runs.
  ready     time until a rank-r request on a cold adapter can run: ensure({a: r}) to copy_done,
            prefix pages (ceil(r/16) pages) vs the whole adapter at r_max (what a whole-adapter
            policy loads), for r_max = 256 and r = 16 .. 256; plus the host cost of ensure() when
            everything is resident.

Timing: CUDA events, median and p95 over repeats after warm-up. Host buffers are pinned.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from rankfuse.multi import build_meta, tile_capacity
from rankfuse.paging.model_pages import ModelPageLayout, ModelPageManager, ModelSpec

MODELS = ("0.5B", "7B")
RANKS = (64, 128, 256)


def _stats(ms: list[float]) -> dict:
    return {"median_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95)),
            "min_ms": float(np.min(ms)), "n": len(ms)}


def _time(fn, repeats: int, warmup: int = 3, stream=None) -> list[float]:
    stream = stream or torch.cuda.current_stream()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        s.record(stream)
        fn()
        e.record(stream)
        e.synchronize()
        out.append(s.elapsed_time(e))
    return out


def _write(out: Path, name: str, rows: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(rows, indent=1))
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with (out / f"{name}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out / name}.json/.csv ({len(rows)} rows)", flush=True)


# ------------------------------------------------------------------------------------------------
# h2d
# ------------------------------------------------------------------------------------------------


def exp_h2d(out: Path, quick: bool) -> None:
    rows = []
    # large enough for the biggest real size (Qwen2.5-7B at r 256: 16 pages, 1.29 GB)
    big = max(1 << 30, 16 * ModelPageLayout(ModelSpec.qwen25("7B")).page_bytes())
    host = torch.empty(big, dtype=torch.uint8, pin_memory=True)
    dev = torch.empty(big, dtype=torch.uint8, device="cuda")
    sizes = [1 << k for k in range(12, 31)]
    for n in sizes:
        reps = 10 if n >= 1 << 28 else 30
        ms = _time(lambda: dev[:n].copy_(host[:n], non_blocking=True), reps if not quick else 5)
        st = _stats(ms)
        rows.append({"kind": "sweep", "label": f"{n} B", "bytes": n, "calls": 1, **st,
                     "gb_s": n / (st["median_ms"] * 1e6)})
        print(f"  sweep {n:>11d} B  {st['median_ms']:.4f} ms  {rows[-1]['gb_s']:.2f} GB/s", flush=True)
    # fit t = t0 + n / BW on sizes >= 64 KiB
    fit = [r for r in rows if r["bytes"] >= 1 << 16]
    A = np.stack([np.ones(len(fit)), [r["bytes"] for r in fit]], 1)
    t0, inv = np.linalg.lstsq(A, np.array([r["median_ms"] for r in fit]), rcond=None)[0]
    print(f"  fit: t0 = {t0 * 1e3:.2f} us, BW = {1 / inv / 1e6:.2f} GB/s", flush=True)

    for size in MODELS:
        L = ModelPageLayout(ModelSpec.qwen25(size))
        pb = L.page_bytes()
        rb = pb // 16
        # one page, and whole adapters as one contiguous copy
        for label, n in [("1 page", pb)] + [(f"adapter r{r} (1 copy)", (r // 16) * pb) for r in RANKS]:
            ms = _time(lambda: dev[:n].copy_(host[:n], non_blocking=True), 20 if not quick else 3)
            st = _stats(ms)
            rows.append({"kind": "real", "model": size, "label": label, "bytes": n, "calls": 1, **st,
                         "gb_s": n / (st["median_ms"] * 1e6)})
            print(f"  {size} {label:24s} {n / 1e6:8.1f} MB  {st['median_ms']:.3f} ms  {rows[-1]['gb_s']:.2f} GB/s",
                  flush=True)
        # vLLM-style: per layer, per module, per matrix, r rows each (true rank copied into the slot)
        spec = L.spec
        mods = []
        for g in ModelSpec.qwen25(size, merged=False).groups:
            mods += [g.h_in * 2, g.slice_sizes[0] * 2]  # bytes per rank row of A and of B
        for r in RANKS:
            chunks = [r * m for m in mods] * spec.num_layers
            total = sum(chunks)
            offs = np.cumsum([0] + chunks[:-1]).tolist()

            def many():
                for o, c in zip(offs, chunks):
                    dev[o:o + c].copy_(host[o:o + c], non_blocking=True)

            ms = _time(many, 10 if not quick else 2)
            st = _stats(ms)
            rows.append({"kind": "real", "model": size, "label": f"adapter r{r} (vLLM-style {len(chunks)} copies)",
                         "bytes": total, "calls": len(chunks), **st, "gb_s": total / (st["median_ms"] * 1e6)})
            print(f"  {size} r{r} vLLM-style {len(chunks)} copies {total / 1e6:.1f} MB {st['median_ms']:.3f} ms",
                  flush=True)
    del host, dev
    torch.cuda.empty_cache()

    # through ModelPageManager: r/16 page copies into scattered frames + one page-table copy
    for size in MODELS:
        L = ModelPageLayout(ModelSpec.qwen25(size))
        frames = 16 + 4
        mgr = ModelPageManager(L, frames, num_adapter_slots=1, r_max=256)
        hostp = torch.zeros(16, L.frame_elems, dtype=torch.float16).pin_memory()
        mgr.register(0, hostp)
        for r in RANKS:
            ms = []
            for i in range(6 if not quick else 2):
                torch.cuda.synchronize()
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record(mgr.copy_stream)
                mgr.begin_step([0])
                mgr.ensure({0: r})
                e.record(mgr.copy_stream)
                mgr.end_step()
                e.synchronize()
                ms.append(s.elapsed_time(e))
                mgr.evict(0)
            ms = ms[1:]
            st = _stats(ms)
            n = (r // 16) * L.page_bytes()
            rows.append({"kind": "manager", "model": size, "label": f"ensure r{r} ({r // 16} page copies)",
                         "bytes": n, "calls": r // 16, **st, "gb_s": n / (st["median_ms"] * 1e6)})
            print(f"  {size} manager r{r}: {st['median_ms']:.3f} ms {rows[-1]['gb_s']:.2f} GB/s", flush=True)
        del mgr, hostp
        torch.cuda.empty_cache()
    rows.append({"kind": "fit", "label": "t = t0 + n/BW (n >= 64 KiB)", "t0_us": float(t0 * 1e3),
                 "bw_gb_s": float(1 / inv / 1e6)})
    _write(out, "h2d", rows)


# ------------------------------------------------------------------------------------------------
# overlap
# ------------------------------------------------------------------------------------------------


def _step_graph(L: ModelPageLayout, mgr: ModelPageManager, T: int, n_adapters: int, seed: int):
    """CUDA graph of one decode step's LoRA deltas on every (layer, group), RF1-MA."""
    gen = torch.Generator().manual_seed(seed)
    ad = torch.randint(0, n_adapters, (T,), generator=gen)
    tiers = torch.tensor([16, 32, 64, 128])
    rk = tiers[torch.randint(0, 4, (T,), generator=gen)]
    meta = build_meta(ad, rk, num_adapters=mgr.num_adapter_slots, r_pad=128,
                      capacity=tile_capacity(T, 16, mgr.num_adapter_slots), bm=16)
    tr = rk.int().cuda()
    keys = list(L.blocks)
    xs = {k: torch.randn(T, L.blocks[k].h_in, generator=gen).half().cuda() for k in keys}
    outs = {k: torch.empty(T, sum(L.blocks[k].slice_sizes), dtype=torch.float16, device="cuda") for k in keys}
    from rankfuse.kernels.rankfuse_multi import multi_lora

    def fn():
        for k in keys:
            multi_lora(xs[k], mgr.weights(*k), tr, meta, out=outs[k])

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def exp_overlap(out: Path, quick: bool) -> None:
    rows = []
    for size in MODELS:
        L = ModelPageLayout(ModelSpec.qwen25(size))
        n_ad = 32
        frames = n_ad * 8  # every adapter resident at r 128
        if size == "7B":
            n_ad, frames = 16, 16 * 8  # 128 x 80.7 MB = 10.3 GB
        mgr = ModelPageManager(L, frames, num_adapter_slots=n_ad, r_max=128)
        hostp = (torch.randn(8, L.frame_elems) * 0.02).half().pin_memory()
        for a in range(n_ad):
            mgr.register(a, hostp)
        mgr.begin_step(range(n_ad))
        mgr.ensure({a: 128 for a in range(n_ad)})
        mgr.wait_for_copies()
        mgr.end_step()
        torch.cuda.synchronize()
        # copy source/target outside the pool: pages streamed by a copy stream
        pb = L.page_bytes()
        n_pages_stream = 8
        src = torch.empty(n_pages_stream * pb, dtype=torch.uint8).pin_memory()
        dst = torch.empty(n_pages_stream * pb, dtype=torch.uint8, device="cuda")
        cs = torch.cuda.Stream()
        for T in (16, 64, 256):
            g = _step_graph(L, mgr, T, n_ad, seed=T)
            reps = 50 if not quick else 5
            alone = _stats(_time(g.replay, reps, warmup=5))
            # with copies: keep the copy stream busy for the whole timed window
            win_ms = alone["median_ms"] * reps
            copies_needed = max(16, int(math.ceil(3.0 * win_ms / (pb / 80e6))))  # outlasts the window even at 80 GB/s
            torch.cuda.synchronize()
            cs_s, cs_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            # whole 8-page chunks per call: ~100s of queued copies, not 1000s (a long queue of small
            # copies blocks the host while enqueuing, so the steps would start after the copies)
            n_calls = -(-copies_needed // n_pages_stream)
            copies_needed = n_calls * n_pages_stream
            with torch.cuda.stream(cs):
                cs_s.record(cs)
                for _ in range(n_calls):
                    dst.copy_(src, non_blocking=True)
                cs_e.record(cs)
            torch.cuda._sleep(1_000_000)  # let the copy stream start first
            busy = []
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for _ in range(reps):
                s.record()
                g.replay()
                e.record()
                e.synchronize()
                busy.append(s.elapsed_time(e))
            still_copying = not cs_e.query()
            cs_e.synchronize()
            copy_ms = cs_s.elapsed_time(cs_e)
            # copy bandwidth alone for the same chain
            torch.cuda.synchronize()
            with torch.cuda.stream(cs):
                cs_s.record(cs)
                for _ in range(n_calls):
                    dst.copy_(src, non_blocking=True)
                cs_e.record(cs)
            cs_e.synchronize()
            copy_alone_ms = cs_s.elapsed_time(cs_e)
            b = _stats(busy)
            row = {"model": size, "tokens": T, "adapters": n_ad, "launches_per_step": len(L.blocks) * 2,
                   "step_alone_median_ms": alone["median_ms"], "step_alone_p95_ms": alone["p95_ms"],
                   "step_with_copy_median_ms": b["median_ms"], "step_with_copy_p95_ms": b["p95_ms"],
                   "slowdown": b["median_ms"] / alone["median_ms"],
                   "copy_covered_window": still_copying,
                   "copy_bytes": copies_needed * pb,
                   "copy_gb_s_with_steps": copies_needed * pb / (copy_ms * 1e6),
                   "copy_gb_s_alone": copies_needed * pb / (copy_alone_ms * 1e6)}
            rows.append(row)
            print(f"  {size} T={T}: step {alone['median_ms']:.3f} -> {b['median_ms']:.3f} ms "
                  f"(x{row['slowdown']:.3f}, copy covered window: {still_copying}); copy "
                  f"{row['copy_gb_s_alone']:.1f} -> {row['copy_gb_s_with_steps']:.1f} GB/s", flush=True)
            del g
        del mgr, hostp, src, dst
        torch.cuda.empty_cache()
    _write(out, "overlap", rows)


# ------------------------------------------------------------------------------------------------
# ready
# ------------------------------------------------------------------------------------------------


def exp_ready(out: Path, quick: bool) -> None:
    rows = []
    for size in MODELS:
        L = ModelPageLayout(ModelSpec.qwen25(size))
        r_max = 256
        mgr = ModelPageManager(L, 16 + 2, num_adapter_slots=2, r_max=r_max)
        hostp = torch.zeros(16, L.frame_elems, dtype=torch.float16).pin_memory()
        mgr.register(0, hostp)
        for _ in range(3):  # warm-up: first DMA from freshly pinned pages, clocks
            mgr.begin_step([0]); mgr.ensure({0: r_max}); mgr.end_step()
            torch.cuda.synchronize()
            mgr.evict(0)
        torch.cuda.synchronize()
        for r in (16, 32, 64, 128, 256):
            for mode, want in (("prefix", r), ("whole", r_max)):
                ms_dev, ms_wall = [], []
                for i in range(9 if not quick else 2):
                    if mgr.resident_pages(0):
                        mgr.evict(0)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    mgr.begin_step([0])
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record(mgr.copy_stream)
                    mgr.ensure({0: want})
                    e.record(mgr.copy_stream)
                    mgr.copy_done.synchronize()
                    t1 = time.perf_counter()
                    mgr.end_step()
                    e.synchronize()
                    ms_dev.append(s.elapsed_time(e))
                    ms_wall.append((t1 - t0) * 1e3)
                rows.append({"model": size, "r_max": r_max, "rank": r, "mode": mode, "pages": math.ceil(want / 16),
                             "bytes": math.ceil(want / 16) * L.page_bytes(),
                             "ready_wall_median_ms": float(np.median(ms_wall[1:] or ms_wall)), "ready_wall_min_ms": float(np.min(ms_wall)),
                             "ready_dev_median_ms": float(np.median(ms_dev[1:] or ms_dev))})
                print(f"  {size} r{r} {mode}: {rows[-1]['pages']} pages, ready after "
                      f"{rows[-1]['ready_wall_median_ms']:.2f} ms", flush=True)
        # host cost of ensure() when every page is resident (the per-step fast path)
        mgr.ensure({0: r_max})
        mgr.end_step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n = 2000
        for _ in range(n):
            mgr.begin_step([0])
            mgr.ensure({0: 64})
            mgr.end_step()
        rows.append({"model": size, "mode": "resident_fast_path", "ensure_host_us": (time.perf_counter() - t0) / n * 1e6})
        print(f"  {size} ensure() fast path: {rows[-1]['ensure_host_us']:.1f} us", flush=True)
        del mgr, hostp
        torch.cuda.empty_cache()
    _write(out, "ready", rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--only", nargs="*", default=["h2d", "overlap", "ready"])
    p.add_argument("--quick", action="store_true", help="few repeats (to test the script)")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = {
        "device": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "host": platform.node(), "slurm_job": os.environ.get("SLURM_JOB_ID"), "quick": args.quick,
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    try:
        import triton
        env["triton"] = triton.__version__
        env["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,pcie.link.gen.max,pcie.link.width.max,clocks.max.sm",
             "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        env["env_error"] = repr(exc)
    print(env, flush=True)
    for name in args.only:
        print(f"== {name}", flush=True)
        {"h2d": exp_h2d, "overlap": exp_overlap, "ready": exp_ready}[name](out, args.quick)
    env["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    (out / f"env{'_quick' if args.quick else ''}.json").write_text(json.dumps(env, indent=1))


if __name__ == "__main__":
    main()
