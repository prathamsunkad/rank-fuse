"""Tables and the copy-cost fit from bench/paging_timing.py output.

    python bench/paging_report.py --src results/a100_2026-11-23_paging/source \
        --tables results/a100_2026-11-23_paging/tables

Prints `--bw <GB/s> --call-us <us>` for bench/paging_sim_sweep.py:
  bw       bandwidth of the t = t0 + n / BW fit over the single-copy size sweep (n >= 64 KiB);
  call_us  extra time per cudaMemcpyAsync, from the vLLM-style per-module copies of a whole adapter
           vs the same bytes in one copy: (t_many - t_one) / (calls - 1), median over models/ranks.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--tables", required=True)
    a = p.parse_args()
    src, out = Path(a.src), Path(a.tables)
    out.mkdir(parents=True, exist_ok=True)
    env = json.loads((src / "env.json").read_text()) if (src / "env.json").exists() else {}
    dev = env.get("device", "?")

    h = json.loads((src / "h2d.json").read_text())
    fit = next(r for r in h if r["kind"] == "fit")
    sweep = [r for r in h if r["kind"] == "sweep"]
    real = [r for r in h if r["kind"] == "real"]
    mgr = [r for r in h if r["kind"] == "manager"]
    overhead = []
    for r in real:
        if "vLLM-style" in r["label"]:
            one = next(x for x in real if x["model"] == r["model"] and x["bytes"] == r["bytes"] and x["calls"] == 1)
            overhead.append((r["median_ms"] - one["median_ms"]) * 1e3 / (r["calls"] - 1))
    call_us = statistics.median(overhead) if overhead else fit["t0_us"]

    L = [f"### Pinned host-to-device copies ({dev})", "",
         f"Fit over one-copy sizes >= 64 KiB: t = {fit['t0_us']:.1f} us + n / {fit['bw_gb_s']:.2f} GB/s. "
         f"Extra cost per copy call (vLLM-style vs one copy, median): {call_us:.1f} us.", "",
         "| model | what | MB | copies | median ms | p95 ms | GB/s |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in real + mgr:
        L.append(f"| {r['model']} | {r['label']} | {r['bytes'] / 1e6:.1f} | {r['calls']} | {r['median_ms']:.3f} | "
                 f"{r['p95_ms']:.3f} | {r['gb_s']:.2f} |")
    L += ["", "Size sweep (one copy):", "", "| bytes | median ms | GB/s |", "|---:|---:|---:|"]
    for r in sweep:
        L.append(f"| {r['bytes']:,} | {r['median_ms']:.4f} | {r['gb_s']:.2f} |")
    (out / "h2d.md").write_text("\n".join(L) + "\n")

    if (src / "overlap.json").exists():
        o = json.loads((src / "overlap.json").read_text())
        L = [f"### RF1-MA decode step with and without a concurrent H2D copy ({dev})", "",
             "One CUDA graph = RF1-MA on every (layer, q|k|v / o / gate|up / down) of the model (2 launches each), "
             "ranks 16/32/64/128 uniform over the step's tokens, every adapter resident at r 128 in the "
             "whole-model pool. 'with copy': a copy stream streams whole-model pages for the whole timed window.", "",
             "| model | tokens | adapters | step alone ms (p95) | step with copy ms (p95) | slowdown | copy GB/s alone | "
             "copy GB/s during steps | copy covered window |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for r in o:
            L.append(f"| {r['model']} | {r['tokens']} | {r['adapters']} | {r['step_alone_median_ms']:.3f} "
                     f"({r['step_alone_p95_ms']:.3f}) | {r['step_with_copy_median_ms']:.3f} ({r['step_with_copy_p95_ms']:.3f}) "
                     f"| x{r['slowdown']:.3f} | {r['copy_gb_s_alone']:.1f} | {r['copy_gb_s_with_steps']:.1f} | "
                     f"{r['copy_covered_window']} |")
        (out / "overlap.md").write_text("\n".join(L) + "\n")

    if (src / "ready.json").exists():
        rd = json.loads((src / "ready.json").read_text())
        L = [f"### Time until a rank-r request on a cold adapter can run ({dev}, r_max 256)", "",
             "ensure() call to copy_done (host wall clock; device copy time in brackets). prefix = ceil(r/16) "
             "whole-model pages; whole = all 16 pages (what a whole-adapter policy loads).", "",
             "| model | rank | prefix pages | prefix ms | whole ms | speed-up |", "|---|---:|---:|---:|---:|---:|"]
        for m in ("0.5B", "7B"):
            for rank in (16, 32, 64, 128, 256):
                pr = [r for r in rd if r.get("model") == m and r.get("rank") == rank and r.get("mode") == "prefix"]
                wh = [r for r in rd if r.get("model") == m and r.get("rank") == rank and r.get("mode") == "whole"]
                if pr and wh:
                    pr, wh = pr[0], wh[0]
                    L.append(f"| {m} | {rank} | {pr['pages']} | {pr['ready_wall_median_ms']:.2f} "
                             f"({pr['ready_dev_median_ms']:.2f}) | {wh['ready_wall_median_ms']:.2f} "
                             f"({wh['ready_dev_median_ms']:.2f}) | x{wh['ready_wall_median_ms'] / pr['ready_wall_median_ms']:.1f} |")
        fast = [r for r in rd if r.get("mode") == "resident_fast_path"]
        L += [""] + [f"Host cost of ensure() when all pages are resident ({r['model']}): {r['ensure_host_us']:.1f} us"
                     for r in fast]
        (out / "ready.md").write_text("\n".join(L) + "\n")
    print(f"--bw {fit['bw_gb_s']:.2f} --call-us {call_us:.2f}")


if __name__ == "__main__":
    main()
