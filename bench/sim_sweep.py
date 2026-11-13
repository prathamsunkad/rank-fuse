"""Run simulator sweeps across the four residency policies and generate graphs.

Policies:
  1. stock vLLM per-(adapter,rank) slots (policy="vllm")
  2. one slot per adapter at r_max with whole-adapter LRU (policy="whole")
  3. whole-adapter at true rank LRU (policy="true_rank")
  4. prefix paging (policy="prefix")

Configurations:
  - Equal byte budgets: 0.5, 1.0, 2.0, 4.0, 8.0, 16.0 GB
  - Qwen2.5-7B sizes: page 81 MB, adapter 323 MB, r_max 64, page_rows 16
  - Adapters: {4, 16, 64, 256, 1024}
  - Popularity: uniform and Zipf (s=1.2)
  - Metrics: hit rate, bytes copied, evictions per s, cold-start fraction

Produces:
  - source/sim_results.json and .csv
  - tables/fig_sim_hit_rate.png (black-and-white graph)
  - tables/fig_sim_bytes_copied.png (black-and-white graph)
"""

from __future__ import annotations

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rankfuse.paging.sim import poisson_trace, simulate


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="results/h200_2026-11-12_multi_adapter")
    p.add_argument("--requests", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    src_dir = out_dir / "source"
    tables_dir = out_dir / "tables"
    src_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    PAGE_BYTES = 81 * 1024 * 1024
    ADAPTER_BYTES = 323 * 1024 * 1024
    R_MAX = 64
    PAGE_ROWS = 16

    budgets_gb = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    adapter_counts = [4, 16, 64, 256, 1024]
    policies = ["vllm", "whole", "true_rank", "prefix"]
    popularities = ["uniform", "zipf"]

    records = []
    print(f"Running simulation sweeps ({args.requests} requests, {len(budgets_gb)} budgets, {len(adapter_counts)} adapter counts)...", flush=True)

    for pop in popularities:
        for n_adapters in adapter_counts:
            trace = poisson_trace(
                args.requests,
                n_adapters,
                seed=args.seed,
                rate=10.0,
                popularity=pop,
                zipf_s=1.2,
                ranks=(1, 2, 4, 8, 16, 32, 64),
            )
            duration_s = trace[-1].arrival - trace[0].arrival if len(trace) > 1 else 1.0

            for b_gb in budgets_gb:
                b_bytes = int(b_gb * 1024**3)
                for pol in policies:
                    try:
                        res = simulate(
                            trace,
                            budget_bytes=b_bytes,
                            r_max=R_MAX,
                            page_rows=PAGE_ROWS,
                            page_bytes=PAGE_BYTES,
                            adapter_bytes=ADAPTER_BYTES,
                            policy=pol,
                        )
                        hit_rate = res.hit_rate
                        bytes_copied_gb = res.bytes_copied / (1024**3)
                        evictions_per_s = res.evictions / duration_s if duration_s > 0 else 0.0
                        cold_frac = res.cold_start_fraction
                    except MemoryError:
                        hit_rate = 0.0
                        bytes_copied_gb = 0.0
                        evictions_per_s = 0.0
                        cold_frac = 1.0

                    records.append({
                        "popularity": pop,
                        "adapters": n_adapters,
                        "budget_gb": b_gb,
                        "policy": pol,
                        "requests": args.requests,
                        "hit_rate": round(hit_rate, 4),
                        "bytes_copied_gb": round(bytes_copied_gb, 3),
                        "evictions_per_s": round(evictions_per_s, 2),
                        "cold_start_fraction": round(cold_frac, 4),
                    })

    # Save JSON and CSV
    json_path = src_dir / "sim_results.json"
    json_path.write_text(json.dumps(records, indent=2))
    print(f"Wrote {json_path}", flush=True)

    csv_path = src_dir / "sim_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {csv_path}", flush=True)

    # =================================================================
    # Black-and-White Graphs
    # =================================================================
    # Figure 1: Hit Rate vs Budget under Zipf (e.g. N=256 adapters)
    # Figure 2: Bytes Copied vs Budget under Zipf (e.g. N=256 adapters)

    styles = {
        "vllm": {"color": "black", "linestyle": ":", "marker": "s", "label": "stock vLLM per-(adapter,rank)"},
        "whole": {"color": "black", "linestyle": "--", "marker": "^", "label": "one slot / adapter at r_max (whole LRU)"},
        "true_rank": {"color": "black", "linestyle": "-.", "marker": "x", "label": "whole-adapter true-rank LRU"},
        "prefix": {"color": "black", "linestyle": "-", "marker": "o", "label": "prefix paging (RankFuse)"},
    }

    n_target = 256
    zipf_recs = [r for r in records if r["popularity"] == "zipf" and r["adapters"] == n_target]

    # Graph 1: Hit Rate
    plt.figure(figsize=(7, 4.5))
    for pol in policies:
        sub = [r for r in zipf_recs if r["policy"] == pol]
        sub.sort(key=lambda r: r["budget_gb"])
        x_vals = [r["budget_gb"] for r in sub]
        y_vals = [r["hit_rate"] * 100 for r in sub]
        st = styles[pol]
        plt.plot(x_vals, y_vals, color=st["color"], linestyle=st["linestyle"], marker=st["marker"], label=st["label"], linewidth=1.8, markersize=6)

    plt.xscale("log", base=2)
    plt.xlabel("GPU Memory Budget (GB)", fontsize=11)
    plt.ylabel("Cache Hit Rate (%)", fontsize=11)
    plt.title(f"Adapter Residency Hit Rate vs Budget (N={n_target}, Zipf s=1.2)", fontsize=12)
    plt.xticks(budgets_gb, [str(b) for b in budgets_gb])
    plt.grid(True, linestyle="--", alpha=0.6, color="gray")
    plt.legend(frameon=True, facecolor="white", edgecolor="black", fontsize=9)
    plt.tight_layout()
    fig1_path = tables_dir / "fig_sim_hit_rate.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved {fig1_path}", flush=True)

    # Graph 2: Bytes Copied
    plt.figure(figsize=(7, 4.5))
    for pol in policies:
        sub = [r for r in zipf_recs if r["policy"] == pol]
        sub.sort(key=lambda r: r["budget_gb"])
        x_vals = [r["budget_gb"] for r in sub]
        y_vals = [r["bytes_copied_gb"] for r in sub]
        st = styles[pol]
        plt.plot(x_vals, y_vals, color=st["color"], linestyle=st["linestyle"], marker=st["marker"], label=st["label"], linewidth=1.8, markersize=6)

    plt.xscale("log", base=2)
    plt.xlabel("GPU Memory Budget (GB)", fontsize=11)
    plt.ylabel("H2D Data Transferred (GB)", fontsize=11)
    plt.title(f"H2D Memory Traffic vs Budget (N={n_target}, Zipf s=1.2)", fontsize=12)
    plt.xticks(budgets_gb, [str(b) for b in budgets_gb])
    plt.grid(True, linestyle="--", alpha=0.6, color="gray")
    plt.legend(frameon=True, facecolor="white", edgecolor="black", fontsize=9)
    plt.tight_layout()
    fig2_path = tables_dir / "fig_sim_bytes_copied.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved {fig2_path}", flush=True)


if __name__ == "__main__":
    main()
