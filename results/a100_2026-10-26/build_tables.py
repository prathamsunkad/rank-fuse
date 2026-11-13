"""Label every A100 (26 Oct 2026) sweep row as decode or prefill, one row per workload.

    python3 results/a100_2026-10-26/build_tables.py
Reads source/{rf_vs_b1_r64,rf_vs_b1_r128,h200_replica}-*A100*.csv (copied from results/raw/); writes all_rows.csv and
ALL_ROWS.md next to this script. Times are median us per call (graph timing, except b2_grouped:
eager, and the *_graphed baselines: their own graph, timed eagerly). Ratios > 1 = first one faster.
"""
import csv
import glob
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "source"  # copies of the results/raw CSVs these tables were built from
SWEEPS = ["rf_vs_b1_r64", "rf_vs_b1_r128", "h200_replica"]
IMPLS = [("b1_padded", "B1"), ("b1_padded_graphed", "B1g"), ("b2_grouped", "B2 (eager)"), ("b2_grouped_graphed", "B2g"),
         ("rf1_single", "RF1 1-kernel"), ("rf1_fused", "RF1"), ("rf2_fused", "RF2 1-kernel"), ("rf2_splitk", "RF2")]
KEY = ["num_requests", "tokens_per_request", "r_max", "tiers", "dist", "seed"]

rows = []
for sweep in SWEEPS:
    (path,) = glob.glob(str(RAW / f"{sweep}-*A100*.csv"))
    by = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        by[tuple(r[k] for k in KEY)][r["impl"]] = r
    for key, impls in by.items():
        any_r = next(iter(impls.values()))
        n, tpr = int(any_r["num_requests"]), int(any_r["tokens_per_request"])
        t = {label: float(impls[name]["median_us"]) if name in impls else None for name, label in IMPLS}
        ratio = lambda a, b: f"{t[b] / t[a]:.2f}" if t[a] and t[b] else ""
        rows.append({
            "sweep": sweep, "gpu": any_r["gpu"], "phase": "decode" if tpr == 1 else "prefill",
            "requests": n, "tokens/request": tpr, "tokens": n * tpr, "r_max": int(any_r["r_max"]),
            "tiers": any_r["tiers"], "mix": any_r["dist"], "seed": int(any_r["seed"]),
            "max rank": int(any_r["batch_max_rank"]), "mean rank": round(float(any_r["mean_rank"]), 1),
            **{f"{label} us": (f"{v:.1f}" if v is not None else "") for label, v in t.items()},
            "RF1 vs B1": ratio("RF1", "B1"), "RF2 vs RF1": ratio("RF2", "RF1"), "RF2 vs B1": ratio("RF2", "B1"),
            "B2g vs B1": ratio("B2g", "B1"),
        })

MIX_ORDER = ["all_min", "skew_low", "uniform", "skew_high", "all_max"]
rows.sort(key=lambda r: (SWEEPS.index(r["sweep"]), r["phase"] != "decode", r["r_max"], r["tokens"],
                         r["requests"], MIX_ORDER.index(r["mix"]), r["seed"]))
cols = list(rows[0])
with open(HERE / "all_rows.csv", "w", newline="") as f:
    w = csv.DictWriter(f, cols)
    w.writeheader()
    w.writerows(rows)

# Markdown: one section per sweep and phase; drop columns that are empty or constant in a section.
show = ["requests", "tokens/request", "tokens", "r_max", "mix", "seed", "max rank", "mean rank", "B1 us", "B2g us",
        "RF1 1-kernel us", "RF1 us", "RF2 1-kernel us", "RF2 us", "RF1 vs B1", "RF2 vs RF1", "RF2 vs B1", "B2g vs B1"]
out = ["# A100 sweep results, 26 Oct 2026, every row labelled\n",
       "Built by `build_tables.py` from `source/` (copies of the raw sweep CSVs). Median us per call, CUDA-graph timing (GPU time).",
       "B2g = b2_grouped_graphed (replays its own graph, timed eagerly); plain b2_grouped (eager, ~200-770 us,",
       "mostly Python overhead) and b1_padded_graphed are in all_rows.csv only. Ratios > 1: the first is faster",
       "(RF2 vs RF1 = RF1 time / RF2 time). decode = 1 token per request; prefill = more than 1.",
       "Ranks are sampled, so max/mean rank vary by seed and request count.\n"]
for sweep in SWEEPS:
    for phase in ["decode", "prefill"]:
        sec = [r for r in rows if r["sweep"] == sweep and r["phase"] == phase]
        if not sec:
            continue
        c = [k for k in show if any(r[k] != "" for r in sec) and not (k in ("r_max", "tokens/request", "seed") and len({r[k] for r in sec}) == 1)]
        out.append(f"## {sweep}: {phase} ({sec[0]['gpu']}, tiers {', '.join(sorted({r['tiers'] for r in sec}))})\n")
        out.append("| " + " | ".join(c) + " |")
        out.append("|" + "---|" * len(c))
        out += ["| " + " | ".join(str(r[k]) for k in c) + " |" for r in sec]
        out.append("")
(HERE / "ALL_ROWS.md").write_text("\n".join(out))
print(len(rows), "rows ->", HERE / "all_rows.csv", "and ALL_ROWS.md")
