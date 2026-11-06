"""RF2 results split by r_max, with random and sorted order kept apart for rf2_segment and rf2_index.

    python analysis/order_tables.py results/raw/rf2_next/rf2_ideal.csv results/raw/rf2_next/rf2_skew.csv \
        --out results/a100_2026-11-05/BY_RMAX.md

Times are the median over seeds, in us. RF1, B1 and the bucketed kernels don't depend on order, so
their columns are the median over both orders. rf1_fused's second pass (the noise check) is left out.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

MIX_ORDER = ["low_frac_25", "low_frac_50", "low_frac_75", "skew_low", "zipf_1", "zipf_2",
             "low_frac_90", "low_frac_99", "low_frac_100"]
MIX_NOTE = {
    "all_max": "all at r_max (nothing to skip)",
    "low_frac_25": "25% rank 1, rest r_max",
    "low_frac_50": "50% rank 1, rest r_max",
    "low_frac_75": "75% rank 1, rest r_max",
    "low_frac_90": "90% rank 1, rest r_max",
    "low_frac_99": "99% rank 1, rest r_max",
    "low_frac_100": "all rank 1 but one r_max",
    "skew_low": "sampled, 78% in bucket 16",
    "zipf_1": "sampled, 81% in bucket 16",
    "zipf_2": "sampled, 95% in bucket 16",
    "all_min": "all rank 1 (ideal bound)",
}
ORDER_FREE = ["rf1_fused", "b1_padded", "rf2_bucketed", "rf2_bucketed_tuned"]
ORDERED = [("rf2_segment", "sorted"), ("rf2_index", "random"), ("rf2_segment", "random"), ("rf2_index", "sorted")]
HEAD = {
    ("rf2_segment", "sorted"): "**segment, sorted**",
    ("rf2_index", "random"): "**index, random**",
    ("rf2_segment", "random"): "segment, random",
    ("rf2_index", "sorted"): "index, sorted",
}


def load(path):
    t = defaultdict(list)  # (r_max, dist, n, tpr, impl, order) -> [median_us]
    gpu = ""
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if int(r.get("impl_pass") or 0):
                continue
            gpu = r["gpu"]
            key = (int(r["r_max"]), r["dist"], int(r["num_requests"]), int(r["tokens_per_request"] or 1),
                   r["impl"], r.get("request_order") or "random")
            t[key].append(float(r["median_us"]))
    return t, gpu


def med(t, r_max, dist, n, tpr, impl, orders=("random", "sorted")):
    vals = [v for o in orders for v in t.get((r_max, dist, n, tpr, impl, o), [])]
    return statistics.median(vals) if vals else None


def fmt(v):
    return f"{v:.1f}" if v is not None else ""


def ratio(base, v):
    return f"{base / v:.2f}×" if base and v else ""


def shape_table(t, r_max, n, tpr, mixes, free):
    """One table for one batch shape: times in µs, RF2 cells also show speedup vs RF1 / vs B1."""
    lines = ["| mix | " + " | ".join(free) + " | " + " | ".join(HEAD[c] for c in ORDERED) + " | ideal |",
             "|" + "---|" * (2 + len(free) + len(ORDERED))]
    ideal = med(t, r_max, "all_min", n, tpr, "rf1_fused")
    for m in mixes:
        if med(t, r_max, m, n, tpr, "rf1_fused") is None:
            continue
        fv = [med(t, r_max, m, n, tpr, i) for i in free]
        ov = [med(t, r_max, m, n, tpr, i, (o,)) for i, o in ORDERED]
        rf1, b1 = fv[0], (fv[1] if len(fv) > 1 else None)
        cells = [f"{fmt(v)} ({ratio(rf1, v)} / {ratio(b1, v)})" if v else "" for v in ov]
        lines.append(f"| {m} | " + " | ".join(map(fmt, fv)) + " | " + " | ".join(cells) + f" | {fmt(ideal)} |")
    return lines


def section(t, r_max, title):
    shapes = sorted({(k[2], k[3]) for k in t if k[0] == r_max}, key=lambda s: (s[0] * s[1], s[1]))
    mixes = [m for m in MIX_ORDER if any(k[0] == r_max and k[1] == m for k in t)]
    if not mixes:
        return []
    have = {k[4] for k in t if k[0] == r_max}
    free = [i for i in ORDER_FREE if i in have]
    out = [f"### {title}", ""]
    for phase, pick in (("Prefill", lambda s: s[1] > 1), ("Decode", lambda s: s[1] == 1)):
        group = [s for s in shapes if pick(s)]
        if not group:
            continue
        out += [f"#### {phase}", ""]
        for tokens in sorted({n * tpr for n, tpr in group}):
            for n, tpr in [s for s in group if s[0] * s[1] == tokens]:
                desc = f"{n} requests × {tpr} tokens" if tpr > 1 else f"{n} requests × 1 token"
                out += [f"**{tokens} tokens** ({desc})", ""] + shape_table(t, r_max, n, tpr, mixes, free) + [""]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csvs", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    data = [(Path(c).stem, *load(c)) for c in a.csvs]
    gpu = data[0][2]
    out = [
        "# RF2 results by r_max, random and sorted order kept apart",
        "",
        f"{gpu}, add mode, CUDA-graph timing, 4096 × 4096 fp16, median over 3 seeds. Built with "
        "`analysis/order_tables.py` from `results/raw/rf2_next/`.",
        "",
        "## How to read this",
        "",
        "`request_order` is the order of the requests **in memory** (the tokens themselves), not an index list.",
        "",
        "* `rf2_segment` never gets an index list. It forms blocks from consecutive rows, so it only skips",
        "  well when the tokens are physically sorted. **segment, sorted** is the physically-sorted-tokens",
        "  assumption; segment, random is what happens if the scheduler does nothing.",
        "* `rf2_index` always builds a sorted index list (argsort) and reads through it, so its blocks are the",
        "  same in either order. **index, random** is the sorted-index assumption (tokens in batch order plus",
        "  a list); index, sorted vs segment, sorted is the cost of the index loads alone.",
        "* The bold columns are the two headline results, one per assumption. The index side also pays the",
        "  argsort once per step (about 690 µs on the host, not included here).",
        "* RF1 (`rf1_fused`), B1 (`b1_padded`) and the bucketed kernels don't depend on order; their",
        "  columns are the median over both orders.",
        "* **ideal** is RF1 on an all-rank-1 batch of the same size: the time if nothing were at r_max.",
        "  Single-rank batches (`all_min`, `all_max`) are not shown as rows: they have nothing to skip.",
        "* RF2 cells read `time (speedup vs RF1 / speedup vs B1)`; above 1× is faster.",
        "* Each sweep is split into prefill (requests of 128–512 tokens) then decode (1 token per",
        "  request), with one table per token count.",
        "",
        "Mixes:",
        "",
        "| Mix | Ranks |",
        "|---|---|",
    ] + [f"| {m} | {MIX_NOTE[m]} |" for m in MIX_ORDER] + [""]
    for r_max in sorted({k[0] for _, t, _ in data for k in t}):
        out += [f"## r_max {r_max}", ""]
        for stem, t, _ in data:
            label = {"rf2_ideal": "Two-rank mixes (rf2_ideal)", "rf2_skew": "Sampled skewed mixes (rf2_skew)"}.get(stem, stem)
            out += section(t, r_max, label)
    Path(a.out).write_text("\n".join(out))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
