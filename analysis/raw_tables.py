"""Raw timings as Markdown: one row per workload, one column per kernel, median us over seeds.

    python analysis/raw_tables.py results/raw/rf2_next/*.csv --out results/a100_2026-11-05/RAW.md

rf1_fused's second pass (the noise check) is left out. Each input CSV gets its own section.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

KEY = ["r_max", "dist", "num_requests", "tokens_per_request", "request_order"]


def table(path: str) -> list[str]:
    times = defaultdict(lambda: defaultdict(list))
    impls: list[str] = []
    gpu = mode = ""
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if int(r.get("impl_pass") or 0):
                continue
            gpu, mode = r["gpu"], r.get("mode", "")
            if r["impl"] not in impls:
                impls.append(r["impl"])
            key = (int(r["r_max"]), r["dist"], int(r["num_requests"]), int(r["tokens_per_request"] or 1),
                   r.get("request_order") or "random", int(r["num_tokens"]))
            times[key][r["impl"]].append(float(r["median_us"]))
    lines = [f"## {Path(path).stem}", "", f"{gpu}, {mode} mode, median us over seeds.", "",
             "| r_max | mix | batch | tokens | order | " + " | ".join(impls) + " |",
             "|" + "---|" * (5 + len(impls))]
    for key in sorted(times):
        r_max, dist, n, tpr, order, tokens = key
        cells = [f"{statistics.median(times[key][i]):.1f}" if times[key].get(i) else "" for i in impls]
        lines.append(f"| {r_max} | {dist} | {n}x{tpr} | {tokens} | {order} | " + " | ".join(cells) + " |")
    return lines + [""]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csvs", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = ["# Raw timings", ""]
    for path in a.csvs:
        out += table(path)
    Path(a.out).write_text("\n".join(out))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
