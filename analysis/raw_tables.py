"""Raw timings as Markdown: one row per workload, one column per kernel, median us over seeds.

    python analysis/raw_tables.py results/raw/rf2_next/*.csv --out results/a100_2026-11-05/RAW.md

rf1_fused's second pass (the noise check) is left out. Each input CSV gets its own section.
--csv also writes the same medians as one wide CSV (a `sweep` column, then one column per kernel) for
analysis by hand; --skip-single leaves out single-rank batches (all_max, all_min).
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

KEY = ["r_max", "dist", "num_requests", "tokens_per_request", "request_order"]


SINGLE = ("all_max", "all_min")


def medians(path: str, skip_single: bool = False):
    """(gpu, mode, impls in file order, {key: {impl: median us}})."""
    times = defaultdict(lambda: defaultdict(list))
    impls: list[str] = []
    gpu = mode = ""
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if int(r.get("impl_pass") or 0):
                continue
            if skip_single and r["dist"] in SINGLE:
                continue
            gpu, mode = r["gpu"], r.get("mode", "")
            if r["impl"] not in impls:
                impls.append(r["impl"])
            key = (int(r["r_max"]), r["dist"], int(r["num_requests"]), int(r["tokens_per_request"] or 1),
                   r.get("request_order") or "random", int(r["num_tokens"]))
            times[key][r["impl"]].append(float(r["median_us"]))
    return gpu, mode, impls, {k: {i: statistics.median(v) for i, v in d.items()} for k, d in times.items()}


def table(path: str, skip_single: bool = False) -> list[str]:
    gpu, mode, impls, times = medians(path, skip_single)
    lines = [f"## {Path(path).stem}", "", f"{gpu}, {mode} mode, median us over seeds.", "",
             "| r_max | mix | batch | tokens | order | " + " | ".join(impls) + " |",
             "|" + "---|" * (5 + len(impls))]
    for key in sorted(times):
        r_max, dist, n, tpr, order, tokens = key
        cells = [f"{times[key][i]:.1f}" if i in times[key] else "" for i in impls]
        lines.append(f"| {r_max} | {dist} | {n}x{tpr} | {tokens} | {order} | " + " | ".join(cells) + " |")
    return lines + [""]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csvs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--csv", help="also write every median to this wide CSV")
    p.add_argument("--skip-single", action="store_true", help="leave out all_max / all_min batches")
    a = p.parse_args()
    out = ["# Raw timings", ""]
    for path in a.csvs:
        out += table(path, a.skip_single)
    Path(a.out).write_text("\n".join(out))
    print(f"-> {a.out}")
    if a.csv:
        data = [(Path(path).stem, *medians(path, a.skip_single)) for path in a.csvs]
        impls = list(dict.fromkeys(i for d in data for i in d[3]))
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sweep", "gpu", *KEY[:4], "num_tokens", "request_order", *(f"{i}_us" for i in impls)])
            for stem, gpu, _, _, times in data:
                for key in sorted(times):
                    r_max, dist, n, tpr, order, tokens = key
                    w.writerow([stem, gpu, r_max, dist, n, tpr, tokens, order,
                                *(f"{times[key][i]:.2f}" if i in times[key] else "" for i in impls)])
        print(f"-> {a.csv}")


if __name__ == "__main__":
    main()
