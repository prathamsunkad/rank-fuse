"""Latency vs number of tokens, one panel per (h, r_max, dist), measured lines + roofline dashes.

    python analysis/plot.py results/raw/decode-*.csv --out results/decode.png
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="+")
    p.add_argument("--out", default="results/plot.png")
    a = p.parse_args()

    rows = load(a.csv)
    panels = defaultdict(lambda: defaultdict(list))
    for r in rows:
        panel = (r["h_in"], r["h_out"], r["r_max"], r["dist"])
        panels[panel][r["impl"]].append(r)

    keys = sorted(panels)
    cols = min(4, len(keys))
    nrows = -(-len(keys) // cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(4.2 * cols, 3.4 * nrows), squeeze=False)
    for ax, key in zip(axes.flat, keys):
        for impl, rs in sorted(panels[key].items()):
            rs = sorted(rs, key=lambda r: int(r["num_tokens"]))
            xs = [int(r["num_tokens"]) for r in rs]
            line = ax.plot(xs, [float(r["median_us"]) for r in rs], marker="o", ms=3, label=impl)[0]
            if rs[0]["roofline_us"]:
                ax.plot(xs, [float(r["roofline_us"]) for r in rs], ls="--", lw=1, color=line.get_color())
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"h={key[0]}x{key[1]} r_max={key[2]} {key[3]}", fontsize=9)
        ax.set_xlabel("tokens")
        ax.set_ylabel("latency (us)")
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("median latency (solid) vs roofline lower bound (dashed)")
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
