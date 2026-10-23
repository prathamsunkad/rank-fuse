"""Merge repeated laptop sweeps and create presentation figures.

Example:
    python analysis/make_slide_figs.py \
      --decode results/raw/laptop_decode_1.csv results/raw/laptop_decode_2.csv results/raw/laptop_decode_3.csv \
      --prefill results/raw/laptop_prefill_1.csv results/raw/laptop_prefill_2.csv results/raw/laptop_prefill_3.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "b1_padded": "#0072B2",
    "b2_grouped": "#D55E00",
    "b2_grouped_graphed": "#009E73",
}
ORDER = ["b1_padded", "b2_grouped", "b2_grouped_graphed"]
DISTS = ["all_min", "skew_low", "uniform", "skew_high", "all_max"]


def load(paths: list[str], phase: str) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, newline="") as stream:
            for row in csv.DictReader(stream):
                row["phase"] = phase
                rows.append(row)
    return rows


def merge(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    ignored = {"median_us", "p20_us", "p80_us", "max_abs_err", "phase"}
    groups = defaultdict(list)
    for row in rows:
        key = tuple((k, row[k]) for k in row if k not in ignored)
        groups[key].append(row)
    merged = []
    noisy = []
    for key, samples in groups.items():
        medians = [float(row["median_us"]) for row in samples]
        row = dict(key)
        row["median_us"] = statistics.median(medians)
        row["min_us"] = min(medians)
        row["max_us"] = max(medians)
        row["noise_ratio"] = max(medians) / min(medians) if min(medians) else float("inf")
        row["max_abs_err"] = max(float(row["max_abs_err"]) for row in samples if row["max_abs_err"])
        merged.append(row)
        if row["noise_ratio"] > 1.3:
            noisy.append(row)
    return merged, noisy


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def footer(fig, gpu: str) -> None:
    fig.text(0.5, 0.012, f"{gpu} | laptop GPU, not Phoenix H100", ha="center", fontsize=14)


def rows_for(rows, **filters):
    return [
        row for row in rows
        if all(row.get(key) == str(value) for key, value in filters.items())
    ]


def error_values(row: dict) -> tuple[float, float]:
    value = float(row["median_us"])
    return value - float(row["min_us"]), float(row["max_us"]) - value


def figure_latency(rows: list[dict], out: Path, gpu: str) -> None:
    fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
    for impl in ORDER:
        data = sorted(
            rows_for(rows, phase="decode", h_in=4096, h_out=4096, r_max=64, dist="uniform", impl=impl),
            key=lambda row: int(row["num_tokens"]),
        )
        x = [int(row["num_tokens"]) for row in data]
        y = [float(row["median_us"]) for row in data]
        low = [error_values(row)[0] for row in data]
        high = [error_values(row)[1] for row in data]
        ax.errorbar(x, y, yerr=[low, high], color=PALETTE[impl], marker="o", label=impl,
                    linewidth=2, capsize=3)
        roof = [float(row["roofline_us"]) for row in data if row["roofline_us"]]
        if roof:
            ax.plot(x[:len(roof)], roof, "--", color=PALETTE[impl], linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("number of tokens (decode batch size)")
    ax.set_ylabel("median latency (us)")
    ax.set_title("Latency vs. batch: decode, uniform ranks, h=4096, r_max=64")
    ax.legend()
    ax.grid(True, which="both", alpha=0.2)
    footer(fig, gpu)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=100)
    plt.close(fig)


def figure_rank_sensitivity(rows: list[dict], out: Path, gpu: str) -> None:
    batches = [1, 32, 256]
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), sharey=True, facecolor="white")
    width = 0.24
    positions = np.arange(len(DISTS))
    for ax, batch in zip(axes, batches):
        for index, impl in enumerate(ORDER):
            values = []
            for dist in DISTS:
                data = rows_for(rows, phase="decode", num_requests=batch, h_in=4096,
                                h_out=4096, r_max=64, dist=dist, impl=impl)
                values.append(float(data[0]["median_us"]) if data else np.nan)
            ax.bar(positions + (index - 1) * width, values, width, label=impl,
                   color=PALETTE[impl])
        ax.set_title(f"batch={batch}")
        ax.set_xticks(positions, DISTS, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("median latency (us)")
    axes[0].legend(fontsize=12)
    fig.suptitle("Rank sensitivity: decode latency by rank distribution")
    footer(fig, gpu)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(out, dpi=100)
    plt.close(fig)


def figure_speedup(rows: list[dict], out: Path, gpu: str) -> None:
    batches = [1, 8, 16, 32, 64, 128, 256]
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True, facecolor="white")
    for ax, impl in zip(axes, ["b2_grouped", "b2_grouped_graphed"]):
        values = np.full((len(DISTS), len(batches)), np.nan)
        for i, dist in enumerate(DISTS):
            for j, batch in enumerate(batches):
                b1 = rows_for(rows, phase="decode", num_requests=batch, h_in=4096,
                              h_out=4096, r_max=64, dist=dist, impl="b1_padded")
                other = rows_for(rows, phase="decode", num_requests=batch, h_in=4096,
                                 h_out=4096, r_max=64, dist=dist, impl=impl)
                if b1 and other:
                    values[i, j] = float(b1[0]["median_us"]) / float(other[0]["median_us"])
        image = ax.imshow(values, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.5)
        ax.set_title(f"{impl} / b1_padded speedup")
        ax.set_xticks(range(len(batches)), batches)
        ax.set_yticks(range(len(DISTS)), DISTS)
        for i in range(len(DISTS)):
            for j in range(len(batches)):
                if np.isfinite(values[i, j]):
                    color = "black" if values[i, j] >= 1 else "#7f0000"
                    ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center",
                            color=color, fontsize=14, fontweight="bold" if values[i, j] < 1 else "normal")
    axes[0].set_ylabel("rank distribution")
    fig.colorbar(image, ax=axes, label="speedup (B1 / implementation)")
    fig.suptitle("Speedup vs. B1 (red text marks slower-than-B1 cells)")
    footer(fig, gpu)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(out, dpi=100)
    plt.close(fig)


def figure_launch(rows: list[dict], out: Path, gpu: str) -> None:
    fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
    for impl in ORDER:
        data = sorted(rows_for(rows, phase="decode", h_in=4096, h_out=4096,
                               r_max=64, dist="uniform", impl=impl),
                      key=lambda row: int(row["num_tokens"]))
        ax.plot([int(row["num_tokens"]) for row in data],
                [float(row["median_us"]) for row in data], marker="o",
                color=PALETTE[impl], label=impl, linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch size")
    ax.set_ylabel("median latency (us)")
    ax.set_title("Launch overhead: eager grouped, graphed grouped, and padded")
    ax.legend()
    ax.grid(True, which="both", alpha=0.2)
    footer(fig, gpu)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=100)
    plt.close(fig)


def figure_roofline_gap(rows: list[dict], out: Path, gpu: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), facecolor="white")
    for ax, phase in zip(axes, ["decode", "prefill"]):
        for impl in ORDER:
            data = sorted(rows_for(rows, phase=phase, impl=impl),
                          key=lambda row: int(row["num_tokens"]))
            data = [row for row in data if row["roofline_us"]]
            ax.plot([int(row["num_tokens"]) for row in data],
                    [float(row["median_us"]) / float(row["roofline_us"]) for row in data],
                    marker="o", color=PALETTE[impl], label=impl, linewidth=2)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(phase)
        ax.set_xlabel("number of tokens")
        ax.grid(True, which="both", alpha=0.2)
    axes[0].set_ylabel("measured latency / roofline lower bound")
    axes[0].legend(fontsize=12)
    fig.suptitle("Roofline gap: distance from the memory/compute lower bound")
    footer(fig, gpu)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(out, dpi=100)
    plt.close(fig)


def write_summary(rows: list[dict], noisy: list[dict], out: Path, commands: str) -> None:
    gpu = rows[0]["gpu"] if rows else "unknown GPU"
    lines = [
        "# Laptop baseline metrics",
        "",
        f"GPU: {gpu}. All figures state that this is a laptop GPU, not Phoenix H100.",
        "",
        "## Decode summary (uniform, h=4096, r_max=64)",
        "",
        "| Batch | B1 padded (us) | B2 grouped (us) | B2g graphed (us) | B2/B1 | B2g/B1 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for batch in [1, 8, 64, 256]:
        values = {}
        for impl in ORDER:
            data = rows_for(rows, phase="decode", num_requests=batch, h_in=4096,
                            h_out=4096, r_max=64, dist="uniform", impl=impl)
            values[impl] = float(data[0]["median_us"]) if data else float("nan")
        lines.append(f"| {batch} | {values['b1_padded']:.3f} | {values['b2_grouped']:.3f} | "
                     f"{values['b2_grouped_graphed']:.3f} | "
                     f"{values['b2_grouped']/values['b1_padded']:.3f} | "
                     f"{values['b2_grouped_graphed']/values['b1_padded']:.3f} |")
    lines += [
        "",
        "## Figure takeaways",
        "",
        "- **latency_vs_batch:** Reports measured latency and repeat ranges for the uniform decode workloads; the plotted relationship is limited to these observations.",
        "- **rank_sensitivity:** Shows how each implementation's measured latency changes across the five rank distributions at batches 1, 32, and 256.",
        "- **speedup_vs_b1:** Cells below 1.0 indicate that the corresponding grouped implementation was slower than B1 for that measured workload.",
        "- **launch_overhead:** Compares the measured eager, graphed, and padded latency curves; it does not isolate causality beyond that comparison.",
        "- **roofline_gap:** Shows measured latency divided by the analytic lower bound for the decode and prefill rows included in the CSVs.",
        "",
        "## Noisy points (max/min > 1.3)",
    ]
    if noisy:
        for row in noisy:
            lines.append(f"- {row['phase']} {row['impl']} requests={row['num_requests']} "
                         f"tokens={row['num_tokens']} dist={row['dist']}: {float(row['noise_ratio']):.2f}x")
    else:
        lines.append("- None.")
    lines += [
        "",
        "## Dropped or skipped workloads",
        "",
        "- None recorded by the figure-generation step.",
        "",
        "## Regeneration commands",
        "",
        "```powershell",
        "python scripts\\measure_gpu.py",
        "python bench\\run_sweep.py bench/configs/laptop_decode.yaml --out results/raw/laptop_decode_1.csv",
        "python bench\\run_sweep.py bench/configs/laptop_decode.yaml --out results/raw/laptop_decode_2.csv",
        "python bench\\run_sweep.py bench/configs/laptop_decode.yaml --out results/raw/laptop_decode_3.csv",
        "python bench\\run_sweep.py bench/configs/laptop_prefill.yaml --out results/raw/laptop_prefill_1.csv",
        "python bench\\run_sweep.py bench/configs/laptop_prefill.yaml --out results/raw/laptop_prefill_2.csv",
        "python bench\\run_sweep.py bench/configs/laptop_prefill.yaml --out results/raw/laptop_prefill_3.csv",
        commands,
        "```",
    ]
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode", nargs="+", required=True)
    parser.add_argument("--prefill", nargs="+", required=True)
    parser.add_argument("--out-dir", default="results/figures")
    parser.add_argument("--merged-out", default="results/laptop_merged.csv")
    parser.add_argument("--summary-out", default="results/summary.md")
    args = parser.parse_args()
    rows, noisy = merge(load(args.decode, "decode") + load(args.prefill, "prefill"))
    if not rows:
        raise RuntimeError("No CSV rows found")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_csv(Path(args.merged_out), rows)
    gpu = rows[0]["gpu"]
    figure_latency(rows, out_dir / "latency_vs_batch.png", gpu)
    figure_rank_sensitivity(rows, out_dir / "rank_sensitivity.png", gpu)
    figure_speedup(rows, out_dir / "speedup_vs_b1.png", gpu)
    figure_launch(rows, out_dir / "launch_overhead.png", gpu)
    figure_roofline_gap(rows, out_dir / "roofline_gap.png", gpu)
    write_summary(rows, noisy, Path(args.summary_out), "python analysis\\make_slide_figs.py --decode ... --prefill ...")


if __name__ == "__main__":
    main()
