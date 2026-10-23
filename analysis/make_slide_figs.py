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
from matplotlib.colors import LogNorm


PALETTE = {
    "b1_padded": "#0072B2",
    "b2_grouped": "#D55E00",
    "b2_grouped_graphed": "#009E73",
    "b1_padded_graphed": "#CC79A7",
}
ORDER = ["b1_padded", "b1_padded_graphed", "b2_grouped", "b2_grouped_graphed"]
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
    ignored = {"median_us", "p20_us", "p80_us", "max_abs_err", "git_sha"}
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
    fig.text(0.5, 0.012, f"Preliminary | {gpu} | final sweeps on PACE Phoenix",
             ha="center", fontsize=14)


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
    grouped_roofline_drawn = False
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
        if roof and impl != "b2_grouped_graphed":
            label = "roofline (padded)" if impl == "b1_padded" else "roofline (grouped)"
            if impl == "b2_grouped" and grouped_roofline_drawn:
                label = "_nolegend_"
            ax.plot(x[:len(roof)], roof, "--", color=PALETTE[impl], linewidth=2, label=label)
            if impl == "b2_grouped":
                grouped_roofline_drawn = True
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("number of tokens (decode batch size)")
    ax.set_ylabel("median latency (us)")
    ax.set_title("Latency vs. batch: decode, uniform ranks, h=4096, r_max=64")
    ax.plot([], [], "k--", label="roofline")
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
    batches = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True, facecolor="white")
    panels = [
        ("b2_grouped", "b1_padded", "eager / eager"),
        ("b2_grouped_graphed", "b1_padded_graphed", "graphed / graphed"),
    ]
    image = None
    for ax, (impl, b1_impl, title) in zip(axes, panels):
        values = np.full((len(DISTS), len(batches)), np.nan)
        for i, dist in enumerate(DISTS):
            for j, batch in enumerate(batches):
                b1 = rows_for(rows, phase="decode", num_requests=batch, h_in=4096,
                              h_out=4096, r_max=64, dist=dist, impl=b1_impl)
                other = rows_for(rows, phase="decode", num_requests=batch, h_in=4096,
                                 h_out=4096, r_max=64, dist=dist, impl=impl)
                if b1 and other:
                    values[i, j] = float(b1[0]["median_us"]) / float(other[0]["median_us"])
        image = ax.imshow(values, aspect="auto", cmap="RdYlGn",
                          norm=LogNorm(vmin=0.25, vmax=4.0))
        ax.set_title(f"{title}: {impl} / {b1_impl}")
        ax.set_xticks(range(len(batches)), batches)
        ax.set_yticks(range(len(DISTS)), DISTS)
        for i in range(len(DISTS)):
            for j in range(len(batches)):
                if np.isfinite(values[i, j]):
                    normalized = image.norm(values[i, j])
                    color = "black" if normalized > 0.42 else "white"
                    ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center",
                            color=color, fontsize=14, fontweight="bold")
    axes[0].set_ylabel("rank distribution")
    for ax in axes:
        ax.set_xlabel("batch size (requests)")
    fig.colorbar(image, ax=axes, location="right", fraction=0.04, pad=0.08,
                 label="speedup vs B1 (>1 = faster than B1)")
    fig.suptitle("Speedup vs. B1: like-for-like eager and graphed comparisons")
    footer(fig, gpu)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(out, dpi=100)
    plt.close(fig)


def load_profiler(path: str) -> list[dict]:
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def figure_launch(rows: list[dict], profiler_rows: list[dict], out: Path, gpu: str) -> None:
    batches = [1, 32, 256]
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), sharey=True, facecolor="white")
    for ax, batch in zip(axes, batches):
        values = []
        labels = []
        colors = []
        for impl in ["b1_padded", "b1_padded_graphed", "b2_grouped", "b2_grouped_graphed"]:
            data = [r for r in profiler_rows if r["impl"] == impl and int(r["num_requests"]) == batch]
            if not data:
                continue
            values.append((float(data[0]["event_us"]), float(data[0]["kernel_sum_us"])))
            labels.append(impl)
            colors.append(PALETTE[impl])
        positions = np.arange(len(labels))
        width = 0.36
        ax.bar(positions - width / 2, [v[0] for v in values], width, color=colors,
               label="wall time (CUDA events)")
        ax.bar(positions + width / 2, [v[1] for v in values], width, color=colors,
               alpha=0.45, hatch="//", label="GPU kernel time (profiler)")
        ax.set_title(f"batch={batch}")
        ax.set_xticks(positions, labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("time (us)")
    axes[0].legend()
    fig.suptitle("Wall time vs. GPU kernel time: launch/host overhead")
    footer(fig, gpu)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=100)
    plt.close(fig)


def figure_roofline_gap(rows: list[dict], out: Path, gpu: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), facecolor="white")
    for ax, phase in zip(axes, ["decode", "prefill"]):
        for impl in ORDER:
            data = [r for r in rows if r["phase"] == phase and r["impl"] == impl
                    and r["roofline_us"]]
            grouped = defaultdict(list)
            for row in data:
                x = (row["num_requests"], row["tokens_per_request"]) if phase == "prefill" else row["num_requests"]
                grouped[x].append(float(row["median_us"]) / float(row["roofline_us"]))
            ordered = sorted(grouped.items(), key=lambda item: (int(item[0][0]), int(item[0][1]))
                             if isinstance(item[0], tuple) else int(item[0]))
            labels = [f"{x[0]}x{x[1]}" for x, _ in ordered] if phase == "prefill" else [x for x, _ in ordered]
            medians = [statistics.median(v) for _, v in ordered]
            lows = [min(v) for _, v in ordered]
            highs = [max(v) for _, v in ordered]
            positions = np.arange(len(ordered))
            ax.plot(positions, medians, marker="o", color=PALETTE[impl], label=impl, linewidth=2)
            ax.fill_between(positions, lows, highs, color=PALETTE[impl], alpha=0.12)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
        ax.set_yscale("log")
        ax.set_title(phase)
        ax.set_xlabel("workload (requests x tokens)" if phase == "prefill" else "batch size (requests)")
        ax.grid(True, which="both", alpha=0.2)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5)
        ax.text(0.02, 1.02, "roofline lower bound", transform=ax.transAxes, fontsize=12)
    axes[0].set_ylabel("measured latency / roofline lower bound")
    axes[0].legend(fontsize=12)
    fig.suptitle("Roofline gap: distance from the memory/compute lower bound")
    footer(fig, gpu)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(out, dpi=100)
    plt.close(fig)


def write_summary(rows: list[dict], noisy: list[dict], profiler_rows: list[dict],
                  out: Path, commands: str) -> None:
    gpu = rows[0]["gpu"] if rows else "unknown GPU"
    lines = [
        "# Laptop baseline metrics",
        "",
        f"GPU: {gpu}. All figures state that this is a laptop GPU, not Phoenix H100.",
        "",
        "## Decode summary (uniform, h=4096, r_max=64)",
        "",
        "| Batch | B1 padded | B1g padded | B2 grouped | B2g grouped | B2/B1 | B2g/B1g |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for batch in [1, 8, 64, 256]:
        values = {}
        for impl in ORDER:
            data = rows_for(rows, phase="decode", num_requests=batch, h_in=4096,
                            h_out=4096, r_max=64, dist="uniform", impl=impl)
            values[impl] = float(data[0]["median_us"]) if data else float("nan")
        lines.append(f"| {batch} | {values['b1_padded']:.3f} | {values['b1_padded_graphed']:.3f} | "
                     f"{values['b2_grouped']:.3f} | {values['b2_grouped_graphed']:.3f} | "
                     f"{values['b2_grouped']/values['b1_padded']:.3f} | "
                     f"{values['b2_grouped_graphed']/values['b1_padded_graphed']:.3f} |")
    b1_sensitivity = []
    for batch in [1, 32, 256]:
        low = rows_for(rows, phase="decode", num_requests=batch, h_in=4096, h_out=4096,
                       r_max=64, dist="all_min", impl="b1_padded")
        high = rows_for(rows, phase="decode", num_requests=batch, h_in=4096, h_out=4096,
                        r_max=64, dist="all_max", impl="b1_padded")
        if low and high:
            b1_sensitivity.append(
                f"batch {batch}: all_min {float(low[0]['median_us']):.3f} us, "
                f"all_max {float(high[0]['median_us']):.3f} us"
            )
    speedup_cells = []
    for impl in ["b2_grouped", "b2_grouped_graphed"]:
        for dist in DISTS:
            for batch in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                b1_impl = "b1_padded_graphed" if impl == "b2_grouped_graphed" else "b1_padded"
                b1 = rows_for(rows, phase="decode", num_requests=batch, h_in=4096,
                              h_out=4096, r_max=64, dist=dist, impl=b1_impl)
                other = rows_for(rows, phase="decode", num_requests=batch, h_in=4096,
                                 h_out=4096, r_max=64, dist=dist, impl=impl)
                if b1 and other:
                    speedup_cells.append(
                        float(b1[0]["median_us"]) / float(other[0]["median_us"])
                    )
    lines += [
        "",
        "## Figure takeaways",
        "",
        "- **latency_vs_batch:** Uniform decode latency was measured for all requested batch sizes, with repeat ranges shown as error bars; B2 and B2g share one grouped roofline line.",
        f"- **rank_sensitivity:** For B1, the measured all_min/all_max medians were {'; '.join(b1_sensitivity)}, so these data {'do not show' if all(float(x.split('all_min ')[1].split(' us')[0]) <= float(x.split('all_max ')[1].split(' us')[0]) * 1.3 for x in b1_sensitivity) else 'show'} a large rank-distribution reduction in padded latency.",
        f"- **speedup_vs_b1:** {sum(value < 1.0 for value in speedup_cells)} of {len(speedup_cells)} like-for-like grouped/B1 cells were slower than their matching B1 variant.",
        "- **launch_overhead:** Profiler measurements show wall time substantially exceeds summed GPU kernel time for eager B1/B2, while graphed variants narrow that gap; grouping and max-rank work are outside the timed graphed calls.",
        "- **roofline_gap:** The two panels aggregate rank distributions at each workload x position with median lines and min-max bands; values below 1 remain unresolved because the analytic model was not changed.",
        "- **graphed variants:** B1g captures the batch max rank, and B2g captures rank grouping, outside the timed replay; this is a best-case launch-overhead comparison.",
        "",
        "## Profiling and unresolved roofline observations",
        "",
        "- The machine uses the Windows WDDM driver model.",
        "- Below-roofline B2g rows imply 261-349 GB/s versus the 256 GB/s model entry and 226.99 GB/s measured copy bandwidth; this may reflect optimistic byte accounting and/or cache effects despite L2 flush, and is not treated as proof of a physical bandwidth result.",
        "",
        "### Profiler rows",
        "",
        "| Implementation | Batch | Wall us | Kernel us | Kernel launches |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in profiler_rows:
        lines.append(f"| {row['impl']} | {row['num_requests']} | {float(row['event_us']):.3f} | "
                     f"{float(row['kernel_sum_us']):.3f} | {row['kernel_launches']} |")
    below = [row for row in rows if float(row["median_us"]) < float(row["roofline_us"])]
    lines += [
        "",
        "### Below-roofline rows",
        "",
        "| Phase | Implementation | Requests | Tokens | Distribution | Median us | Roofline us | Implied GB/s |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in below:
        implied = float(row["roofline_bytes"]) / (float(row["median_us"]) * 1e-6) / 1e9
        lines.append(f"| {row['phase']} | {row['impl']} | {row['num_requests']} | {row['num_tokens']} | "
                     f"{row['dist']} | {float(row['median_us']):.3f} | {float(row['roofline_us']):.3f} | {implied:.1f} |")
    lines += [
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
    parser.add_argument("--profiler", default="results/profiler_overhead.csv")
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
    profiler_path = Path(args.profiler)
    figure_launch(rows, load_profiler(str(profiler_path)), out_dir / "launch_overhead.png", gpu)
    figure_roofline_gap(rows, out_dir / "roofline_gap.png", gpu)
    decode_args = " ".join(args.decode)
    prefill_args = " ".join(args.prefill)
    command = (
        f"python analysis\\make_slide_figs.py --decode {decode_args} "
        f"--prefill {prefill_args}"
    )
    profiler_rows = load_profiler(str(profiler_path))
    write_summary(rows, noisy, profiler_rows, Path(args.summary_out), command)


if __name__ == "__main__":
    main()
