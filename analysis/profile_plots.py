"""Regenerate profiling tables and figures from measured CSV files.

No values are synthesized: missing inputs produce a warning and the corresponding
figure is omitted.  Usage:

    python analysis/profile_plots.py results/h200_ice_YYYY-MM-DD
"""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SUBTITLE = "NVIDIA H200 (PACE ICE)"
COLORS = {
    "sort": "#9467bd",
    "shrink": "#1f77b4",
    "expand": "#ff7f0e",
    "index setup": "#2ca02c",
    "gathered x": "#d62728",
    "bucket branch": "#8c564b",
    "scattered stores": "#e377c2",
    "RF1": "#17becf",
    "RF2": "#bcbd22",
}


def read_csvs(root: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for path in sorted(root.rglob("*.csv")):
        try:
            frame = pd.read_csv(path)
        except (OSError, ValueError) as exc:
            warnings.warn(f"Skipping {path}: {exc}")
            continue
        if not frame.empty:
            frames[path.stem] = frame
    return frames


def combined(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames.values(), ignore_index=True) if frames else pd.DataFrame()


def write_table(frame: pd.DataFrame, tables: Path, name: str) -> None:
    if frame.empty:
        return
    frame.to_csv(tables / f"{name}.csv", index=False)
    (tables / f"{name}.md").write_text(frame.to_markdown(index=False) + "\n")


def save(fig: plt.Figure, figures: Path, name: str, data: pd.DataFrame) -> None:
    fig.tight_layout()
    fig.savefig(figures / f"{name}.png", dpi=200)
    data.to_csv(figures / f"{name}.csv", index=False)
    plt.close(fig)


def finish_axis(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(f"{title}\n{SUBTITLE}", fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


def latency_breakdown(df: pd.DataFrame, tables: Path, figures: Path) -> None:
    required = {"impl", "median_us"}
    if not required <= set(df):
        return
    keys = [c for c in ("tokens", "num_tokens", "r_max", "dist", "shape") if c in df]
    pivot = df.pivot_table(index=keys, columns="impl", values="median_us", aggfunc="median").reset_index()
    write_table(pivot, tables, "latency_breakdown")
    components = {"sort": ["sort_us", "sort_median_us"], "shrink": ["shrink_us"], "expand": ["expand_us"]}
    available = {k: next((c for c in v if c in df), None) for k, v in components.items()}
    if not any(available.values()):
        return
    plot = df.copy()
    for component, col in available.items():
        if col:
            plot[component] = plot[col]
    if not {"rf1_us", "rf2_us"} <= set(plot):
        return
    data = plot[["rf1_us", "rf2_us"] + [x for x in available if available[x]]].copy()
    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(len(data))
    bottom = [0.0] * len(data)
    for comp in ("sort", "shrink", "expand"):
        if comp in data:
            ax.bar(x, data[comp], bottom=bottom, color=COLORS[comp], label=comp)
            bottom = [a + b for a, b in zip(bottom, data[comp])]
    finish_axis(ax, "RF1/RF2 component latency", "Workload index", "Latency (us)")
    ax.legend()
    save(fig, figures, "01_latency_breakdown", data)


def ratios(df: pd.DataFrame, tables: Path, figures: Path) -> None:
    if not {"tokens", "rf1_us", "rf2_us"} <= set(df):
        return
    data = df.copy()
    data["rf2_rf1_ratio"] = data["rf2_us"] / data["rf1_us"]
    write_table(data, tables, "latency_ratios")
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, group in data.groupby([c for c in ("r_max", "phase", "sort") if c in data] or ["r_max"]):
        group = group.sort_values("tokens")
        ax.plot(group["tokens"], group["rf2_rf1_ratio"], marker="o", label=str(label))
    ax.axhline(1, color="black", linewidth=1)
    finish_axis(ax, "RF2/RF1 latency ratio", "Tokens", "RF2 / RF1")
    ax.legend()
    save(fig, figures, "02_rf2_rf1_ratio", data)


def generic_metrics(df: pd.DataFrame, tables: Path, figures: Path) -> None:
    if df.empty:
        return
    write_table(df, tables, "all_measurements")
    names = [c for c in ("dram_pct", "sm_pct", "tensor_pct") if c in df]
    if {"impl", "kernel"} <= set(df) and names:
        grouped = df.groupby(["impl", "kernel"], as_index=False)[names].median()
        write_table(grouped, tables, "ncu_metrics")
        fig, ax = plt.subplots(figsize=(11, 5))
        grouped.set_index(["impl", "kernel"])[names].plot.bar(ax=ax)
        finish_axis(ax, "Per-kernel utilization", "Implementation / kernel", "Percent of peak")
        save(fig, figures, "05_kernel_utilization", grouped)


def component_figures(frames: dict[str, pd.DataFrame], tables: Path, figures: Path) -> None:
    comp = next((f for f in frames.values() if {"component", "median_us", "kind"} <= set(f)), pd.DataFrame())
    if not comp.empty:
        write_table(comp, tables, "component_timings")
        measured = comp[comp["kind"].eq("component")].copy()
        if not measured.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            measured.plot.bar(x="component", y="median_us", ax=ax, color="#1f77b4", legend=False)
            finish_axis(ax, "Measured component latency", "Component", "Median latency (us)")
            save(fig, figures, "03_sort_and_kernel_time", measured)
        ab = comp[comp["kind"].eq("ablation")].copy()
        if not ab.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            ab.plot.bar(x="component", y="median_us", ax=ax, color="#d62728", legend=False)
            finish_axis(ax, "Measured step-level ablations", "Ablation", "Median latency (us)")
            save(fig, figures, "04_step_ablation", ab)

    ncu = next((f for f in frames.values() if "gpu__time_duration.sum" in f), pd.DataFrame())
    if not ncu.empty:
        write_table(ncu, tables, "ncu_raw_metrics")
        aliases = {
            "dram__bytes_read.sum": "DRAM read bytes",
            "dram__bytes_write.sum": "DRAM write bytes",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": "SM peak %",
            "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active": "Tensor pipe %",
            "launch__occupancy_limit_registers": "Occupancy register limit",
        }
        cols = [c for c in aliases if c in ncu]
        if cols:
            plot = ncu[cols].rename(columns=aliases)
            fig, ax = plt.subplots(figsize=(12, 5))
            plot.plot.bar(ax=ax)
            finish_axis(ax, "Measured Nsight Compute kernel metrics", "Kernel report row", "Measured value")
            save(fig, figures, "06_occupancy_and_stalls", plot)
            if {"DRAM read bytes", "DRAM write bytes"} <= set(plot):
                fig, ax = plt.subplots(figsize=(10, 5))
                plot[["DRAM read bytes", "DRAM write bytes"]].plot.bar(ax=ax)
                finish_axis(ax, "Measured DRAM traffic", "Kernel report row", "Bytes")
                save(fig, figures, "07_measured_vs_roofline_bytes", plot[["DRAM read bytes", "DRAM write bytes"]])

    if not comp.empty and "median_us" in comp:
        before_after = comp.pivot_table(index="component", columns="impl", values="median_us", aggfunc="median").reset_index()
        if len(before_after.columns) > 1:
            write_table(before_after, tables, "optimization_before_after")
            fig, ax = plt.subplots(figsize=(10, 5))
            before_after.set_index("component").plot.bar(ax=ax)
            finish_axis(ax, "Measured before/after optimization candidates", "Component", "Median latency (us)")
            save(fig, figures, "08_optimization_before_after", before_after)


def make_index(figures: Path) -> None:
    rows = ["# Profiling figures", ""]
    takeaways = {
        "01_latency_breakdown": "Measured sort, shrink, and expand contributions by workload.",
        "02_rf2_rf1_ratio": "Measured RF2/RF1 ratio, including the y=1 parity reference.",
        "05_kernel_utilization": "Measured Nsight Compute memory and SM utilization.",
    }
    for path in sorted(figures.glob("*.png")):
        stem = path.stem
        rows.append(f"- [{path.name}]({path.name}): {takeaways.get(stem, 'Measured profiling result; see the companion CSV.')}")
    (figures / "INDEX.md").write_text("\n".join(rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    tables, figures = root / "tables", root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    frames = read_csvs(root)
    df = combined(frames)
    latency_breakdown(df, tables, figures)
    ratios(df, tables, figures)
    generic_metrics(df, tables, figures)
    component_figures(frames, tables, figures)
    make_index(figures)


if __name__ == "__main__":
    main()
