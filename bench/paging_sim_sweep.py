"""Sweep the four residency policies of rankfuse/paging/policy_sim.py over real page sizes.

    python bench/paging_sim_sweep.py --out results/a100_2026-11-23_paging \
        --bw 24.0 --call-us 8.0 [--workers 8]

Grid: models Qwen2.5-0.5B / 7B x r_max 64 / 128 / 256 x budgets 0.5 .. 20 GB x 100 / 300 / 1000
adapters x Zipf alpha 0.8 / 1.0 / 1.2 x rank mixes (uniform, skew_low, skew_high, all_max over
tiers 1, 2, 4, .., r_max) x policies (vllm, slora, rf_whole, rf_prefix). One seeded trace per
(adapters, alpha, mix, r_max) of --requests Poisson arrivals at --rate req/s; the first --warmup
requests only warm the cache. Copy times use --bw (GB/s) and --call-us (per cudaMemcpyAsync),
fitted from the A100 h2d measurements (source/h2d.json).

Writes source/sim_results.{csv,json}, tables/*.md and figures/*.png under --out.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from rankfuse.paging.policy_sim import POLICIES, RANK_MIXES, CostModel, SizeModel, make_trace, simulate_policy

BUDGETS_GB = [0.5, 1, 2, 4, 8, 12, 16, 20]
ADAPTERS = [100, 300, 1000]
ALPHAS = [0.8, 1.0, 1.2]
MIXES = list(RANK_MIXES)
MODELS = ["0.5B", "7B"]
RMAX = [64, 128, 256]
LABEL = {"vllm": "stock vLLM (slot per adapter-tier, padded)", "slora": "S-LoRA-style (true rank per tier)",
         "rf_whole": "RankFuse whole adapter (r_max)", "rf_prefix": "RankFuse prefix paging"}


def _job(args):
    (n_ad, alpha, mix, r_max), cfg = args
    trace = make_trace(cfg["requests"], n_ad, r_max=r_max, zipf_alpha=alpha, rank_mix=mix, rate=cfg["rate"],
                       seed=cfg["seed"])
    cost = CostModel(cfg["bw"], cfg["call_us"])
    rows = []
    for model in MODELS:
        size = SizeModel.qwen25(model, r_max)
        for b in BUDGETS_GB:
            for pol in POLICIES:
                r = simulate_policy(trace, pol, int(b * 1e9), size, cost, cfg["warmup"])
                row = {"model": model, "r_max": r_max, "budget_gb": b, "adapters": n_ad, "zipf_alpha": alpha,
                       "rank_mix": mix, "rate": cfg["rate"], **r.row()}
                rows.append(row)
    return rows


def _fmt(v, kind):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if kind == "pct":
        return f"{100 * v:.1f}"
    if kind == "gb":
        return f"{v:.0f}" if v >= 100 else f"{v:.1f}"
    if kind == "ms":
        return f"{v:.1f}" if v < 1000 else f"{v / 1000:.1f}s"
    return f"{v:.0f}" if v >= 10 else f"{v:.1f}"


def _table(rows, title, metric, kind, fixed, out):
    lines = [f"### {title}", "", f"Fixed: {fixed}. Rows: budget (GB); columns: policy. — = infeasible "
             "(one request's item does not fit the budget).", ""]
    for model in MODELS:
        for r_max in RMAX:
            sub = [r for r in rows if r["model"] == model and r["r_max"] == r_max]
            lines += [f"**Qwen2.5-{model}, r_max {r_max}**", "", "| budget GB | " + " | ".join(POLICIES) + " |",
                      "|---:|" + "---:|" * len(POLICIES)]
            for b in BUDGETS_GB:
                cells = []
                for pol in POLICIES:
                    m = [r for r in sub if r["budget_gb"] == b and r["policy"] == pol]
                    cells.append(_fmt(m[0][metric] if m and m[0]["feasible"] else None, kind))
                lines.append(f"| {b} | " + " | ".join(cells) + " |")
            lines.append("")
    out.write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--requests", type=int, default=30000)
    p.add_argument("--warmup", type=int, default=10000)
    p.add_argument("--rate", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bw", type=float, required=True, help="pinned H2D GB/s (A100 fit)")
    p.add_argument("--call-us", type=float, required=True, help="fixed us per cudaMemcpyAsync (A100 fit)")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--tag", default="", help="suffix of the output files")
    args = p.parse_args()
    out = Path(args.out)
    for d in ("source", "tables", "figures"):
        (out / d).mkdir(parents=True, exist_ok=True)
    cfg = dict(requests=args.requests, warmup=args.warmup, rate=args.rate, seed=args.seed, bw=args.bw,
               call_us=args.call_us)
    grid = list(itertools.product(ADAPTERS, ALPHAS, MIXES, RMAX))
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, part in enumerate(ex.map(_job, [(g, cfg) for g in grid])):
            rows += part
            if i % 12 == 0:
                print(f"  {i + 1}/{len(grid)} traces", flush=True)
    tag = args.tag
    (out / "source" / f"sim_results{tag}.json").write_text(json.dumps({"config": cfg, "rows": rows}))
    with (out / "source" / f"sim_results{tag}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows", flush=True)

    # Headline tables: 1000 adapters, alpha 1.0, uniform tiers.
    head = [r for r in rows if r["adapters"] == 1000 and r["zipf_alpha"] == 1.0 and r["rank_mix"] == "uniform"]
    fixed = "1000 adapters, Zipf alpha 1.0, rank tiers uniform over 1, 2, 4, .., r_max"
    T = out / "tables"
    _table(head, "Hit rate (%)", "hit_rate", "pct", fixed, T / f"hit_rate{tag}.md")
    _table(head, "GB copied host->GPU (20k counted requests)", "gb_copied", "gb", fixed, T / f"gb_copied{tag}.md")
    _table(head, "Adapters with any resident data (mean)", "resident_adapters", "n", fixed, T / f"resident{tag}.md")
    _table(head, f"Mean load wait per request (ms), {args.rate:g} req/s, {args.bw:g} GB/s, {args.call_us:g} us/copy",
           "mean_wait_ms", "ms", fixed, T / f"wait{tag}.md")
    _summary(rows, T / f"summary{tag}.md")
    _figures(rows, out / "figures", tag)


def _summary(rows, path):
    """Ratios of rf_prefix to each baseline, over every feasible config of the grid."""
    import statistics as st

    key = lambda r: (r["model"], r["r_max"], r["budget_gb"], r["adapters"], r["zipf_alpha"], r["rank_mix"])  # noqa: E731
    by = {}
    for r in rows:
        by.setdefault(key(r), {})[r["policy"]] = r
    lines = ["### rf_prefix vs each baseline over the whole grid", "",
             "Per config (model, r_max, budget, adapters, alpha, mix) where both are feasible. "
             "Miss ratio = baseline misses / rf_prefix misses (>1: prefix misses less); "
             "bytes ratio = baseline bytes / rf_prefix bytes; wait ratio likewise (configs with prefix wait > 0).", "",
             "| baseline | configs | prefix hit rate >= baseline | median hit-rate gain (pts) | max gain (pts) | "
             "median miss ratio | median bytes ratio | median wait ratio | only prefix feasible |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for base in ("vllm", "slora", "rf_whole"):
        gains, mr, br, wr, n, ge, only = [], [], [], [], 0, 0, 0
        for d in by.values():
            p, b = d.get("rf_prefix"), d.get(base)
            if p and b and p["feasible"] and not b["feasible"]:
                only += 1
            if not (p and b and p["feasible"] and b["feasible"]):
                continue
            n += 1
            g = p["hit_rate"] - b["hit_rate"]
            ge += g >= -1e-12
            gains.append(100 * g)
            if p["misses"]:
                mr.append(b["misses"] / p["misses"])
            if p["bytes_copied"]:
                br.append(b["bytes_copied"] / p["bytes_copied"])
            if p["wait_s"] > 0:
                wr.append(b["wait_s"] / p["wait_s"])
        med = lambda v: f"{st.median(v):.2f}" if v else "—"  # noqa: E731
        lines.append(f"| {base} | {n} | {ge} | {med(gains)} | {max(gains):.1f} | {med(mr)} | {med(br)} | {med(wr)} | {only} |")
    # minimum budget reaching a hit-rate threshold
    for thr in (0.5, 0.8):
        lines += ["", f"### Smallest budget (GB) reaching a {100 * thr:.0f}% hit rate", "",
                  "1000 adapters, uniform tiers; >20 = not reached at 20 GB.", "",
                  "| model | r_max | alpha | " + " | ".join(POLICIES) + " |", "|---|---:|---:|" + "---:|" * len(POLICIES)]
        for model in MODELS:
            for r_max in RMAX:
                for a in ALPHAS:
                    cells = []
                    for pol in POLICIES:
                        ok = [r["budget_gb"] for r in rows if r["model"] == model and r["r_max"] == r_max
                              and r["zipf_alpha"] == a and r["adapters"] == 1000 and r["rank_mix"] == "uniform"
                              and r["policy"] == pol and r["feasible"] and r["hit_rate"] >= thr]
                        cells.append(f"{min(ok):g}" if ok else ">20")
                    lines.append(f"| {model} | {r_max} | {a} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def _figures(rows, fig_dir, tag):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {"vllm": ("#777777", ":", "s"), "slora": ("#1f77b4", "-.", "^"),
             "rf_whole": ("#ff7f0e", "--", "D"), "rf_prefix": ("#d62728", "-", "o")}
    for metric, ylabel, fname, logy in (
        ("hit_rate", "hit rate", "hit_rate", False),
        ("gb_copied", "GB copied (20k requests)", "gb_copied", True),
        ("mean_wait_ms", "mean load wait per request (ms)", "wait", True),
        ("resident_adapters", "adapters resident (mean)", "resident", True),
    ):
        for mix in ("uniform", "skew_low"):
            fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
            for i, model in enumerate(MODELS):
                for j, r_max in enumerate(RMAX):
                    ax = axes[i, j]
                    for pol in POLICIES:
                        sub = sorted((r for r in rows if r["model"] == model and r["r_max"] == r_max
                                      and r["adapters"] == 1000 and r["zipf_alpha"] == 1.0 and r["rank_mix"] == mix
                                      and r["policy"] == pol and r["feasible"]), key=lambda r: r["budget_gb"])
                        if not sub:
                            continue
                        c, ls, mk = style[pol]
                        y = [max(r[metric], 1e-3) if logy else r[metric] for r in sub]
                        ax.plot([r["budget_gb"] for r in sub], y, color=c, ls=ls, marker=mk, ms=4, label=LABEL[pol])
                    ax.set_xscale("log", base=2)
                    if logy:
                        ax.set_yscale("log")
                    ax.set_title(f"Qwen2.5-{model}, r_max {r_max}", fontsize=10)
                    ax.grid(True, alpha=0.3)
                    if i == 1:
                        ax.set_xlabel("adapter memory budget (GB)")
                    if j == 0:
                        ax.set_ylabel(ylabel)
            axes[0, 0].legend(fontsize=8)
            fig.suptitle(f"{ylabel}: 1000 adapters, Zipf 1.0, rank tiers {mix}", fontsize=11)
            fig.tight_layout()
            fig.savefig(fig_dir / f"{fname}_{mix}{tag}.png", dpi=130)
            plt.close(fig)


if __name__ == "__main__":
    main()
