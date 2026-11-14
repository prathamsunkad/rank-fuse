"""Tables for the standardized RF1 / RF2 sweeps: cost of the RF2 path, skipping alone, net gain.

    python analysis/rf2_eval.py results/raw/rf2/*.csv --baseline results/raw/baseline_pre.csv \
        --out results/a100_<date>/REPORT.md

Ratios are time(second) / time(first), so above 1 means the first kernel in the label is faster.
Each workload is compared on the GPU it ran on, with the seeds' medians first, and each table
keeps one GPU model. rf1_fused listed twice per workload gives the noise floor: a ratio counts
as a win or a loss only outside the 90th percentile of |pass 1 / pass 0 - 1| for its token range.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

FW = "rf2_segment_fw"
SEG = "rf2_segment"
IDX = "rf2_index"
OVERHEAD_STEPS = ["rf2_probe_read", "rf2_probe_branch", FW, "rf2_index_fw"]
COMPARISONS = [
    ("Cost of the RF2 path: rf1_fused vs full-width segment", "rf1_fused", FW, {}),
    ("Skipping alone: segment vs its full-width control", SEG, FW, {}),
    ("Net: segment vs rf1_fused", SEG, "rf1_fused", {}),
    ("Net: segment vs B1", SEG, "b1_padded", {}),
    ("Skipping alone: sorted-index vs its full-width control", IDX, "rf2_index_fw", {}),
    ("Index cost: segment vs sorted-index (read the sorted rows: same blocks)", SEG, IDX, {}),
    ("Net: sorted-index vs rf1_fused", IDX, "rf1_fused", {}),
    ("Net: sorted-index vs B1", IDX, "b1_padded", {}),
    ("Bucketed vs segment", "rf2_bucketed", SEG, {}),
    ("Bucketed: extra tuning", "rf2_bucketed_tuned", "rf2_bucketed", {}),
    ("Bucketed: concurrent streams vs one after another", "rf2_bucketed", "rf2_bucketed_seq", {}),
]
KEY = ["gpu", "mode", "shape", "num_requests", "tokens_per_request", "lengths", "r_max", "tiers", "dist", "prefill_dist", "request_order", "dtype"]
TOKEN_RANGES = [(1, 128, "<=128"), (129, 1024, "129-1024"), (1025, 4096, "1025-4096"), (4097, 10**9, ">4096")]


def token_range(tokens: int) -> str:
    return next(label for lo, hi, label in TOKEN_RANGES if lo <= tokens <= hi)


def load(paths: list[str], timing: str = "graph") -> list[dict]:
    """One dict per workload and seed: its fields plus {'t': {(impl, pass): us}, 'prepare': {impl: us}}."""
    runs: dict[tuple, dict] = {}
    for path in paths:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("timing") != timing:
                    continue
                key = (*(r.get(k, "") for k in KEY), r["seed"])
                run = runs.setdefault(key, {**{k: r.get(k, "") for k in KEY}, "seed": r["seed"], "tokens": int(r["num_tokens"]),
                                            "mean_rank": float(r["mean_rank"]), "t": {}, "prepare": {}})
                run["t"][r["impl"], int(r.get("impl_pass") or 0)] = float(r["median_us"])
                if r.get("prepare_us"):
                    run["prepare"][r["impl"]] = float(r["prepare_us"])
    return list(runs.values())


def by_workload(runs: list[dict]) -> dict[tuple, list[dict]]:
    groups = defaultdict(list)
    for run in runs:
        groups[tuple(run[k] for k in KEY)].append(run)
    return groups


def noise_floor(runs: list[dict]) -> dict[tuple[str, str], float]:
    """(gpu, token range) -> 90th percentile of the A/A difference between two rf1_fused passes."""
    diffs = defaultdict(list)
    for run in runs:
        t = run["t"]
        if ("rf1_fused", 1) in t:
            diffs[run["gpu"], token_range(run["tokens"])].append(abs(t["rf1_fused", 1] / t["rf1_fused", 0] - 1))
    return {k: statistics.quantiles(v, n=10)[-1] if len(v) > 1 else v[0] for k, v in diffs.items()}


def ratio(group: list[dict], a: str, b: str) -> float | None:
    """time(b) / time(a), median over seeds; rf1_fused uses its first pass."""
    values = [run["t"][b, 0] / run["t"][a, 0] for run in group if (a, 0) in run["t"] and (b, 0) in run["t"]]
    return statistics.median(values) if values else None


def rows_to_md(header: list[str], rows: list[list]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def comparison_table(groups: dict[tuple, list[dict]], gpu: str, floor: dict, a: str, b: str, by: list[str]) -> str | None:
    cells = defaultdict(list)
    for key, group in groups.items():
        run = group[0]
        value = ratio(group, a, b)
        if run["gpu"] == gpu and value is not None:
            cells[(*(run[k] for k in by), token_range(run["tokens"]))].append(value)
    if not cells:
        return None
    rows = []
    for cell in sorted(cells, key=lambda c: (tuple(int(x) if str(x).isdigit() else x for x in c[:-1]), [t[2] for t in TOKEN_RANGES].index(c[-1]))):
        v = cells[cell]
        tie = floor.get((gpu, cell[-1]), 0.02)
        rows.append([*cell, len(v), f"{statistics.median(v):.3f}x", f"{min(v):.2f}x", f"{max(v):.2f}x", sum(x > 1 + tie for x in v),
                     sum(abs(x - 1) <= tie for x in v), sum(x < 1 - tie for x in v)])
    return rows_to_md([*by, "tokens", "workloads", "median", "min", "max", "wins", "ties", "losses"], rows)


def capture_table(groups: dict[tuple, list[dict]], gpu: str) -> str | None:
    """Share of the gap from full width to ideal skipping that each kernel captures.

    Ideal = f * t(all at the low rank) + (1 - f) * t(all at r_max), f = low-rank token share, from
    rf2_segment_fw on the same batch, r_max and mode. A linear estimate, not a measured bound.
    """
    ends = {}
    for group in groups.values():
        run = group[0]
        if run["dist"] in ("all_min", "all_max"):
            k = (run["gpu"], run["mode"], run["shape"], run["num_requests"], run["tokens_per_request"], run["r_max"], run["dist"])
            times = [r["t"][FW, 0] for r in group if (FW, 0) in r["t"]]
            if times:
                ends[k] = statistics.median(times)
    cells = defaultdict(list)
    for group in groups.values():
        run = group[0]
        if run["gpu"] != gpu or not run["dist"].startswith("low_frac_"):
            continue
        k = (run["gpu"], run["mode"], run["shape"], run["num_requests"], run["tokens_per_request"], run["r_max"])
        low, high = ends.get((*k, "all_min")), ends.get((*k, "all_max"))
        if low is None or high is None or high <= low:
            continue
        r_max = int(run["r_max"])
        for impl, label in ((SEG, "segment"), ("rf2_bucketed", "bucketed")):
            done = []
            for r in group:
                if (impl, 0) in r["t"] and (FW, 0) in r["t"]:
                    f = (r_max - r["mean_rank"]) / (r_max - 1)
                    ideal = f * low + (1 - f) * high
                    done.append((r["t"][FW, 0] - r["t"][impl, 0]) / max(r["t"][FW, 0] - ideal, 1e-9))
            if done:
                cells[run["r_max"], run["dist"], run["request_order"], label].append(statistics.median(done))
    if not cells:
        return None
    rows = [[*c, len(v), f"{100 * statistics.median(v):.0f}%"] for c, v in sorted(cells.items(), key=lambda kv: (int(kv[0][0]), *kv[0][1:]))]
    return rows_to_md(["r_max", "mix", "order", "kernel", "batches", "median share captured"], rows)


def ordering_table(groups: dict[tuple, list[dict]], gpu: str, floor: dict) -> str | None:
    """rf2_segment on the same batches, time(random) / time(sorted); the sort itself is not included."""
    order_at = KEY.index("request_order")
    pairs = defaultdict(dict)
    for key, group in groups.items():
        times = [r["t"][SEG, 0] for r in group if (SEG, 0) in r["t"]]
        if group[0]["gpu"] == gpu and times:
            pairs[key[:order_at] + key[order_at + 1:]][group[0]["request_order"]] = (statistics.median(times), group[0])
    cells = defaultdict(list)
    for sides in pairs.values():
        if "random" in sides and "sorted" in sides:
            run = sides["random"][1]
            cells[run["r_max"], token_range(run["tokens"])].append(sides["random"][0] / sides["sorted"][0])
    if not cells:
        return None
    rows = []
    for c, v in sorted(cells.items(), key=lambda kv: (int(kv[0][0]), [t[2] for t in TOKEN_RANGES].index(kv[0][1]))):
        tie = floor.get((gpu, c[1]), 0.02)
        rows.append([*c, len(v), f"{statistics.median(v):.3f}x", f"{min(v):.2f}x", f"{max(v):.2f}x", sum(x > 1 + tie for x in v),
                     sum(abs(x - 1) <= tie for x in v), sum(x < 1 - tie for x in v)])
    return rows_to_md(["r_max", "tokens", "workloads", "median", "min", "max", "sorted faster", "ties", "random faster"], rows)


def _median_us(group: list[dict], impl: str) -> float | None:
    values = [run["t"][impl, 0] for run in group if (impl, 0) in run["t"]]
    return statistics.median(values) if values else None


def overhead_table(groups: dict[tuple, list[dict]], gpu: str) -> str | None:
    """One-rank batches: rf1_fused's time and what each step of the RF2 path adds, in us."""
    rows = []
    for key, group in groups.items():
        run = group[0]
        base = _median_us(group, "rf1_fused")
        if run["gpu"] != gpu or run["dist"] not in ("all_min", "all_max") or base is None:
            continue
        steps = [_median_us(group, impl) for impl in OVERHEAD_STEPS]
        if all(v is None for v in steps):
            continue
        rows.append([int(run["r_max"]), run["dist"], run["tokens"], f'{run["num_requests"]}x{run["tokens_per_request"] or "-"}', f"{base:.1f}",
                     *("" if v is None else f"{v - base:+.2f} ({100 * (v / base - 1):+.1f}%)" for v in steps)])
    if not rows:
        return None
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    return rows_to_md(["r_max", "mix", "tokens", "batch", "rf1_fused us", *(f"{s} vs rf1" for s in OVERHEAD_STEPS)], rows)


def assumption_table(groups: dict[tuple, list[dict]], gpu: str) -> str | None:
    """Mixed batches under the two scheduler assumptions: physically sorted tokens (rf2_segment on
    sorted batches) and a sorted index list (rf2_index on random-order batches, sort reported apart)."""
    pairs = defaultdict(dict)
    order_at = KEY.index("request_order")
    for key, group in groups.items():
        if group[0]["gpu"] == gpu and group[0]["dist"] not in ("all_min", "all_max"):
            pairs[key[:order_at] + key[order_at + 1:]][group[0]["request_order"]] = group
    rows = []
    for both in pairs.values():
        if "sorted" not in both or "random" not in both:
            continue
        srt, rnd = both["sorted"], both["random"]
        rf1, b1 = _median_us(rnd, "rf1_fused"), _median_us(rnd, "b1_padded")
        seg, idx = _median_us(srt, SEG), _median_us(rnd, IDX)
        if rf1 is None or seg is None or idx is None:
            continue
        sort_us = [run["prepare"][IDX] - run["prepare"].get(SEG, 0.0) for run in rnd if IDX in run["prepare"]]
        run = rnd[0]
        rows.append([int(run["r_max"]), run["dist"], f'{run["num_requests"]}x{run["tokens_per_request"] or "-"}', run["tokens"],
                     f"{rf1:.1f}", f"{rf1 / seg:.2f}x", f"{rf1 / idx:.2f}x",
                     "" if b1 is None else f"{b1 / seg:.2f}x", "" if b1 is None else f"{b1 / idx:.2f}x",
                     f"{statistics.median(sort_us):.0f}" if sort_us else ""])
    if not rows:
        return None
    rows.sort(key=lambda r: (r[0], r[1], r[3]))
    return rows_to_md(["r_max", "mix", "batch", "tokens", "rf1_fused us", "sorted tokens vs RF1", "sorted index vs RF1",
                       "sorted tokens vs B1", "sorted index vs B1", "sort us (host, per step)"], rows)


def baseline_table(runs: list[dict], baseline: list[dict], gpu: str) -> str | None:
    now = {(r["shape"], r["num_requests"], r["tokens_per_request"], r["r_max"], r["tiers"], r["dist"], r["seed"]): r
           for r in runs if r["gpu"] == gpu and r["mode"] == "write" and r["request_order"] in ("random", "")}
    diffs = defaultdict(list)
    for old in baseline:
        k = (old["shape"], old["num_requests"], old["tokens_per_request"], old["r_max"], old["tiers"], old["dist"], old["seed"])
        new = now.get(k)
        if old["gpu"] == gpu and new and ("rf1_fused", 0) in old["t"] and ("rf1_fused", 0) in new["t"]:
            diffs[old["r_max"], token_range(new["tokens"])].append(new["t"]["rf1_fused", 0] / old["t"]["rf1_fused", 0])
    if not diffs:
        return None
    rows = [[*k, len(v), f"{statistics.median(v):.3f}x", f"{min(v):.2f}x", f"{max(v):.2f}x"] for k, v in sorted(diffs.items(), key=lambda kv: int(kv[0][0]))]
    return rows_to_md(["r_max", "tokens", "workloads", "median new / old", "min", "max"], rows)


def launch_table(runs: list[dict], eager: list[dict], gpu: str) -> str | None:
    """Eager time minus graph time on one workload: launch overhead with metadata already built."""
    graph = {(*(r[k] for k in KEY), r["seed"]): r for r in runs if r["gpu"] == gpu}
    cells = defaultdict(list)
    for e in eager:
        g = graph.get((*(e[k] for k in KEY), e["seed"]))
        if g:
            for (impl, n), us in e["t"].items():
                if (impl, n) in g["t"] and n == 0:
                    cells[impl, token_range(e["tokens"])].append(us - g["t"][impl, n])
    if not cells:
        return None
    rows = [[*c, len(v), f"{statistics.median(v):.1f}"] for c, v in sorted(cells.items())]
    return rows_to_md(["kernel", "tokens", "workloads", "launch overhead us per call"], rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csvs", nargs="+")
    p.add_argument("--baseline", default=None, help="CSV from before the shared kernel, to check rf1_fused did not move")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    runs = load(a.csvs)
    baseline = load([a.baseline]) if a.baseline else []
    eager = load(a.csvs, "eager")
    groups = by_workload(runs)
    floor = noise_floor(runs)
    out = ["# RF1 and RF2 after standardization", "",
           "Above 1x means the first kernel named is faster. Wins and losses lie outside the noise floor.", ""]
    for gpu in sorted({r["gpu"] for r in runs}):
        out += [f"## {gpu}", "", "### Noise floor (90th percentile of |rf1_fused pass 1 / pass 0 - 1|)", ""]
        out += [rows_to_md(["tokens", "noise"], [[l, f"{100 * floor[gpu, l]:.1f}%"] for _, _, l in TOKEN_RANGES if (gpu, l) in floor]), ""]
        table = baseline_table(runs, baseline, gpu)
        if table:
            out += ["### rf1_fused against the pre-refactor baseline", "", table, ""]
        one_rank = {k: g for k, g in groups.items() if g[0]["dist"] in ("all_min", "all_max")}
        table = comparison_table(one_rank, gpu, floor, "rf1_fused", FW, ["mode", "r_max", "dist"])
        if table:
            out += ["### Parity: nothing to skip, rf1_fused vs full-width segment", "", table, ""]
        mixed = {k: g for k, g in groups.items() if g[0]["dist"] not in ("all_min", "all_max")}
        for suite, keep in (("Ideal mixes", lambda d: d.startswith("low_frac_")), ("Sampled mixes", lambda d: not d.startswith("low_frac_"))):
            subset = {k: g for k, g in mixed.items() if keep(g[0]["dist"])}
            for title, x, y, _ in COMPARISONS:
                table = comparison_table(subset, gpu, floor, x, y, ["shape", "r_max", "request_order"])
                if table:
                    out += [f"### {suite}. {title}", "", table, ""]
        table = capture_table(groups, gpu)
        if table:
            out += ["### Ideal mixes: share of the full-width-to-ideal gap captured (linear estimate)", "", table, ""]
        table = launch_table(runs, eager, gpu)
        if table:
            out += ["### Python launch overhead (eager minus graph)", "", table, ""]
        table = overhead_table(groups, gpu)
        if table:
            out += ["### Overhead with nothing to skip, in us (one-rank batches)", "", table, ""]
        table = assumption_table(groups, gpu)
        if table:
            out += ["### Two scheduler assumptions: speedup of RF2 (above 1x is faster)", "", table, ""]
        table = ordering_table(mixed, gpu, floor)
        if table:
            out += ["### Does ordering requests help? rf2_segment, sorted vs random (bench/sort_cost.py has the sort's cost)", "", table, ""]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(out))
    print(f"{len(runs)} workload-seed runs -> {a.out}")


if __name__ == "__main__":
    main()
