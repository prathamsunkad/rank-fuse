"""Speedup tables for the mixed-batching (chunked-prefill) sweeps.

    D=results/a100_2026-11-23_mixed
    python analysis/mixed_tables.py --single $D/raw/single-mixed-r{64,128,256}.csv \
        --multi $D/raw/multi-mixed-r{64,128,256}.jsonl --out $D/SUMMARY.md

Prepare-cost files (`<sweep stem>-prepare.csv`, bench/mixed_prepare_cost.py) next to the sweep files
are picked up automatically.

Kernel-only speedup of a RankFuse kernel over a baseline = baseline median_us / RankFuse median_us on
the same workload (same batch, seed and request order). "incl." speedup also charges each side its
per-step sort / prepare cost, amortized over one model's LoRA calls per step:
    (base_us + base_step_us / n) / (rf_us + rf_step_us / n),  n = 24 x 7 = 168 (Qwen2.5-0.5B) or 28 x 7 = 196 (7B)
with the kernel times of the 4096 x 4096 layer (an approximation: the models' projections are
smaller, so the real prepare share is somewhat larger). Per-step cost charged:
    rf2_ranges, sorted order   sorted_path_host (sort the requests by rank + RangePlan, host)
    rf2_segment, sorted order  sort_requests_host + its prepare (batch max rank)
    rf2_ranges, random order   prepare:rf2_ranges (RangePlan from GPU ranks; a single rf1_fused run)
    any other impl             its prepare (wall_us); B3/B4-multi: per-step metadata only (step_meta:)
For each (r_max, step type, budget) group: median speedup over the group's workloads, p10-p90, the
minimum, the share of workloads where RankFuse is slower (< 1) and "seed spread" = the median over the
group's configurations of (max - min) / median of the speedup across the three seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

STEP_ORDER = ["decode_only", "one_prompt_fills", "short_prompts", "partial_prefills"]
CALLS = {"0.5B": 24 * 7, "7B": 28 * 7}
B4_NOTE = ("**B4 caveat:** S-LoRA's bgmv decode kernel applied to *all* tokens, prompts included. Real S-LoRA "
           "runs prompts through a separate prefill kernel, so on prefill-heavy steps this is a model of S-LoRA, "
           "not its exact behaviour (it overstates S-LoRA's cost there).")


def step_type(lengths: str | None) -> str:
    if not lengths:
        return "decode_only"
    if lengths.startswith("serving_"):
        return "short_prompts"
    b, c = lengths.removeprefix("chunked_").split("_")
    return "one_prompt_fills" if b == c else "partial_prefills"


def budget(lengths: str | None) -> str:
    if not lengths:
        return "-"
    return lengths.split("_")[1] if lengths.startswith("chunked_") else lengths.split("_")[2]


def q(v: list[float], p: float) -> float:
    v = sorted(v)
    k = (len(v) - 1) * p
    lo = int(k)
    return v[lo] + (v[min(lo + 1, len(v) - 1)] - v[lo]) * (k - lo)


def sort_key(vals) -> tuple:
    return tuple(STEP_ORDER.index(v) if v in STEP_ORDER else (int(v) if str(v).isdigit() else -1 if v == "-" else str(v))
                 for v in vals)


class Data:
    """Kernel times and per-step costs keyed by (impl, *config, seed)."""

    def __init__(self, rows: list[dict], costs: dict, cfg_keys: list[str], cost_of):
        self.t, self.group_row, self.cfg_keys = {}, {}, cfg_keys
        self.costs, self.cost_of = costs, cost_of
        for r in rows:
            if r["median_us"] != r["median_us"]:
                continue
            cfg = tuple(r[k] for k in cfg_keys)
            self.t[(r["impl"], *cfg, r["seed"])] = r["median_us"]
            self.group_row[cfg] = r

    def step_cost(self, impl: str, cfg: tuple, seed: str) -> float | None:
        items = self.cost_of(impl, dict(zip(self.cfg_keys, cfg)))
        vals = [self.costs.get((it, *cfg, seed)) for it in items]
        return None if any(v is None for v in vals) else sum(vals)

    def summarize(self, rf: str, base: str, group_keys: list[str], rows_filter=None) -> list[dict]:
        by_group, by_cfg, incl = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(list))
        for (impl, *cfg, seed), us in self.t.items():
            if impl != rf:
                continue
            cfg = tuple(cfg)
            r = self.group_row[cfg]
            if rows_filter and not rows_filter(r):
                continue
            b = self.t.get((base, *cfg, seed))
            if b is None:
                continue
            g = tuple(r[k] for k in group_keys)
            by_group[g].append(b / us)
            by_cfg[(g, cfg)].append(b / us)
            cr, cb = self.step_cost(rf, cfg, seed), self.step_cost(base, cfg, seed)
            if cr is not None and cb is not None:
                for m, n in CALLS.items():
                    incl[g][m].append((b + cb / n) / (us + cr / n))
        spread = defaultdict(list)
        for (g, _), v in by_cfg.items():
            if len(v) > 1:
                spread[g].append((max(v) - min(v)) / statistics.median(v))
        out = []
        for g, v in by_group.items():
            out.append(dict(zip(group_keys, g), n=len(v), median=statistics.median(v), p10=q(v, 0.1), p90=q(v, 0.9),
                            min=min(v), lose=sum(s < 1 for s in v) / len(v),
                            seed_spread=statistics.median(spread[g]) if spread[g] else float("nan"),
                            **{f"incl_{m}": statistics.median(incl[g][m]) if incl[g][m] else float("nan") for m in CALLS},
                            incl_lose=(sum(s < 1 for s in incl[g]["7B"]) / len(incl[g]["7B"])) if incl[g]["7B"] else float("nan")))
        return out


def fmt_table(out: list[dict], group_keys: list[str], title: str, note: str | None = None) -> str:
    lines = [f"#### {title}", ""]
    if note:
        lines += [note, ""]
    lines += ["| " + " | ".join(group_keys) + " | n | kernel-only median | p10-p90 | min | slower | seed spread | "
              "incl. prepare 0.5B | incl. prepare 7B | slower incl. 7B |",
              "|" + "---|" * (len(group_keys) + 9)]
    for r in sorted(out, key=lambda r: sort_key([r[k] for k in group_keys])):
        lines.append("| " + " | ".join(str(r[k]) for k in group_keys)
                     + f" | {r['n']} | **{r['median']:.2f}x** | {r['p10']:.2f}-{r['p90']:.2f} | {r['min']:.2f} | "
                       f"{100 * r['lose']:.0f}% | {100 * r['seed_spread']:.1f}% | {r['incl_0.5B']:.2f}x | "
                       f"{r['incl_7B']:.2f}x | {100 * r['incl_lose']:.0f}% |")
    return "\n".join(lines) + "\n"


def norm(v) -> str:
    return "" if v is None or v == "None" else str(v)


def load_single(paths: list[str]) -> tuple[list[dict], list[dict]]:
    rows, cost_rows = [], []
    for p in paths:
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                if r["impl"] == "rf1_fused" and r["impl_pass"] != "0":
                    r["impl"] = "rf1_fused_pass1"
                r["median_us"] = float(r["median_us"])
                r["step_type"], r["budget"] = step_type(r["lengths"]), budget(r["lengths"])
                rows.append(r)
        cp = Path(p).with_name(Path(p).stem + "-prepare.csv")
        if cp.exists():
            with cp.open(newline="") as f:
                cost_rows += list(csv.DictReader(f))
    return rows, cost_rows


def load_multi(paths: list[str]) -> tuple[list[dict], list[dict]]:
    rows, cost_rows = [], []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            if r.get("error"):
                continue
            r["budget"], r["step_type"] = budget(r["lengths"]), step_type(r["lengths"])
            for k in ("r_max", "num_decodes", "num_adapters", "seed", "zipf_alpha", "prefill_dist", "lengths"):
                r[k] = norm(r[k])
            rows.append(r)
        cp = Path(p).with_name(Path(p).stem + "-prepare.csv")
        if cp.exists():
            with cp.open(newline="") as f:
                for r in csv.DictReader(f):
                    for k in ("prefill_dist", "lengths"):
                        r[k] = norm(r[k])
                    cost_rows.append(r)
    return rows, cost_rows


def cost_index(cost_rows: list[dict], cfg_keys: list[str]) -> dict:
    """(item, *config, seed) -> per-step us (wall_us, else gpu_us)."""
    out = defaultdict(list)
    for r in cost_rows:
        v = r["wall_us"] or r["gpu_us"]
        if v:
            out[(r["item"], *(r[k] for k in cfg_keys), r["seed"])].append(float(v))
    return {k: statistics.median(v) for k, v in out.items()}


def cost_table(cost_rows: list[dict], items: list[str], title: str, group_keys: list[str]) -> str:
    g = defaultdict(lambda: defaultdict(list))
    for r in cost_rows:
        if r["item"] in items:
            v = r["wall_us"] or r["gpu_us"]
            if v:
                key = tuple(budget(r["lengths"]) if k == "budget" else step_type(r["lengths"]) if k == "step_type" else r[k]
                            for k in group_keys)
                g[key][r["item"]].append(float(v))
    lines = [f"#### {title}", "",
             "Median per-step cost in us (wall = host + GPU with syncs; `*_gpu` = CUDA-graph GPU time; "
             "permute_per_call_gpu is per LoRA call, not per step). In brackets: amortized per LoRA call over "
             "7B's 196 calls.", "",
             "| " + " | ".join(group_keys) + " | " + " | ".join(items) + " |", "|" + "---|" * (len(group_keys) + len(items))]
    for k in sorted(g, key=sort_key):
        cells = []
        for it in items:
            v = g[k][it]
            if not v:
                cells.append("-")
            elif it == "permute_per_call_gpu":
                cells.append(f"{statistics.median(v):.1f}")
            else:
                cells.append(f"{statistics.median(v):.0f} ({statistics.median(v) / CALLS['7B']:.2f})")
        lines.append("| " + " | ".join(map(str, k)) + " | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def medians_table(rows, impls, group_keys, title, note=None) -> str:
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["impl"] in impls:
            g[tuple(r[k] for k in group_keys)][r["impl"]].append(r["median_us"])
    lines = [f"#### {title}", ""] + ([note, ""] if note else []) + [
        "| " + " | ".join(group_keys) + " | " + " | ".join(impls) + " |", "|" + "---|" * (len(group_keys) + len(impls))]
    for k in sorted(g, key=sort_key):
        lines.append("| " + " | ".join(map(str, k)) + " | "
                     + " | ".join(f"{statistics.median(g[k][i]):.1f}" if g[k][i] else "-" for i in impls) + " |")
    return "\n".join(lines) + "\n"


def add_no_reorder(rows: list[dict], costs: dict, ck: list[str]) -> None:
    """Synthetic impl "rf2_ranges+permute": the engine cannot reorder requests, so every LoRA call
    gathers x into rank order and adds the delta back (permute_per_call_gpu) around rf2_ranges' kernel
    on the sorted batch. Placed on the random-order workload, to compare with baselines on the batch
    as the engine gives it; its per-step cost is argsort_gpu + rangeplan_host (single_cost_of)."""
    new = []
    for r in rows:
        if r["impl"] != "rf2_ranges" or r["request_order"] != "sorted":
            continue
        cfg = tuple(r[k] for k in ck)
        perm = costs.get(("permute_per_call_gpu", *cfg, r["seed"]))
        if perm is None:
            continue
        n = dict(r, impl="rf2_ranges+permute", request_order="random", median_us=r["median_us"] + perm)
        new.append(n)
        rcfg = tuple(n[k] for k in ck)
        for it in ("argsort_gpu", "rangeplan_host"):
            v = costs.get((it, *cfg, r["seed"]))
            if v is not None:
                costs[(f"noreorder:{it}", *rcfg, r["seed"])] = v
    rows += new


def single_cost_of(impl: str, cfg: dict) -> list[str]:
    if impl == "rf2_ranges+permute":
        return ["noreorder:argsort_gpu", "noreorder:rangeplan_host"]
    if cfg["request_order"] == "sorted" and impl == "rf2_ranges":
        return ["sorted_path_host"]
    if cfg["request_order"] == "sorted" and impl == "rf2_segment":
        return ["sort_requests_host", "prepare:rf2_segment"]
    return [f"prepare:{impl}"]


def multi_cost_of(impl: str, cfg: dict) -> list[str]:
    return [f"step_meta:{impl}"] if impl in ("b3_multi", "b4_multi_copies") else [f"prepare:{impl}"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--single", nargs="*", default=[])
    p.add_argument("--multi", nargs="*", default=[])
    p.add_argument("--multi-rf", default="rf1_ma,rf1_ma_legacy,rf2_ma_splitk,rf2_ma_widths,rf2_ma_segment,rf2_ma_bucketed,"
                                         "rf2_ma_splitk_legacy,rf2_ma_segment_legacy,rf2_ma_bucketed_legacy")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    md = ["# Mixed-batching speedups (generated by analysis/mixed_tables.py)", "",
          "Speedup = baseline time / RankFuse time on the same batch (> 1: RankFuse faster). Kernel-only: CUDA-graph "
          "GPU time of the LoRA call. incl. prepare: each side also pays its per-step sort / prepare cost, amortized "
          "over the model's LoRA calls per step (0.5B: 168, 7B: 196); see analysis/mixed_tables.py.", ""]
    gk = ["r_max", "step_type", "budget"]
    if a.single:
        rows, cost_rows = load_single(a.single)
        ck = ["r_max", "num_requests", "dist", "request_order", "lengths", "prefill_dist"]
        costs = cost_index(cost_rows, ck)
        add_no_reorder(rows, costs, ck)
        d = Data(rows, costs, ck, single_cost_of)
        md += ["## Single adapter", "",
               "RankFuse = rf2_ranges. On sorted order: rows sorted by rank before timing (the scheduler reorders "
               "requests; its cost is in the incl. columns). On random order it is one rf1_fused launch. "
               "rf2_ranges+permute (random-order tables): the engine cannot reorder requests, so each LoRA call also "
               "gathers x into rank order and adds the delta back (GPU time, per call) around rf2_ranges' sorted kernel.", ""]
        noise = d.summarize("rf1_fused_pass1", "rf1_fused", ["r_max"])
        md.append("Noise (rf1_fused pass 1 vs pass 0, same workload): " + ", ".join(
            f"r_max {r['r_max']}: median {r['median']:.3f}, p10-p90 {r['p10']:.3f}-{r['p90']:.3f}" for r in
            sorted(noise, key=lambda r: int(r["r_max"]))) + "\n")
        if cost_rows:
            md.append(cost_table(cost_rows, ["sorted_path_host", "sort_requests_host", "rangeplan_host", "prepare:rf2_ranges",
                                             "prepare:rf1_fused", "prepare:rf2_index", "argsort_gpu", "permute_per_call_gpu",
                                             "prepare:b1_padded", "prepare:b3_vllm", "prepare:b4_slora_copies"],
                                  "Single adapter: per-step sort / prepare cost", gk))
        for order in ("sorted", "random"):
            flt = lambda r, o=order: r["request_order"] == o  # noqa: E731
            for rf in ("rf2_ranges", "rf2_ranges+permute", "rf1_fused", "rf2_index", "rf2_segment"):
                if (order == "random") == (rf in ("rf2_ranges", "rf2_segment")) and rf != "rf1_fused" and rf != "rf2_index" \
                        or (order == "sorted" and rf == "rf2_ranges+permute"):
                    continue
                for base in ("b1_padded", "b3_vllm", "b4_slora_copies", "b4_slora_shared", "rf1_fused"):
                    if base == rf:
                        continue
                    out = d.summarize(rf, base, gk, flt)
                    if out:
                        md.append(fmt_table(out, gk, f"{rf} vs {base}, request order {order}",
                                            B4_NOTE if base.startswith("b4_") else None))
            md.append(medians_table([r for r in rows if flt(r)],
                                    ["rf2_ranges", "rf1_fused", "rf2_index", "rf2_segment", "rf2_bucketed_tuned", "b1_padded",
                                     "b3_vllm", "b4_slora_copies", "b4_slora_shared"], gk,
                                    f"Single adapter: median kernel time (us), request order {order}", B4_NOTE))
    if a.multi:
        rows, cost_rows = load_multi(a.multi)
        ck = ["r_max", "num_decodes", "num_adapters", "zipf_alpha", "dist", "lengths", "prefill_dist"]
        d = Data(rows, cost_index(cost_rows, ck), ck, multi_cost_of)
        rfs = [r for r in a.multi_rf.split(",") if any(x["impl"] == r for x in rows)]
        md += ["## Multi adapter", "",
               "Batches in arrival order; each kernel's sort / grouping in its prepare (not in the kernel time; in the "
               "incl. columns).", ""]
        if cost_rows:
            md.append(cost_table(cost_rows, [f"prepare:{r}" for r in rfs] + ["prepare:b1_multi", "step_meta:b3_multi",
                                                                          "step_meta:b4_multi_copies", "prepare:b4_multi_shared"],
                                  "Multi adapter: per-step prepare cost", gk))
        for rf in rfs:
            for base in ("b1_multi", "b3_multi", "b4_multi_copies", "b4_multi_shared"):
                out = d.summarize(rf, base, gk)
                if out:
                    md.append(fmt_table(out, gk, f"{rf} vs {base}", B4_NOTE if base.startswith("b4_") else None))
            if f"{rf}_legacy" in rfs:  # new kernels (2c30e56) vs the Stage 2 ones they replace
                out = d.summarize(rf, f"{rf}_legacy", gk)
                if out:
                    md.append(fmt_table(out, gk, f"{rf} (new, shared split-K) vs {rf}_legacy (Stage 2)"))
            out = d.summarize(rf, "b3_multi", ["r_max", "step_type", "num_adapters"])
            if out:
                md.append(fmt_table(out, ["r_max", "step_type", "num_adapters"], f"{rf} vs b3_multi by adapter count"))
        md.append(medians_table(rows, rfs + ["b1_multi", "b3_multi", "b4_multi_copies", "b4_multi_shared"], gk,
                                "Multi adapter: median kernel time (us)", B4_NOTE))
    Path(a.out).write_text("\n".join(md))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
