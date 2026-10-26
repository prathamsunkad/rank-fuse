"""Slide tables for the baseline discussion (B1, B2g vs RF1, RF2), from the 26 Oct A100 sweeps.

    python results/a100_2026-10-26/slide_tables.py      (needs torch: recomputes rank-group counts)
Writes slides/*.md and slides/*.csv next to this script. Uniform rank mix unless noted; times are
median us per call over seeds (graph timing); ratios > 1 = the first one is faster.
"""
import csv
import glob
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from rankfuse.workloads import Workload, sample_request_ranks  # noqa: E402

RAW = HERE / "source"  # copies of the results/raw CSVs these tables were built from
OUT = HERE / "slides"
OUT.mkdir(exist_ok=True)
NAMES = {"b1_padded": "B1", "b2_grouped_graphed": "B2g", "b2_grouped": "B2", "rf1_fused": "RF1", "rf2_splitk": "RF2"}


def load(sweep):
    (path,) = glob.glob(str(RAW / f"{sweep}-*A100*.csv"))
    g = defaultdict(lambda: defaultdict(list))
    meta = {}
    for r in csv.DictReader(open(path)):
        k = (int(r["num_requests"]), int(r["tokens_per_request"]), int(r["r_max"]), r["dist"])
        if r["impl"] in NAMES:
            g[k][NAMES[r["impl"]]].append(float(r["median_us"]))
        tiers = tuple(int(t) for t in r["tiers"].split("/"))
        w = Workload(k[0], k[1], r_max=k[2], dist=k[3], seed=int(r["seed"]), tiers=tiers, h_in=8, h_out=8)
        ranks = sample_request_ranks(w, torch.Generator().manual_seed(w.seed + 1))  # as make_batch draws them
        meta.setdefault(k, []).append((int(r["batch_max_rank"]), len(set(ranks))))
        gpu = r["gpu"]
    return {k: {i: st.median(v) for i, v in d.items()} for k, d in g.items()}, meta, gpu


def x(a, b):
    return f"{a / b:.2f}x"


def seeds(vals):
    vals = sorted(set(vals))
    return str(vals[0]) if len(vals) == 1 else "/".join(map(str, vals))


def write(name, title, note, cols, rows):
    with open(OUT / f"{name}.csv", "w", newline="") as f:
        csv.writer(f).writerows([cols] + rows)
    md = [f"### {title}", "", note, "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    (OUT / f"{name}.md").write_text("\n".join(md) + "\n")
    return "\n".join(md)


TIMES = ["B1 us", "B2g us", "RF1 us", "RF2 us", "RF1 vs B1", "B2g vs B1", "RF2 vs RF1"]


def time_cols(t):
    return [f"{t['B1']:.1f}", f"{t['B2g']:.1f}", f"{t['RF1']:.1f}", f"{t['RF2']:.1f}",
            x(t["B1"], t["RF1"]), x(t["B1"], t["B2g"]), x(t["RF1"], t["RF2"])]


rep, rep_meta, rep_gpu = load("h200_replica")
r64, m64, gpu40 = load("rf_vs_b1_r64")
r128, m128, _ = load("rf_vs_b1_r128")
docs = []

# 1. Decode scaling, r_max 64.
rows = []
for n in [1, 8, 32, 128, 512, 2048, 8192]:
    k = (n, 1, 64, "uniform")
    rows.append([n, seeds(m for m, _ in rep_meta[k]), seeds(g for _, g in rep_meta[k])] + time_cols(rep[k]))
docs.append(write("01_decode_r64", "Decode, r_max 64: baselines vs RankFuse",
                  f"{rep_gpu}, 4096 x 4096, fp16, 7 tiers (1..64), uniform mix, median of 3 seeds "
                  "(max rank and rank groups listed per seed when they differ).",
                  ["tokens", "max rank", "rank groups"] + TIMES, rows))

# 2. Long-prompt prefill, r_max 64.
rows = []
for n, t in [(1, 2048), (4, 512), (16, 128), (4, 2048), (8, 1024), (16, 512)]:
    k = (n, t, 64, "uniform")
    rows.append([f"{n} x {t}", n * t, seeds(m for m, _ in rep_meta[k]), seeds(g for _, g in rep_meta[k])]
                + time_cols(rep[k]))
docs.append(write("02_prefill_r64", "Prefill (long prompts), r_max 64: baselines vs RankFuse",
                  f"{rep_gpu}, 4096 x 4096, fp16, 7 tiers, uniform mix, median of 3 seeds. With 1-16 "
                  "requests the max rank depends on the draw, so RF1 is sometimes already narrow.",
                  ["requests x tokens", "tokens", "max rank", "rank groups"] + TIMES, rows))

# 3. r_max 64 vs 128 at the same shapes (where rank skipping starts to pay).
rows = []
for phase, n, t in [("decode", 64, 1), ("decode", 256, 1), ("prefill", 16, 16), ("prefill", 64, 16), ("prefill", 256, 16)]:
    for rmax, d, meta in [(64, r64, m64), (128, r128, m128)]:
        for mix in ["uniform", "skew_low", "all_max"]:
            k = (n, t, rmax, mix)
            rows.append([phase, f"{n} x {t}", n * t, rmax, mix, meta[k][0][0], meta[k][0][1]] + time_cols(d[k]))
docs.append(write("03_rmax_64_vs_128", "r_max 64 vs 128 at the same batch shapes",
                  f"{gpu40}, 4096 x 4096, fp16, tiers r_max/8, /4, /2, full, seed 0. all_max = nothing to "
                  "skip (control). Prefill here = 16-token requests.",
                  ["phase", "requests x tokens", "tokens", "r_max", "mix", "max rank", "rank groups"] + TIMES, rows))

# 4. Why B2 is slow: cost vs number of rank groups.
by_groups = defaultdict(list)
for d, meta in [(r64, m64), (r128, m128)]:
    for k, t in d.items():
        for _, groups in meta[k][:1]:
            size = "<=64 tokens" if k[0] * k[1] <= 64 else "256-4096 tokens"
            by_groups[(size, groups)].append((t["B1"] / t["B2g"], t["B2g"], t["B1"], t["B2"]))
rows = []
for (size, groups), v in sorted(by_groups.items(), key=lambda kv: (kv[0][0] != "<=64 tokens", kv[0][1])):
    rows.append([size, groups, len(v), f"{st.median(a for a, *_ in v):.2f}x",
                 f"{st.median(b for _, b, _, _ in v):.1f}", f"{st.median(c for *_, c, _ in v):.1f}",
                 f"{st.median(e for *_, e in v):.0f}"])
docs.append(write("04_b2_vs_rank_groups", "Why B2 is slow: cost grows with the number of rank groups",
                  f"{gpu40}, rf_vs_b1 r_max 64 + 128 (seed 0), all mixes pooled. B2g = grouping built once, "
                  "then gather + 2 matmuls + scatter per group replayed from a CUDA graph; B2 = the same "
                  "in eager PyTorch, regrouping (with host syncs) every call.",
                  ["batch size", "rank groups", "workloads", "B2g vs B1 (median)", "B2g us", "B1 us", "B2 eager us"], rows))

(OUT / "ALL_SLIDES.md").write_text("\n\n".join(docs) + "\n")
print("\n\n".join(docs))
