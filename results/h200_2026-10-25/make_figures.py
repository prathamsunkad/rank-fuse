"""Draw the results figures (CUDA-graph timings only) from tables/*.csv.

    python make_figures.py          # writes figures/*.png (slides) and figures/*.pdf (report)

Needs matplotlib. Every number comes from tables/, which build_tables.py makes from source/.
Colours: a fixed, colour-blind-checked order (RF1 blue, B1 orange, B2 aqua, B3 yellow; RF2 variants
magenta / green / violet), validated with the dataviz palette checker; every bar carries a value label
because two of these colours sit below 3:1 contrast on white.
"""

import csv
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
TABLES, OUT = HERE / "tables", HERE / "figures"
OUT.mkdir(exist_ok=True)

C = {"RF1": "#2a78d6", "B1": "#eb6834", "B2": "#1baf7a", "B3": "#eda100",
     "RF2 sorted": "#e87ba4", "RF2 segment": "#008300", "RF2 bucketed": "#6250d6"}
INK, INK2, GRID, BASE = "#0b0b0b", "#52514e", "#e4e3df", "#c9c8c2"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlecolor": INK, "legend.frameon": False,
    "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
})
FOOT = "One NVIDIA H200, fp16, GPU time by CUDA-graph replay."


def rows(stem):
    with open(TABLES / f"{stem}.csv") as f:
        return list(csv.DictReader(f))


def num(v):
    return float(v) if v not in ("", None) else None


def grid_y(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def grid_x(ax):
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name, foot=FOOT):
    # wrap the caption to the figure's width so the saved image is no wider than the chart
    fig.text(0.01, -0.02, textwrap.fill(foot, int(fig.get_figwidth() * 15)), fontsize=9, color=INK2,
             ha="left", va="top")
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


def times_fmt(v, _):
    return f"{v:g}×"


# Fig 1: latency vs tokens, one layer --------------------------------------------------------------
t = [r for r in rows("01_rf1_vs_b1_one_layer") if r["rank mix"] == "uniform"]
tok = [int(r["tokens"]) for r in t]
fig, ax = plt.subplots(figsize=(10, 5.4))
series = [("B2", "B2 us", "B2: grouped by rank (PyTorch)"), ("B1", "B1 us", "B1: pad + mask (PyTorch)"),
          ("RF1 single", "RF1 one-kernel us", "RF1 one kernel (first design)"),
          ("RF1", "RF1 split-K us", "RF1 split-K (final)")]
END_DY = {"B1": -9, "RF1": 9}
for key, col, label in series:
    y = [num(r[col]) for r in t]
    style = dict(color="#8a8984", linestyle="--", linewidth=1.6) if key == "RF1 single" else \
        dict(color=C[key], linewidth=2.2)
    ax.plot(tok, y, marker="o", markersize=5, label=label, **style)
    ax.annotate(f"{key if key != 'RF1 single' else 'RF1 one kernel'} {y[-1]:.0f} µs", (tok[-1], y[-1]),
                xytext=(6, END_DY.get(key, 0)), textcoords="offset points", va="center", fontsize=9, color=INK2)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(tok)
ax.set_xticklabels([str(v) for v in tok])
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("tokens in the batch (one per request)")
ax.set_ylabel("time per call (µs, log scale)")
ax.set_title("One LoRA layer: RF1 split-K vs the PyTorch baselines")
grid_y(ax)
ax.legend(loc="upper left", fontsize=10)
save(fig, "fig1_latency_vs_tokens",
     FOOT + " 4096 × 4096 layer, r_max 64, uniform rank mix (ranks 1–64). Table 01.")

# Fig 2: RF1 speedup over each baseline, per workload ----------------------------------------------
t = rows("03_rf1_vs_b3_vllm_kernels")
keep = ["decode 1 x 1, uniform", "decode 8 x 1, uniform", "decode 32 x 1, uniform", "decode 128 x 1, uniform",
        "decode 2048 x 1, uniform", "decode 8192 x 1, uniform", "decode 32 x 1, all_max",
        "prompts 4 x 512, uniform", "prompts 4 x 2048, uniform",
        "prompts 15 x 512 at rank 4 + 1 x 512 at rank 64"]
nice = {"decode 32 x 1, all_max": "decode 32 × 1, all rank 64",
        "prompts 15 x 512 at rank 4 + 1 x 512 at rank 64": "prompts 15×512 @r4 + 1×512 @r64"}
t = [r for r in t if r["workload"] in keep]
t.sort(key=lambda r: keep.index(r["workload"]))
labels = [nice.get(r["workload"], r["workload"].replace(" x ", " × ")) for r in t]
fig, ax = plt.subplots(figsize=(10, 6.2))
h = 0.26
for j, (key, col) in enumerate([("B1", "RF1 speedup vs B1"), ("B2", "RF1 speedup vs B2"), ("B3", "RF1 speedup vs B3")]):
    ys = [i + (j - 1) * h for i in range(len(t))]
    vals = [num(r[col]) for r in t]
    ax.barh(ys, vals, height=h - 0.03, color=C[key], label=f"vs {key}", edgecolor="white", linewidth=0.8)
    for y, v in zip(ys, vals):
        ax.text(v * 1.03, y, f"{v:.1f}×", va="center", fontsize=8.5, color=INK2)
ax.set_xscale("log")
ax.set_xticks([1, 2, 5, 10, 20])
ax.xaxis.set_major_formatter(FuncFormatter(times_fmt))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.axvline(1, color=INK2, linewidth=1)
ax.set_yticks(range(len(t)))
ax.set_yticklabels(labels, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("RF1 speedup (× faster than the baseline, log scale)")
ax.set_title("RF1 split-K speedup over B1, B2 and vLLM's kernels (B3)")
grid_x(ax)
ax.legend(loc="lower right", fontsize=10)
save(fig, "fig2_speedup_vs_baselines",
     FOOT + " 4096 × 4096, r_max 64, delta added into the output. B3 = vLLM 0.32 kernels with 7 padded rank "
     "copies (default kernel configs). Table 03.")

# Fig 3: memory read and kernel count (ncu) --------------------------------------------------------
t = rows("04_memory_traffic_ncu")
order = [("RF1 split-K", "RF1"), ("B1 (PyTorch)", "B1"), ("B2 (PyTorch, grouped)", "B2"), ("B3 (vLLM kernels)", "B3")]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
for ax, tokens in zip(axes, ("32", "2048")):
    sub = {r["implementation"]: r for r in t if r["tokens"] == tokens}
    names = [k for k, _ in order if k in sub]
    vals = [num(sub[k]["DRAM read MB"]) for k in names]
    kern = [int(sub[k]["kernels"]) for k in names]
    xs = range(len(names))
    ax.bar(xs, vals, width=0.62, color=[C[s] for k, s in order if k in sub], edgecolor="white", linewidth=1)
    for x_, v, k in zip(xs, vals, kern):
        ax.text(x_, v * 1.02, f"{v:.1f} MB\n{k} kernels", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([s for k, s in order if k in sub])
    ax.set_ylim(0, max(vals) * 1.3)
    ax.set_title(f"{tokens} tokens")
    ax.set_ylabel("DRAM read per call (MB)")
    grid_y(ax)
fig.suptitle("Memory read and kernel launches per LoRA call (Nsight Compute)", fontweight="bold", color=INK, y=1.02)
save(fig, "fig3_memory_traffic",
     "Nsight Compute 2026.3, one H200, 4096 × 4096, r_max 64, uniform ranks, add into output; last of three calls. "
     "Table 04.")

# Fig 4: design steps -------------------------------------------------------------------------------
t1 = {(r["tokens"], r["rank mix"]): r for r in rows("01_rf1_vs_b1_one_layer")}
t6 = {r["tokens"]: r for r in rows("06_rf1_long_prompt_expand_fix")}
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
for ax, tokens in zip(axes, ("32", "8192")):
    r1 = t1[(tokens, "uniform")]
    steps = [("RF1 one kernel", num(r1["RF1 one-kernel us"])), ("+ split-K shrink", num(t6[tokens]["RF1 before us"])),
             ("+ tiled expand\n(final)", num(r1["RF1 split-K us"]))]
    xs = range(len(steps))
    vals = [v for _, v in steps]
    ax.bar(xs, vals, width=0.6, color=["#a9c8ef", "#6a9fe0", C["RF1"]], edgecolor="white", linewidth=1)
    for x_, v in zip(xs, vals):
        ax.text(x_, v * 1.02, f"{v:.1f} µs", ha="center", va="bottom", fontsize=10, color=INK2)
    b1 = num(r1["B1 us"])
    ax.axhline(b1, color=C["B1"], linestyle="--", linewidth=1.6, label=f"B1 (PyTorch) {b1:.1f} µs")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([s for s, _ in steps], fontsize=10)
    ax.set_ylim(0, max(vals + [b1]) * 1.25)
    ax.set_title(f"{tokens} tokens")
    ax.set_ylabel("time per call (µs)")
    grid_y(ax)
fig.suptitle("How RF1 got faster: one kernel → split-K → tiled expand", fontweight="bold", color=INK, y=1.02)
save(fig, "fig4_design_steps", FOOT + " 4096 × 4096, r_max 64, uniform ranks. Tables 01 and 06.")

# Fig 5: RF2 variants vs RF1, and the rank ceiling ---------------------------------------------------
seg = {r["batch"]: r for r in rows("08_rf2_segment")}
bkt = {r["batch"]: r for r in rows("09_rf2_bucketed")}
picks = [("15 x 512 @r4 + 1 x 512 @r64", "15×512 r4\n+1×512 r64"), ("7 x 1024 @r8 + 1 x 1024 @r64", "7×1024 r8\n+1×1024 r64"),
         ("mix 1/2/4/8/16/32/64 x 1024", "7×1024\nr1…r64"), ("32 x 256 uniform", "32×256\nuniform"),
         ("16 x 512 skew_low", "16×512\nmax rank 8"), ("32 x 1 uniform", "decode\n32 tok."),
         ("8192 x 1 uniform", "decode\n8192 tok.")]
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [2.4, 1]})
h = 0.26
for j, (key, src, col) in enumerate([("RF2 sorted", seg, "RF2 sorted us"), ("RF2 segment", seg, "RF2 segment (contig) us"),
                                     ("RF2 bucketed", bkt, "RF2 bucketed us")]):
    xs = [i + (j - 1) * h for i in range(len(picks))]
    vals = [num(seg[b]["RF1 split-K us"]) / num(src[b][col]) for b, _ in picks]
    ax.bar(xs, vals, width=h - 0.03, color=C[key], label=key, edgecolor="white", linewidth=0.8)
    for x_, v in zip(xs, vals):
        ax.text(x_, v + 0.004, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2, rotation=90)
ax.axhline(1, color=C["RF1"], linewidth=1.6)
ax.set_xticks(range(len(picks)))
ax.set_xticklabels([l for _, l in picks], fontsize=9.5)
ax.set_ylim(0.75, 1.27)
ax.set_ylabel("speedup over RF1 split-K (>1 = faster)")
ax.set_title("Rank skipping (RF2) vs RF1")
grid_y(ax)
handles, labels_ = ax.get_legend_handles_labels()
handles.append(Line2D([0], [0], color=C["RF1"], linewidth=1.6))
labels_.append("RF1 split-K = 1.0")
ax.legend(handles, labels_, loc="upper right", fontsize=9.5, ncol=4)
# ceiling: RF1 time vs the batch's highest rank at 8192 tokens
t7 = [r for r in rows("07_rf2_sorted_splitk") if r["tokens"] == "8192"]
by_rank = {}
for r in t7:
    by_rank.setdefault(int(r["max rank"]), []).append(num(r["RF1 split-K us"]))
ranks = sorted(by_rank)
vals = [sum(by_rank[k]) / len(by_rank[k]) for k in ranks]
ax2.bar(range(len(ranks)), vals, width=0.6, color=C["RF1"], edgecolor="white", linewidth=1)
for x_, v in enumerate(vals):
    ax2.text(x_, v + 0.6, f"{v:.0f}", ha="center", va="bottom", fontsize=9, color=INK2)
ax2.set_xticks(range(len(ranks)))
ax2.set_xticklabels([str(k) for k in ranks])
ax2.set_xlabel("highest rank in the batch")
ax2.set_ylabel("RF1 time per call (µs)")
ax2.set_ylim(0, max(vals) * 1.18)
ax2.set_title("Why the gain is capped: 8192 tokens")
grid_y(ax2)
save(fig, "fig5_rf2_vs_rf1",
     FOOT + " 4096 × 4096, r_max 64. Left: tables 08–09 (A×L @rR = A requests of L tokens at rank R). Right: "
     "table 07; rank work is only ~20% of RF1's time (49 µs at max rank ≤ 8 vs 61.5 µs at 64).")

# Fig 6: whole model, base + LoRA ---------------------------------------------------------------
t = rows("10_whole_model_graph")
wl = [("decode", "1 x 1"), ("decode", "8 x 1"), ("decode", "32 x 1"), ("decode", "128 x 1"),
      ("prefill", "4 x 512"), ("prefill", "16 x 128")]
sel = {(r["phase"], r["requests x tokens"]): r for r in t if r["rank mix"] == "uniform"}
methods = [("RF1", "RF1 ms"), ("B1", "B1 ms"), ("B2", "B2 ms"), ("B3", "B3 ms")]
fig, ax = plt.subplots(figsize=(12, 5.4))
w = 0.2
for i, key in enumerate(wl):
    r = sel[key]
    base = num(r["no LoRA ms"])
    for j, (m, col) in enumerate(methods):
        x_ = i + (j - 1.5) * w
        tot = num(r[col])
        ax.bar(x_, base, width=w - 0.02, color=BASE, edgecolor="white", linewidth=0.8)
        ax.bar(x_, tot - base, bottom=base, width=w - 0.02, color=C[m], edgecolor="white", linewidth=0.8,
               label=m if i == 0 else None)
        ax.text(x_, tot + 0.15, f"{tot:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
ax.bar([0], [0], color=BASE, label="base model (no LoRA)")
ax.set_xticks(range(len(wl)))
ax.set_xticklabels([f"{p}\n{k.replace(' x ', ' × ')}" for p, k in wl])
ax.set_ylabel("forward pass time (ms)")
ax.set_title("Whole Qwen2.5-0.5B forward pass: base model + LoRA (all 168 projections)")
grid_y(ax)
ax.legend(loc="upper left", fontsize=9.5, ncol=5)
save(fig, "fig6_whole_model",
     FOOT + " Qwen2.5-0.5B-Instruct in PyTorch, one shared rank-64 Matryoshka adapter, uniform rank mix; "
     "decode on a 128-token KV cache. Table 10.")

# Fig 7: LoRA share of a stock vLLM step (graphs on) -------------------------------------------------
t = [r for r in rows("12_vllm_lora_share") if r["CUDA graphs"] == "graphs"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, phase, xl in zip(axes, ("decode step", "prefill"), ("decode batch size (requests)", "prompt length (tokens)")):
    sub = [r for r in t if r["phase"] == phase]
    xs = range(len(sub))
    base = [num(r["base ms"]) for r in sub]
    lora = [num(r["rank 64 ms"]) - b for r, b in zip(sub, base)]
    ax.bar(xs, base, width=0.6, color=BASE, edgecolor="white", linewidth=1, label="base model")
    ax.bar(xs, lora, bottom=base, width=0.6, color="#5f5e5a", edgecolor="white", linewidth=1, label="LoRA (rank 64)")
    for x_, r in zip(xs, sub):
        ax.text(x_, num(r["rank 64 ms"]) * 1.02, f"{num(r['LoRA % r64']):.0f}%", ha="center", va="bottom",
                fontsize=9.5, color=INK2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([r["batch or prompt tokens"] for r in sub])
    ax.set_xlabel(xl)
    ax.set_ylabel("time (ms)")
    ax.set_title("decode step" if phase == "decode step" else "prefill (one request)")
    ax.set_ylim(0, max(num(r["rank 64 ms"]) for r in sub) * 1.18)
    grid_y(ax)
axes[0].legend(loc="upper left", fontsize=9.5)
fig.suptitle("How much of a stock vLLM step is LoRA (% on top of each bar)", fontweight="bold", color=INK, y=1.02)
save(fig, "fig7_vllm_lora_share",
     "Stock vLLM 0.32.0, Qwen2.5-0.5B-Instruct, one H200, CUDA graphs on, prefix caching off; wall-clock step time. "
     "Table 12.")

# Fig 8 / 9: whole model, uniform ranks: base vs every method, decode and prefill separately ----------
t = {(r["phase"], r["requests x tokens"]): r for r in rows("10_whole_model_graph") if r["rank mix"] == "uniform"}
METHODS = [("Base model (no LoRA)", "no LoRA ms", "#c9c8c2"), ("B1", "B1 ms", C["B1"]), ("B2", "B2 ms", C["B2"]),
           ("B3 (vLLM kernels)", "B3 ms", C["B3"]), ("RF1 split-K", "RF1 ms", C["RF1"]),
           ("RF2 split-K", "RF2 sorted split-K ms", C["RF2 sorted"])]
for name, phase, groups, xlabel in (
        ("fig8_whole_model_decode", "decode", ["1 x 1", "8 x 1", "32 x 1", "128 x 1"], "decode batch (requests, 1 new token each)"),
        ("fig9_whole_model_prefill", "prefill", ["4 x 512", "16 x 128"], "prefill batch (requests × prompt tokens)")):
    fig, ax = plt.subplots(figsize=(11, 5.4))
    w = 0.135
    for j, (label, col, color) in enumerate(METHODS):
        xs = [i + (j - 2.5) * w for i in range(len(groups))]
        vals = [num(t[(phase, g)][col]) for g in groups]
        ax.bar(xs, vals, width=w - 0.012, color=color, label=label, edgecolor="white", linewidth=0.8)
        for x_, v in zip(xs, vals):
            ax.text(x_, v + max(vals) * 0.012, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g.replace(" x ", " × ") for g in groups], fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("forward pass time (ms)")
    top = max(num(t[(phase, g)][c]) for g in groups for _, c, _ in METHODS)
    ax.set_ylim(0, top * 1.12)
    ax.set_title(f"Whole Qwen2.5-0.5B forward pass, {phase}: base model vs LoRA methods (uniform ranks)")
    grid_y(ax)
    ax.legend(loc="upper left", fontsize=9.5, ncol=3)
    save(fig, name, FOOT + " Qwen2.5-0.5B-Instruct in PyTorch, LoRA on all 168 projections (q,k,v,o,gate,up,down), "
         "one shared rank-64 Matryoshka adapter, uniform rank mix" +
         ("; one new token per request on a 128-token KV cache." if phase == "decode" else "; whole prompts, no KV cache.")
         + " Table 10.")
