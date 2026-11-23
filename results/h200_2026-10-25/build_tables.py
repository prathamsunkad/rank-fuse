"""Build every results table (CSV + Markdown) from the raw measurements in source/.

    python3 build_tables.py        # writes tables/*.csv, tables/*.md and ALL_TABLES.md

Standard library only. Every number in tables/ comes from a file in source/; nothing is typed in
by hand except the labels.
"""

import csv
import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC, OUT = HERE / "source", HERE / "tables"
OUT.mkdir(exist_ok=True)
TABLES = []  # (file stem, title, notes, header, rows) in order


def table(stem, title, notes, header, rows):
    TABLES.append((stem, title, notes, header, rows))
    with open(OUT / f"{stem}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    (OUT / f"{stem}.md").write_text(markdown(title, notes, header, rows))


def fmt(v):
    if isinstance(v, float):
        return f"{v:.2f}" if abs(v) < 100 else f"{v:.1f}"
    return "" if v is None else str(v)


def markdown(title, notes, header, rows):
    lines = [f"### {title}", "", notes, "", "| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def x(a, b):
    """Speedup a / b, rounded, as a float."""
    return round(a / b, 2) if a and b else None


def load(name):
    return json.loads((SRC / name).read_text())


TIMING_GRAPH = ("GPU time only: each call captured in a CUDA graph and replayed (median of 7 runs of "
                "50 replays). One NVIDIA H200, fp16, 4096 x 4096 layer, r_max 64 (tiers 1, 2, 4, ..., 64).")

# B2 (grouped by rank) was timed later (b2_grouped.json) on exactly the same batches. Only B2's columns come from
# that run; every other number is unchanged, and speedups over B2 use this table's own RF1 / RF2 numbers.
B2 = load("b2_grouped.json")
B2_NOTE = (" B2 = group tokens by rank, one PyTorch matmul pair per distinct rank (b2_grouped's body, groups built "
           "once per step outside the graph), timed later on the same batches with the same method "
           "(b2_grouped.json); speedups over B2 use this table's own RankFuse numbers.")

# 1. RF1 vs B1, one layer -----------------------------------------------------------------------
d = load("rf1_final_sweep.json")
rows, drift = [], []
for dist, by_t in d["write"].items():
    for t, v in by_t.items():
        b = B2["one_layer_write"][dist][t]
        drift.append(abs(b["rf1"] / v["rf1_fused"] - 1))
        rows.append([int(t), dist, round(v["b1"], 2), round(b["b2"], 2), round(v["rf1_single"], 2),
                     round(v["rf1_fused"], 2), x(v["b1"], v["rf1_fused"]), x(b["b2"], v["rf1_fused"]),
                     x(v["rf1_single"], v["rf1_fused"])])
rows.sort(key=lambda r: (r[1] != "uniform", r[0]))
DRIFT = (f" Check: RF1 re-timed alongside B2 differs from this table's RF1 by a median of "
         f"{100 * statistics.median(drift):.1f}% (max {100 * max(drift):.1f}%).")
table("01_rf1_vs_b1_one_layer", "RF1 vs B1 and B2, one layer, output written (us per call)",
      TIMING_GRAPH + " Decode-shaped batches (T requests, 1 token each); uniform and skew_low rank mixes. "
      "B1 = PyTorch pad-to-batch-max + mask (b1_padded_graphed's body). RF1 one kernel = rf1_single (first "
      "design); RF1 split-K = rf1_fused (final). Source: rf1_final_sweep.json." + B2_NOTE + DRIFT,
      ["tokens", "rank mix", "B1 us", "B2 us", "RF1 one-kernel us", "RF1 split-K us", "speedup vs B1",
       "RF1 speedup vs B2", "speedup vs one-kernel"], rows)

rows = []
for t, v in d["add"].items():
    b = B2["one_layer_add"][t]
    rows.append([int(t), round(v["b1_plus_add"], 2), round(b["b2_add"], 2), round(v["rf1_fused_add"], 2),
                 x(v["b1_plus_add"], v["rf1_fused_add"]), x(b["b2_add"], v["rf1_fused_add"])])
table("02_rf1_vs_b1_add_mode", "RF1 vs B1 and B2, delta added into the base output (us per call)",
      TIMING_GRAPH + " out += delta, as inside a real layer: RF1 adds inside its expand kernel; B1 needs a "
      "separate add; B2 adds each group with index_add_. Uniform ranks. Source: rf1_final_sweep.json." + B2_NOTE,
      ["tokens", "B1 + add us", "B2 (index_add) us", "RF1 split-K (add) us", "speedup",
       "RF1 speedup vs B2"], rows)

# 3. vs B3 (vLLM's kernels) ---------------------------------------------------------------------
def b3_label(w):
    kind, a, b, dist = (w.split(":") + [""])[:4]
    if kind == "mixed":
        return "prompts 15 x 512 at rank 4 + 1 x 512 at rank 64", 16 * 512
    if kind == "decode":
        return f"decode {a} x 1, {dist}", int(a)
    return f"prompts {a} x {b}, {dist}", int(a) * int(b)


rows = []
b2_by_w = {r["workload"]: r for r in B2["vs_b3_add"]}
for r in load("b3_vllm.json"):
    label, tokens = b3_label(r["workload"])
    b = b2_by_w[r["workload"]]
    rows.append([label, tokens, r["active_copies"], r["max_rank"], round(r["mean_rank"], 1), round(r["b1"], 2),
                 round(b["b2_add"], 2), round(r["b3"], 2), round(r["rf1"], 2), round(r["rf2seg"], 2),
                 x(r["b3"], r["rf1"]), x(r["b1"], r["rf1"]), x(b["b2_add"], r["rf1"])])
table("03_rf1_vs_b3_vllm_kernels", "RF1 vs B3 (vLLM 0.32's own LoRA kernels), delta added into the output (us per call)",
      TIMING_GRAPH + " B3 = how stock vLLM serves a Matryoshka adapter: every rank tier is a separate adapter "
      "padded to rank 64 (7 copies), computed with vLLM's _lora_shrink/_lora_expand; 'copies used' = distinct "
      "ranks in the batch. all_max = every token at rank 64 (B3 then reads one copy). Source: b3_vllm.json."
      + B2_NOTE,
      ["workload", "tokens", "B3 copies used", "max rank", "mean rank", "B1 us", "B2 us", "B3 us", "RF1 us",
       "RF2 segment us", "RF1 speedup vs B3", "RF1 speedup vs B1", "RF1 speedup vs B2"], rows)


# 4. Memory traffic (ncu) -----------------------------------------------------------------------
SCALE = {"byte": 1e-6, "Kbyte": 1e-3, "Mbyte": 1.0, "Gbyte": 1e3, "ns": 1e-3, "us": 1.0, "usecond": 1.0,
         "ms": 1e3, "msecond": 1e3, "nsecond": 1e-3, "sector": 1.0, "Ksector": 1e3, "Msector": 1e6}


def ncu_rows(path):
    lines = [l for l in path.read_text().splitlines() if not l.startswith("==")]
    rows = list(csv.reader(lines))
    header, units, data = rows[0], rows[1], rows[2:]
    out = []
    for r in data:
        rec = {"kernel": r[header.index("Kernel Name")]}
        for i, h in enumerate(header):
            if "__" in h:
                try:
                    rec[h] = float(r[i].replace(",", "")) * SCALE.get(units[i], 1.0)
                except ValueError:
                    pass
        out.append(rec)
    return out


def ncu_summary(path):
    k = ncu_rows(path)
    time_us = sum(r.get("gpu__time_duration.sum", 0) for r in k)
    read = sum(r.get("dram__bytes_read.sum", 0) for r in k)
    write = sum(r.get("dram__bytes_write.sum", 0) for r in k)
    l2 = sum(r.get("lts__t_sectors.sum", 0) for r in k) * 32 / 1e6
    bw = 100 * (read + write) * 1e6 / (time_us * 1e-6) / 4.8e12 if time_us else None  # MB, us -> % of 4.8 TB/s
    return len(k), time_us, read, write, l2, bw


NAMES = {"b1": "B1 (PyTorch)", "b2": "B2 (PyTorch, grouped)", "b3": "B3 (vLLM kernels)", "rf1": "RF1 split-K",
         "rf2seg": "RF2 segment"}
rows = []
for t in (32, 2048):
    for impl in ("b1", "b2", "b3", "rf1", "rf2seg"):
        n, tm, rd, wr, l2, bw = ncu_summary(SRC / "ncu" / f"b3cmp-{impl}-t{t}.csv")
        rows.append([t, NAMES[impl], n, round(tm, 1), round(rd, 2), round(wr, 2), round(l2, 1), round(bw, 1)])
table("04_memory_traffic_ncu", "Memory traffic per call, Nsight Compute (add-into-output mode)",
      "ncu 2026.3, one H200, 4096 x 4096, r_max 64, uniform ranks (T=32 uses all 7 ranks), fp16. Only the last "
      "of three calls profiled; sums over that call's kernels. ncu flushes caches and locks clocks to base for "
      "every kernel, so times are longer than the graph-replay numbers. 'DRAM % peak' = (read + write) / time "
      "against 4.8 TB/s; writes that stay in L2 until the kernel ends are not counted as DRAM writes. B3's extra "
      "reads are its 7 padded adapter copies; B2 launches several small kernels per rank group. "
      "Source: ncu/b3cmp-*.csv (B2 profiled later, same settings and batch).",
      ["tokens", "implementation", "kernels", "kernel time us (ncu)", "DRAM read MB", "DRAM write MB",
       "L2 traffic MB", "DRAM % peak"], rows)

rows = []
for t in (2048, 8192):
    for impl, label in (("b1_graphed", "B1 (PyTorch, graphed)"), ("rf1_tuned", "RF1 one kernel, tuned"),
                        ("splitk", "RF1 split-K (prototype, before tiled expand)")):
        n, tm, rd, wr, l2, bw = ncu_summary(SRC / "ncu" / f"{impl}-t{t}-prefill.csv")
        rows.append([t, label, n, round(tm, 1), round(rd, 2), round(wr, 2), round(l2, 1), round(bw, 1)])
table("05_long_prompt_profile_ncu", "Long-prompt profile that led to the tiled expand (ncu, output written)",
      "Development-stage profile (23 Oct): at 2048-8192 tokens the split-K shrink matched cuBLAS but the expand "
      "was ~4x slower, which motivated the one-tile-per-program expand now in rf1_fused (table 06). Same ncu "
      "settings as table 04, uniform ranks. Source: ncu/*-prefill.csv.",
      ["tokens", "implementation", "kernels", "kernel time us (ncu)", "DRAM read MB", "DRAM write MB",
       "L2 traffic MB", "DRAM % peak"], rows)

# 6. Expand fix ---------------------------------------------------------------------------------
e = load("expand_prefill.json")
final = d["write"]["uniform"]
rows = []
for t in ("32", "128", "512", "1024", "2048", "4096", "8192"):
    after = final.get(t, {}).get("rf1_fused")
    rows.append([int(t), round(e["b1"][t], 2), round(e["repo"][t], 2), round(after, 2) if after else None,
                 x(e["b1"][t], e["repo"][t]), x(final[t]["b1"], after) if after else None])
table("06_rf1_long_prompt_expand_fix", "RF1 before and after the tiled expand (us per call, output written)",
      TIMING_GRAPH + " 'before' = rf1_fused on 24 Oct with the strip expand (expand_prefill.json); 'after' = final "
      "rf1_fused with the one-tile-per-program expand and vectorised stores (rf1_final_sweep.json, uniform). "
      "Speedups are against B1 measured in the same run.",
      ["tokens", "B1 us", "RF1 before us", "RF1 after us", "before: speedup vs B1", "after: speedup vs B1"], rows)


# 7-9. RF2 variants -----------------------------------------------------------------------------
rows = []
b2_by_s = {(r["requests"], r["tokens_per_request"], r["dist"]): r for r in B2["rf2_sorted"]}
for r in load("rf2_splitk_v2.json"):
    b = b2_by_s[(r["requests"], r["tokens_per_request"], r["dist"])]
    rows.append([r["requests"], r["tokens_per_request"], r["requests"] * r["tokens_per_request"], r["dist"],
                 r["max_rank"], round(r["mean_rank"], 1), round(r["b1"], 2), round(b["b2"], 2),
                 round(r["rf1_splitk"], 2), round(r["rf2_repo"], 2), round(r["rf2_splitk"], 2), round(r["sort"], 2),
                 x(r["rf1_splitk"], r["rf2_splitk"]), x(b["b2"], r["rf2_splitk"])])
table("07_rf2_sorted_splitk", "RF2 sorted (split-K and one-kernel) vs RF1 (us per call, output written)",
      TIMING_GRAPH + " RF2 sorts tokens by rank so each token block runs at its own rank bucket. 'sort us' = the "
      "argsort, timed apart (a server would build it once per step). >1 in the last column = RF2 faster. "
      "skew_high mixes have max rank 32. Source: rf2_splitk_v2.json." + B2_NOTE,
      ["requests", "tokens/request", "tokens", "rank mix", "max rank", "mean rank", "B1 us", "B2 us",
       "RF1 split-K us", "RF2 one-kernel (rf2_fused) us", "RF2 sorted split-K us", "sort us", "RF1 / RF2 split-K",
       "RF2 split-K speedup vs B2"], rows)

rows = []
b2_by_b = {r["batch"]: r for r in B2["rf2_segment_batches"]}
for r in load("rf2_segment.json"):
    b = b2_by_b[r["batch"]]
    rows.append([r["batch"], r["max_rank"], round(r["mean_rank"], 1), round(r["b1"], 2), round(b["b2"], 2),
                 round(r["rf1_splitk"], 2), round(r["rf2_sorted"], 2), round(r["rf2_contig"], 2),
                 round(r["rf2_aligned"], 2), x(r["rf1_splitk"], r["rf2_contig"]), x(b["b2"], r["rf2_contig"])])
table("08_rf2_segment", "RF2 segment (consecutive-token blocks, no sort) vs RF1 (us per call, output written)",
      TIMING_GRAPH + " 'A x L @rR' = A requests of L tokens at rank R; 'skew_low'/'uniform' = one rank per request "
      "drawn from that mix. contig = fixed blocks (kept); aligned = blocks never cross a request (slow at decode). "
      ">1 in 'RF1 / RF2 segment' = segment RF2 faster. Source: rf2_segment.json." + B2_NOTE,
      ["batch", "max rank", "mean rank", "B1 us", "B2 us", "RF1 split-K us", "RF2 sorted us", "RF2 segment (contig) us",
       "RF2 segment (aligned) us", "RF1 / RF2 segment", "RF2 segment speedup vs B2"], rows)


def batch_shape(label):
    """(tokens, tokens per request) from a batch label of rf2_segment / rf2_bucketed."""
    if label.startswith("mix"):
        n = int(label.split(" x ")[1])
        return 7 * n, n
    parts = re.findall(r"(\d+) x (\d+)", label)
    return sum(int(a) * int(b) for a, b in parts), int(parts[0][1])


rows = []
for r in load("rf2_bucketed.json"):
    tokens, tpr = batch_shape(r["batch"])
    pick = tokens > 128 and r["work_ratio"] <= 0.45 and (tokens >= 4096 or tpr >= 64)
    chosen = r["rf2_bucketed"] if pick else r["rf1"]
    b = b2_by_b[r["batch"]]
    rows.append([r["batch"], tokens, r["max_rank"], round(r["mean_rank"], 1), round(r["work_ratio"], 2),
                 round(b["b2"], 2), round(r["rf1"], 2), round(r["rf2_segment"], 2), round(r["rf2_bucketed"], 2),
                 round(r["rf2_bucketed_1stream"], 2), x(r["rf1"], r["rf2_bucketed"]),
                 "RF2 bucketed" if pick else "RF1", x(r["rf1"], chosen), x(b["b2"], r["rf2_bucketed"])])
table("09_rf2_bucketed", "RF2 bucketed (one kernel pair per rank bucket, concurrent streams) vs RF1 (us per call)",
      TIMING_GRAPH + " work ratio = bucketed rank work / RF1's (lower = more to skip). '1 stream' = buckets run one "
      "after another. 'use_rf2 picks' applies the rule now in the repo (graphs on, > 128 tokens, work ratio <= "
      "0.45, and >= 4096 tokens or >= 64 tokens per request); the rule was fitted on this same data. "
      "Source: rf2_bucketed.json." + B2_NOTE,
      ["batch", "tokens", "max rank", "mean rank", "work ratio", "B2 us", "RF1 us", "RF2 segment us", "RF2 bucketed us",
       "RF2 bucketed, 1 stream us", "RF1 / RF2 bucketed", "use_rf2 picks", "RF1 / picked",
       "RF2 bucketed speedup vs B2"], rows)


# 10-11. Whole model ----------------------------------------------------------------------------
MODEL_SRC = "model_forward_repo.json"
MODEL_B2 = {(r["kind"], r["batch"], r["tokens"], r["mix"]): r for r in load("model_forward_b2.json")}
MODEL_SK = {(r["kind"], r["batch"], r["tokens"], r["mix"]): r for r in load("model_forward_rf2sk.json")}
BACK = ["none", "b1", "b2", "b3", "rf1", "rf2seg", "rf2bkt", "rf2auto"]
for key, stem, title, how in (
        ("graph_ms", "10_whole_model_graph", "Whole Qwen2.5-0.5B forward, CUDA graphs (ms per forward)",
         "GPU time only: the whole forward captured in a CUDA graph and replayed (median of 7 x 20)."),
        ("eager_ms", "11_whole_model_eager", "Whole Qwen2.5-0.5B forward, eager (ms per forward)",
         "Wall time of a normal forward plus a sync (median of 10): includes Python and launch overhead.")):
    rows = []
    for r in load(MODEL_SRC):
        k4 = (r["kind"], r["batch"], r["tokens"], r["mix"])
        t = dict(r[key], b2=MODEL_B2[k4][key]["b2"], rf2sk=MODEL_SK[k4][key]["rf2sk"])
        if not all(b in t for b in BACK):
            continue
        share = lambda b: round(100 * (t[b] - t["none"]) / t[b], 1)
        rows.append([r["kind"], f"{r['batch']} x {r['tokens']}", r["mix"]] + [round(t[b], 3) for b in BACK]
                    + [share("b1"), share("b2"), share("b3"), share("rf1"), x(t["b1"], t["rf1"]),
                       x(t["b2"], t["rf1"]), x(t["b3"], t["rf1"]), x(t["rf1"], t["rf2bkt"]),
                       round(t["rf2sk"], 3), share("rf2sk"), x(t["rf1"], t["rf2sk"])])
    table(stem, title,
          how + " Qwen2.5-0.5B-Instruct (24 layers, hidden 896) in PyTorch, fp16, all 168 projections (q,k,v,o,"
          "gate,up,down) carrying one shared rank-64 Matryoshka adapter; each request has one rank. Decode = one "
          "new token per request on a 128-token KV cache; prefill = whole prompts. Mixes: all64, uniform over the "
          "7 tiers, mostlylow (skew_low), mostlylow+64 (one request at 64). 'LoRA %' = (time - none) / time. "
          f"Every backend matches an fp32 reference (table 13). Source: {MODEL_SRC}; the B2 column (b2_grouped's "
          "body, groups built once per forward, added with index_add_) was timed later with the same method "
          "(model_forward_b2.json); the RF2 sorted split-K columns (rf2_splitk, stable rank order built once per "
          "forward) likewise (model_forward_rf2sk.json).",
          ["phase", "requests x tokens", "rank mix", "no LoRA ms", "B1 ms", "B2 ms", "B3 ms", "RF1 ms",
           "RF2 segment ms", "RF2 bucketed ms", "RF2 auto ms", "LoRA % B1", "LoRA % B2", "LoRA % B3", "LoRA % RF1",
           "RF1 speedup vs B1", "RF1 speedup vs B2", "RF1 speedup vs B3", "RF1 / RF2 bucketed",
           "RF2 sorted split-K ms", "LoRA % RF2 sorted split-K", "RF1 / RF2 sorted split-K"], rows)


# 12. LoRA share of a vLLM step -----------------------------------------------------------------
p = [json.loads(l) for l in (SRC / "p22_lora_share.jsonl").read_text().splitlines()]
by = {(r["mode"], r["graphs"]): r for r in p}
rows = []
for graphs in ("graphs", "eager"):
    base = by[("base", graphs)]
    for phase, key, sizes in (("decode step", "decode_ms", ("1", "8", "32", "128")),
                              ("prefill", "prefill_ms", ("128", "512", "2048", "8192"))):
        for s in sizes:
            b = base[key][s]
            row = [graphs, phase, int(s), round(b, 3)]
            for m in ("lora16", "lora64", "lora64-unused"):
                v = by[(m, graphs)][key][s]
                row += [round(v, 3), round(100 * (v - b) / v, 1)]
            rows.append(row)
table("12_vllm_lora_share", "How much of a stock vLLM step is LoRA (Qwen2.5-0.5B, vLLM 0.32)",
      "Stock vLLM 0.32.0 on one H200, prefix caching off, one engine per run. Decode step time at batch B "
      "(prompts of 128 tokens; (t_144 - t_16) / 128 removes the prefill); prefill = one request of L tokens. "
      "lora16 / lora64 = engine with max_lora_rank 16 / 64 serving one adapter of that rank; 'unused' = LoRA "
      "enabled but the request carries no adapter. 'LoRA %' = (time - base) / time. Source: p22_lora_share.jsonl.",
      ["CUDA graphs", "phase", "batch or prompt tokens", "base ms", "rank 16 ms", "LoRA % r16", "rank 64 ms",
       "LoRA % r64", "LoRA enabled, unused ms", "overhead % unused"], rows)


# 14. Repo-code validation sweep ----------------------------------------------------------------
def sweep(name):
    out = {}
    for r in csv.DictReader(open(SRC / name)):
        out[(r["impl"], int(r["num_tokens"]))] = (float(r["median_us"]), r["timing"])
    return out


g, ea = sweep("repo_sweep_graph.csv"), sweep("repo_sweep_eager.csv")
for name, dst in (("repo_sweep_graph_b2.csv", g), ("repo_sweep_eager_b2.csv", ea)):
    dst.update({k: v for k, v in sweep(name).items() if k[0] == "b2_grouped"})
impls = ["b1_padded", "b1_padded_graphed", "b2_grouped", "b3_vllm", "rf1_fused", "rf2_fused", "rf2_splitk",
         "rf2_segment", "rf2_bucketed"]
rows = []
for t in sorted({k[1] for k in g}):
    for i in impls:
        if (i, t) in g:
            rows.append([t, i, round(g[(i, t)][0], 1), g[(i, t)][1], round(ea[(i, t)][0], 1)])
table("14_repo_sweep_all_implementations", "Every registered implementation, repo code (us per call)",
      "bench/run_sweep.py on the committed repo (25 Oct), one H200, 4096 x 4096, r_max 64, uniform ranks, fp16, "
      "L2 flushed before each call. 'graph' column = --timing graph (GPU time; b1_padded_graphed replays its own "
      "graph so it is timed eagerly); 'eager' column = --timing eager (includes ~70-110 us of Python launch "
      "overhead per call, B3 and RF2 bucketed also build their per-step metadata per call). Sources: "
      "repo_sweep_graph.csv, repo_sweep_eager.csv; b2_grouped rows from a later sweep with the same config "
      "(repo_sweep_*_b2.csv), with b2_grouped given its per-step groups (prepare) so it can be graph-captured.",
      ["tokens", "implementation", "graph us", "graph timing used", "eager us"], rows)


# ALL_TABLES.md ---------------------------------------------------------------------------------
doc = ["# RankFuse: all result tables", "",
       "Generated by `build_tables.py` from `source/`. CSV copies are in `tables/`. Table 13 (correctness) is "
       "in README.md.", ""]
for stem, title, notes, header, rows in TABLES:
    doc.append(markdown(f"{stem.split('_')[0]}. {title}", notes, header, rows))
(HERE / "ALL_TABLES.md").write_text("\n".join(doc))
print(f"wrote {len(TABLES)} tables to {OUT} and ALL_TABLES.md (whole model from {MODEL_SRC})")
