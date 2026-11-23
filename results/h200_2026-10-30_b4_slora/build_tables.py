"""Build the B4 (S-LoRA bgmv) tables from the run_sweep CSVs and the ncu CSVs in source/.

    python results/h200_2026-10-30_b4_slora/build_tables.py

Writes source/*.json (raw rows, one JSON list per sweep) and tables/*.md + *.csv.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC, OUT = HERE / "source", HERE / "tables"
SHAPE_ORDER = ["layer_4096", "q_o", "k_v", "gate_up", "down"]
SHAPE_LABEL = {"layer_4096": "4096x4096", "q_o": "q/o 896x896", "k_v": "k/v 896x128",
               "gate_up": "gate/up 896x4864", "down": "down 4864x896"}
DIST_ORDER = ["uniform", "skew_low", "all_max"]
COLS = [("b1_padded_graphed", "B1 us"), ("b3_vllm", "B3 us"), ("rf1_fused", "RF1 us"),
        ("b4_slora_copies", "B4 copies us"), ("b4_slora_shared", "B4 shared us")]


def fmt(v: float) -> str:
    return f"{v:.4g}" if v >= 100 else f"{v:.2f}"


def load(name: str) -> list[dict]:
    rows = []
    for p in sorted(SRC.glob(f"{name}*.csv")):
        with p.open() as f:
            rows += list(csv.DictReader(f))
    (SRC / f"{name}.json").write_text(json.dumps(rows, indent=1))
    return rows


def write(stem: str, title: str, header: str, cols: list[str], rows: list[list]) -> None:
    md = [f"### {title}", "", header, "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    (OUT / f"{stem}.md").write_text("\n".join(md) + "\n")
    with (OUT / f"{stem}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)


def timing_table(rows: list[dict], kind: str) -> tuple[list[str], list[list]]:
    by = {}
    for r in rows:
        key = (r["shape"], int(r["num_requests"]), int(r["tokens_per_request"]), r["dist"])
        by.setdefault(key, {})[r["impl"]] = r
    cols = ["shape", "workload", "tokens", "max rank", "mean rank"] + [c for _, c in COLS] + [
        "RF1 speedup vs B4 copies", "RF1 speedup vs B4 shared", "RF1 speedup vs B3", "RF1 speedup vs B1"]
    out = []
    for key in sorted(by, key=lambda k: (SHAPE_ORDER.index(k[0]), k[1] * k[2], DIST_ORDER.index(k[3]))):
        shape, n, tpr, dist = key
        d = by[key]
        any_row = next(iter(d.values()))
        us = {impl: float(d[impl]["median_us"]) for impl, _ in COLS if impl in d}
        rf1 = us.get("rf1_fused")
        wl = f"{kind} {n} x {tpr}, {dist}"
        line = [SHAPE_LABEL[shape], wl, n * tpr, any_row["batch_max_rank"], f"{float(any_row['mean_rank']):.2f}"]
        line += [fmt(us[i]) if i in us else "" for i, _ in COLS]
        for other in ("b4_slora_copies", "b4_slora_shared", "b3_vllm", "b1_padded_graphed"):
            line.append(f"{us[other] / rf1:.2f}" if rf1 and other in us else "")
        out.append(line)
    return cols, out


def geomean_table(rows: list[dict]) -> tuple[list[str], list[list]]:
    """Geometric mean of RF1's speedup over each baseline, per shape and over all shapes."""
    import math

    by = {}
    for r in rows:
        key = (r["shape"], r["num_requests"], r["tokens_per_request"], r["dist"])
        by.setdefault(key, {})[r["impl"]] = float(r["median_us"])
    others = [("b4_slora_copies", "B4 copies"), ("b4_slora_shared", "B4 shared"), ("b3_vllm", "B3"),
              ("b1_padded_graphed", "B1")]
    cols = ["shape", "workloads"] + [f"RF1 vs {lab} (geomean, min-max)" for _, lab in others]
    out = []
    for shape in SHAPE_ORDER + ["all"]:
        ks = [k for k in by if shape in ("all", k[0]) and "rf1_fused" in by[k]]
        if not ks:
            continue
        line = [SHAPE_LABEL.get(shape, "all shapes"), len(ks)]
        for impl, _ in others:
            s = [by[k][impl] / by[k]["rf1_fused"] for k in ks if impl in by[k]]
            line.append(f"{math.exp(statistics.fmean(map(math.log, s))):.2f} ({min(s):.2f}-{max(s):.2f})" if s else "")
        out.append(line)
    return cols, out


def ncu_table() -> tuple[list[str], list[list]] | None:
    files = sorted((SRC / "ncu").glob("*.csv"))
    if not files:
        return None
    cols = ["impl", "kernels", "kernel time us (sum)", "DRAM read MB", "DRAM write MB", "DRAM total MB",
            "L2 traffic MB", "kernel names"]
    out, raw = [], {}
    for p in files:
        with p.open() as f:
            lines = list(csv.reader(f))
        start = next(i for i, l in enumerate(lines) if l and l[0] == "ID")  # skip ==PROF== lines
        head, units, data = lines[start], lines[start + 1], lines[start + 2:]
        recs = [dict(zip(head, d)) for d in data]
        raw[p.stem] = recs
        num = lambda r, k: float(r[k].replace(",", "")) if r.get(k) else 0.0
        # ncu reports gpu__time_duration in the unit on the units line (us or ns).
        tu = units[head.index("gpu__time_duration.sum")]
        tscale = {"ns": 1e-3, "us": 1.0, "usecond": 1.0, "nsecond": 1e-3, "ms": 1e3, "msecond": 1e3}.get(tu, 1.0)
        bunit = {"byte": 1e-6, "Kbyte": 1e-3, "Mbyte": 1.0, "Gbyte": 1e3}
        rd = sum(num(r, "dram__bytes_read.sum") * bunit.get(units[head.index("dram__bytes_read.sum")], 1e-6) for r in recs)
        wr = sum(num(r, "dram__bytes_write.sum") * bunit.get(units[head.index("dram__bytes_write.sum")], 1e-6) for r in recs)
        t = sum(num(r, "gpu__time_duration.sum") * tscale for r in recs)
        l2 = sum(num(r, "lts__t_sectors.sum") for r in recs) * 32 / 1e6  # 32-byte sectors
        names = "; ".join(r.get("Kernel Name", "")[:40] for r in recs)
        out.append([p.stem.split("-r")[0], len(recs), f"{t:.2f}", f"{rd:.2f}", f"{wr:.2f}", f"{rd + wr:.2f}",
                    f"{l2:.2f}", names])
    (SRC / "ncu.json").write_text(json.dumps(raw, indent=1))
    return cols, out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    common = ("GPU time only: each call is captured in a CUDA graph with the per-step metadata its `prepare` "
              "builds (pools, token -> entry map, zeroed buffers) and replayed (bench/run_sweep.py --timing graph; "
              "median of {reps} repetitions of {replays} replays; L2 flushed before each call, flush time subtracted), "
              "except B1 (b1_padded_graphed), which replays its own graph and is timed with CUDA events around the call. "
              "One NVIDIA H200, fp16, r_max 64 (tiers 1, 2, 4, ..., 64), each call returns the LoRA delta [T, h_out]. "
              "Shapes: the 4096 x 4096 test layer and Qwen2.5-0.5B's projections. "
              "B4 copies = S-LoRA's bgmv kernel serving each rank tier as its own adapter at its true rank "
              "(ranks 1/2/4 stored padded to 8: 144 pool rows per matrix); B4 shared = the same kernel with A_t and "
              "B^T stored once (64 rows), shrink at each token's true rank, expand over all 64 columns. "
              "B3 = vLLM 0.32's shrink/expand with 7 copies padded to 64; B1 = PyTorch padded to the batch max rank. "
              "Speedup = baseline us / RF1 us.")
    dec = load("decode")
    if dec:
        cols, rows = timing_table(dec, "decode")
        write("01_b4_decode", "RF1 vs B4 (S-LoRA bgmv), decode, one token per request (us per call)",
              common.format(reps=200, replays=20), cols, rows)
        cols, rows = geomean_table(dec)
        write("03_b4_decode_summary", "RF1 speedup over each baseline at decode, geometric mean per shape",
              "Summary of 01_b4_decode (T = 1, 8, 32, 128, 256; uniform, skew_low, all_max).", cols, rows)
    pre = load("prefill")
    if pre:
        cols, rows = timing_table(pre, "prompts")
        write("02_b4_prefill_LABELLED", "RF1 vs B4 on prompt-sized batches (us per call) -- NOT how S-LoRA serves prompts",
              "**S-LoRA runs bgmv for decode only; for prompts (>= 200 tokens) it uses a separate Triton kernel. "
              "These rows run the bgmv kernel on 2048- and 8192-token batches (4 or 16 requests x 512 tokens, one "
              "rank per request) to show how the kernel itself scales, not S-LoRA's prefill time.** "
              + common.format(reps=50, replays=10), cols, rows)
    nc = ncu_table()
    if nc:
        write("04_b4_ncu_dram_bytes", "HBM traffic per call at decode T = 32, uniform ranks (Nsight Compute)",
              "scripts/ice/ncu_profile.sh <impl> 32 1 4096 4096 64 uniform fp16: one profiled call (per-step metadata "
              "built before it), ncu defaults (caches flushed, clocks locked to base, so kernel times are longer than "
              "in the timing tables). Sums over every kernel of the call, including buffer zeroing (memset kernels). "
              "One NVIDIA H200, fp16, 4096 x 4096, r_max 64.", *nc)


if __name__ == "__main__":
    main()
