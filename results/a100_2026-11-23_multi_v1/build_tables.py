"""Rebuild tables/ and ALL_TABLES.md from raw/*.jsonl and source/ncu/*.csv (plain Python, no pandas).

    python3 build_tables.py

Built the way results/h200_2026-11-03_multi/build_tables.py builds them, with r_max (64, 128,
256) and seed (0, 1, 2) added: speedups are time(baseline) / time(RF1-MA) on the same batch (same
r_max, seed, shape, adapters, popularity, rank mix and tokens); summaries over several batches are
geometric means with the min-max range. Fractions of bandwidth use the A100 80GB PCIe spec sheet,
1.935 TB/s. Table 11 compares with the H200 run (results/h200_2026-11-03_multi, r_max 64, seed 0).
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW, TABLES, NCU = HERE / "raw", HERE / "tables", HERE / "source" / "ncu"
H200_RAW = HERE.parent / "h200_2026-11-03_multi" / "raw" / "multi-main.jsonl"
BW = 1.935e12  # A100 80GB PCIe, bytes/s (spec sheet)
SHAPES = ["4096x4096", "q_o", "k_v", "gate_up", "down", "q7_o", "kv7", "gate_up7", "down7"]
SHAPE_DIMS = {"4096x4096": "4096->4096", "q_o": "0.5B q/o 896->896", "k_v": "0.5B k/v 896->128",
              "gate_up": "0.5B gate/up 896->4864", "down": "0.5B down 4864->896", "q7_o": "7B q/o 3584->3584",
              "kv7": "7B k/v 3584->512", "gate_up7": "7B gate/up 3584->18944", "down7": "7B down 18944->3584"}
LABELS = ["decode-1", "decode-8", "decode-32", "decode-128", "decode-256", "prefill-2048", "prefill-8192"]
ADAPTERS = [1, 4, 16, 64, 256]
RMAX = [64, 128, 256]
RF2 = ["rf2_ma_splitk", "rf2_ma_segment", "rf2_ma_bucketed"]
BASES = ["b3_multi", "b4_multi_copies", "b4_multi_shared", "b1_multi"]
ALL_IMPLS = ["rf1_ma", *RF2, *BASES, "stage1_rf1_fused"]


def read_jsonl(paths) -> list[dict]:
    rows = []
    for p in paths:
        rows += [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]
    return rows


def ok(r: dict) -> bool:
    return not r.get("error") and r["median_us"] == r["median_us"]


def key(r: dict) -> tuple:
    return (r["r_max"], r["seed"], r["shape"], r["label"], r["num_adapters"], r["popularity"], r["dist"])


def gmean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def fmt_ratio(xs) -> str:
    xs = [x for x in xs if x == x]
    if not xs:
        return "-"
    g = gmean(xs)
    return f"{g:.2f}x" if len(xs) == 1 else f"{g:.2f}x ({min(xs):.2f}-{max(xs):.2f})"


def write(name: str, title: str, note: str, header: list[str], body: list[list]) -> str:
    TABLES.mkdir(exist_ok=True)
    with (TABLES / f"{name}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(body)
    md = [f"### {title}", "", note, "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    text = "\n".join(md) + "\n"
    (TABLES / f"{name}.md").write_text(text)
    return text


_PAIRS: dict = {}


def pair_ratios(idx, base: str, impl: str = "rf1_ma") -> list[tuple[tuple, float]]:
    """[(batch key, time(base) / time(impl))] over every batch both ran (cached)."""
    if (base, impl) not in _PAIRS:
        _PAIRS[(base, impl)] = [(tuple(k), idx[(base, *k)]["median_us"] / r["median_us"])
                                for (name, *k), r in idx.items() if name == impl and (base, *k) in idx]
    return _PAIRS[(base, impl)]


def grouped(idx, base: str, by, impl: str = "rf1_ma") -> dict:
    """{by(key): [ratios]}; by returns None to drop a batch. key = (r_max, seed, shape, label, N, pop, dist)."""
    out = defaultdict(list)
    for k, x in pair_ratios(idx, base, impl):
        g = by(k)
        if g is not None:
            out[g].append(x)
    return out


def ratios(idx, base: str, impl: str = "rf1_ma", where=lambda k: True) -> list[float]:
    """time(base) / time(impl) over every batch key matching `where`."""
    return [x for k, x in pair_ratios(idx, base, impl) if where(k)]


def main() -> None:
    parts = []
    rows = read_jsonl(sorted(RAW.glob("multi-a100-r*.jsonl")))
    good = [r for r in rows if ok(r)]
    idx = {(r["impl"], *key(r)): r for r in good}
    rmaxes = [R for R in RMAX if any(r["r_max"] == R for r in good)]
    cell = defaultdict(list)  # (impl, r_max, shape, label, N) -> rows
    for r in good:
        cell[(r["impl"], r["r_max"], r["shape"], r["label"], r["num_adapters"])].append(r)
    seeds = sorted({r["seed"] for r in good})

    # 00: coverage
    body = []
    for R in rmaxes:
        for impl in ALL_IMPLS:
            rs = [r for r in rows if r["r_max"] == R and r["impl"] == impl]
            if not rs:
                continue
            errs = [r for r in rs if not ok(r)]
            first = (errs[0]["error"] or "nan")[:120] if errs else ""
            shapes_missing = [s for s in SHAPES if not any(r["shape"] == s for r in rs)]
            body.append([R, impl, len(rs), len(errs), ", ".join(shapes_missing) or "-", first])
    parts.append(write("00_coverage", "00. Rows per implementation and r_max",
                       f"Timed rows in raw/ (seeds {seeds}); errors = rows whose call raised (time NaN, excluded from every "
                       "table); shapes not run = skipped as unsupported (B4: rank width must be below h_in and h_out) or capped.",
                       ["r_max", "impl", "rows", "errors", "shapes not run", "first error"], body))

    # 01: RF1-MA time
    body = []
    for R in rmaxes:
        for shape in SHAPES:
            for n in ADAPTERS:
                row = [R, SHAPE_DIMS[shape], n]
                for label in LABELS:
                    ts = [r["median_us"] for r in cell[("rf1_ma", R, shape, label, n)]]
                    row.append(f"{gmean(ts):.1f}" if ts else "-")
                body.append(row)
    parts.append(write("01_rf1_ma_time", "01. RF1-MA GPU time (us)",
                       "CUDA-graph replay, add into the output, L2 flushed; median of 7 reps x 20 replays per batch, then "
                       "geometric mean over 3 seeds x popularity (uniform, Zipf 1.0) x rank mix (uniform, skew_low, all_max). "
                       "Weights in a shuffled page pool.",
                       ["r_max", "shape", "adapters", *LABELS], body))

    # S1: headline speedups by r_max x adapters x tokens (all shapes)
    for tag, base, title in (("S1", "b3_multi", "B3-multi (vLLM 0.32 ops, padded pair copies)"),
                             ("S2", "b4_multi_copies", "B4-multi copies (S-LoRA bgmv)"),
                             ("S3", "b4_multi_shared", "B4-multi shared (S-LoRA bgmv)")):
        g = grouped(idx, base, lambda k: (k[0], k[4], k[3]))
        body = [[R, n, *[fmt_ratio(g.get((R, n, l), [])) for l in LABELS]] for R in rmaxes for n in ADAPTERS]
        note = (f"time({base}) / time(rf1_ma) on the same batch; geometric mean (min-max) over all shapes x 3 seeds x "
                "popularity x rank mix. > 1: RF1-MA is faster.")
        if base.startswith("b4"):
            note += (" Prefill columns: S-LoRA serves prompts with a different (Triton) kernel; these measure bgmv itself. "
                     "At r_max >= 128 the 896->128 projection is not included (bgmv cannot run it).")
        parts.append(write(f"{tag}_summary_vs_{base}", f"{tag}. RF1-MA speedup over {title}, all shapes", note,
                           ["r_max", "adapters", *LABELS], body))

    # S4: one line per r_max x phase x baseline
    body = []
    for R in rmaxes:
        for phase in ("decode", "prefill"):
            row = [R, phase]
            for base in BASES:
                xs = grouped(idx, base, lambda k: (k[0], k[3].split("-")[0])).get((R, phase), [])
                below = sum(1 for x in xs if x < 0.95)
                row.append(f"{fmt_ratio(xs)}; {below}/{len(xs)} < 0.95x" if xs else "-")
            body.append(row)
    parts.append(write("S4_headline", "S4. Headline: RF1-MA speedup per r_max and phase",
                       "Geometric mean (min-max) of time(baseline) / time(rf1_ma) over every batch (shapes x adapters x "
                       "tokens x popularity x rank mix x seeds); then the number of batches where RF1-MA is > 5% slower.",
                       ["r_max", "phase", *BASES], body))

    # 02-05: speedups vs baselines per shape
    for num, base, title in (("02", "b3_multi", "vLLM 0.32 ops, one padded copy per (adapter, rank) pair (B3-multi)"),
                             ("03", "b4_multi_copies", "S-LoRA bgmv, one adapter per (adapter, rank) pair (B4-multi copies)"),
                             ("04", "b4_multi_shared", "S-LoRA bgmv, one shared copy per adapter (B4-multi shared)"),
                             ("05", "b1_multi", "PyTorch, padded per adapter group (B1-multi)")):
        cells = grouped(idx, base, lambda k: (k[0], k[2], k[4], k[3]))
        body = [[R, SHAPE_DIMS[s], n, *[fmt_ratio(cells.get((R, s, n, l), [])) for l in LABELS]]
                for R in rmaxes for s in SHAPES for n in ADAPTERS if any((R, s, n, l) in cells for l in LABELS)]
        note = (f"time({base}) / time(rf1_ma) on the same batch; geometric mean (min-max) over 3 seeds x popularity x rank "
                "mix. > 1: RF1-MA is faster.")
        if base.startswith("b4"):
            note += " Prefill columns: S-LoRA serves prompts with a different (Triton) kernel; these measure bgmv itself."
        parts.append(write(f"{num}_rf1_ma_vs_{base}", f"{num}. RF1-MA speedup over {title}", note,
                           ["r_max", "shape", "adapters", *LABELS], body))

    # 06: RF2 variants vs RF1-MA
    body = []
    for R in rmaxes:
        for impl in RF2:
            for phase in ("decode", "prefill"):
                xs, best = [], None
                for k, sp in pair_ratios(idx, "rf1_ma", impl):
                    if k[0] != R or not k[3].startswith(phase):
                        continue
                    xs.append(sp)
                    if best is None or sp > best[0]:
                        best = (sp, f"{k[2]} {k[3]} N={k[4]} {k[5]} {k[6]} seed {k[1]}")
                wins = sum(1 for x in xs if x > 1.05)
                body.append([R, impl, phase, fmt_ratio(xs), f"{wins}/{len(xs)}", f"{best[0]:.2f}x ({best[1]})" if best else "-"])
    parts.append(write("06_rf2_vs_rf1_ma", "06. RF2-multi variants vs RF1-MA",
                       "time(rf1_ma) / time(variant) over all shapes; > 1: the variant is faster. wins = batches where it is "
                       "> 5% faster.", ["r_max", "variant", "phase", "speedup gmean (min-max)", "wins >5%", "best batch"], body))

    # 07: multi-adapter machinery vs Stage 1 at one adapter
    body = []
    for R in rmaxes:
        for phase in ("decode", "prefill"):
            xs = []
            for (name, *k), r in idx.items():
                if name == "rf1_ma" and k[0] == R and k[4] == 1 and k[3].startswith(phase):
                    s1 = idx.get(("stage1_rf1_fused", *k))
                    if s1:
                        xs.append(r["median_us"] / s1["median_us"])
            body.append([R, phase, fmt_ratio(xs), len(xs)])
    parts.append(write("07_rf1_ma_vs_stage1", "07. RF1-MA (shuffled page pool) vs Stage 1 rf1_fused at one adapter",
                       "time(rf1_ma) / time(stage1_rf1_fused) on the same batch (1.00x = no cost of the multi-adapter "
                       "machinery and page table).", ["r_max", "phase", "time ratio gmean (min-max)", "batches"], body))

    # 08: fraction of DRAM bandwidth
    def frac(impl, R, shape, label, n):
        rs = cell[(impl, R, shape, label, n)]
        return gmean([r["ideal_bytes"] / BW / (r["median_us"] * 1e-6) for r in rs]) if rs else float("nan")

    body = []
    for R in rmaxes:
        for shape in SHAPES:
            for label in LABELS:
                cells = [frac("rf1_ma", R, shape, label, n) for n in ADAPTERS]
                body.append([R, SHAPE_DIMS[shape], label, *[f"{100 * c:.0f}%" if c == c else "-" for c in cells]])
    parts.append(write("08_rf1_ma_fraction_of_bandwidth", "08. RF1-MA fraction of A100 DRAM bandwidth, by adapters",
                       "ideal bytes / (1.935 TB/s x measured time); ideal bytes = x read + out read and written + each active "
                       "adapter's rank rows up to its group's maximum rank, read once. Geometric mean over 3 seeds x "
                       "popularity x rank mix. Columns: number of adapters.",
                       ["r_max", "shape", "tokens", *[f"N={n}" for n in ADAPTERS]], body))
    body = []
    for R in rmaxes:
        for shape in SHAPES:
            for label in LABELS:
                cells = [frac(impl, R, shape, label, 16) for impl in ["rf1_ma", *RF2, *BASES]]
                body.append([R, SHAPE_DIMS[shape], label, *[f"{100 * c:.0f}%" if c == c else "-" for c in cells]])
    parts.append(write("08b_fraction_of_bandwidth_16_adapters", "08b. Fraction of A100 DRAM bandwidth, 16 adapters, every implementation",
                       "As table 08 (the same ideal bytes for every implementation, so this is also a time ratio).",
                       ["r_max", "shape", "tokens", "rf1_ma", *RF2, *BASES], body))

    # 09: few tokens per adapter (decode)
    body = []
    for R in rmaxes:
        for shape in ("4096x4096", "q7_o", "down7", "down"):
            for label in ("decode-32", "decode-128", "decode-256"):
                for n in (16, 64, 256):
                    rs = cell[("rf1_ma", R, shape, label, n)]
                    if not rs:
                        continue
                    tpa = sum(r["tokens_per_active_adapter"] for r in rs) / len(rs)
                    eff = gmean([r["ideal_bytes"] / BW / (r["median_us"] * 1e-6) for r in rs])
                    ratio = lambda b: fmt_ratio([idx[(b, *key(r))]["median_us"] / r["median_us"] for r in rs if (b, *key(r)) in idx])  # noqa: E731
                    body.append([R, SHAPE_DIMS[shape], label, n, f"{tpa:.1f}", f"{gmean([r['median_us'] for r in rs]):.1f}",
                                 f"{100 * eff:.0f}%", ratio("b3_multi"), ratio("b4_multi_copies"), ratio("b4_multi_shared")])
    parts.append(write("09_few_tokens_per_adapter", "09. Decode with few tokens per adapter",
                       "RF1-MA at decode with many adapters; tokens/adapter = mean tokens per active adapter; of bandwidth as "
                       "in table 08; speedups as in 02-04 (over popularity x rank mix x seeds).",
                       ["r_max", "shape", "tokens", "adapters", "tokens/adapter", "rf1_ma us", "of bandwidth", "vs B3-multi",
                        "vs B4 copies", "vs B4 shared"], body))

    # 10: ncu bytes
    body = []
    for p in sorted(NCU.glob("*.csv")):
        rs = list(csv.reader(p.open()))
        if len(rs) < 3:
            continue
        hdr, data = rs[0], rs[2:]  # row 1 holds units
        col = {h: i for i, h in enumerate(hdr)}
        unit = dict(zip(hdr, rs[1]))

        def total(m):
            scale = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "sector": 32, "nsecond": 1e-3, "ns": 1e-3,
                     "usecond": 1, "us": 1, "ms": 1e3, "msecond": 1e3}.get(unit.get(m, ""), 1)
            return sum(float(r[col[m]].replace(",", "")) for r in data if r[col[m]]) * scale

        impl, shape, n, req, tpr, rtag = p.stem.split("-")[:6]
        R = int(rtag.removeprefix("r"))
        T = int(req) * int(tpr)
        ideal = [r for k, r in idx.items() if k[0] == "rf1_ma" and k[1] == R and k[2] == 0 and k[3] == shape
                 and k[4] == f"decode-{T}" and k[5] == int(n) and k[6] == "uniform" and k[7] == "uniform"]
        body.append([R, impl, shape, n, T, len(data), f"{total('dram__bytes_read.sum') / 1e6:.2f}",
                     f"{ideal[0]['ideal_bytes'] / 1e6:.2f}" if ideal else "-",
                     f"{total('dram__bytes_write.sum') / 1e6:.2f}", f"{total('lts__t_sectors.sum') / 1e6:.2f}",
                     f"{total('gpu__time_duration.sum'):.1f}"])
    body.sort(key=lambda b: (b[0], b[4], ALL_IMPLS.index(b[1]) if b[1] in ALL_IMPLS else 99))
    parts.append(write("10_ncu_bytes", "10. Nsight Compute: DRAM and L2 traffic per call (A100)",
                       "One prepared call, 16 adapters, uniform popularity and ranks, seed 0 (caches flushed, clocks locked by "
                       "ncu; compare bytes, not time). Summed over the call's kernels. ideal MB = the sweep's ideal bytes for "
                       "the same batch. L2 = lts__t_sectors x 32 B.",
                       ["r_max", "impl", "shape", "adapters", "tokens", "kernels", "DRAM read MB", "ideal MB", "DRAM write MB",
                        "L2 MB", "ncu time us"], body))

    # 11: H200 vs A100 at r_max 64 (seed 0, the H200 shapes)
    if H200_RAW.exists() and 64 in rmaxes:
        h = {(r["impl"], r["shape"], r["label"], r["num_adapters"], r["popularity"], r["dist"]): r
             for r in read_jsonl([H200_RAW]) if ok(r)}
        a = {(k[0], k[3], k[4], k[5], k[6], k[7]): r for k, r in idx.items() if k[1] == 64 and k[2] == 0}
        body = []
        for phase in ("decode", "prefill"):
            for impl in ["rf1_ma", "b3_multi", "b4_multi_copies", "b4_multi_shared", "b1_multi"]:
                xs = [a[k]["median_us"] / r["median_us"] for k, r in h.items() if k[0] == impl and k[2].startswith(phase) and k in a]
                body.append([phase, impl, "time A100 / time H200", fmt_ratio(xs), len(xs)])
            for base in ["b3_multi", "b4_multi_copies", "b4_multi_shared"]:
                for gpu, d in (("H200", h), ("A100", a)):
                    xs = [d[(base, *k[1:])]["median_us"] / r["median_us"] for k, r in d.items()
                          if k[0] == "rf1_ma" and k[2].startswith(phase) and (base, *k[1:]) in d and k in h and k in a]
                    body.append([phase, f"rf1_ma vs {base}", f"speedup on {gpu}", fmt_ratio(xs), len(xs)])
        parts.append(write("11_h200_vs_a100_r64", "11. H200 vs A100 80GB PCIe at r_max 64",
                           "The same batches (5 shapes of the H200 run, seed 0, popularity x rank mix x adapters x tokens) on "
                           "both GPUs. H200: results/h200_2026-11-03_multi/raw/multi-main.jsonl (same timing method). Spec "
                           "bandwidth ratio 4.8 / 1.935 = 2.48x.",
                           ["phase", "what", "ratio", "gmean (min-max)", "batches"], body))
        body = []
        for label in LABELS:
            row = [label]
            for base in ["b3_multi", "b4_multi_copies"]:
                for gpu, d in (("H200", h), ("A100", a)):
                    xs = [d[(base, *k[1:])]["median_us"] / r["median_us"] for k, r in d.items()
                          if k[0] == "rf1_ma" and k[2] == label and (base, *k[1:]) in d and k in h and k in a]
                    row.append(fmt_ratio(xs))
            xs = [a[k]["median_us"] / r["median_us"] for k, r in h.items() if k[0] == "rf1_ma" and k[2] == label and k in a]
            row.append(fmt_ratio(xs))
            body.append(row)
        parts.append(write("11b_h200_vs_a100_r64_by_tokens", "11b. H200 vs A100 at r_max 64, by token count",
                           "As table 11, per token count (all 5 shapes x adapters x popularity x rank mix, seed 0).",
                           ["tokens", "vs B3 H200", "vs B3 A100", "vs B4 copies H200", "vs B4 copies A100",
                            "rf1_ma time A100 / H200"], body))

    # 12: seed spread
    body = []
    for R in rmaxes:
        for base in ("b3_multi", "b4_multi_copies"):
            g = grouped(idx, base, lambda k: (k[0], k[1]))
            per_seed = [gmean(g.get((R, s), [])) for s in seeds]
            rel = []
            for (name, *k), r in idx.items():
                if name != "rf1_ma" or k[0] != R or k[1] != 0:
                    continue
                ts = [idx[("rf1_ma", R, s, *k[2:])]["median_us"] for s in seeds if ("rf1_ma", R, s, *k[2:]) in idx]
                if len(ts) == len(seeds):
                    rel.append(max(ts) / min(ts))
            body.append([R, base, " / ".join(f"{x:.3f}x" for x in per_seed), f"{gmean(rel):.3f}x (max {max(rel):.2f}x)" if rel else "-"])
    parts.append(write("12_seed_spread", "12. Seed-to-seed spread",
                       "Per seed: geometric-mean speedup of RF1-MA over every batch of that seed. Last column: max / min "
                       "RF1-MA time over the 3 seeds of the same batch (seeds change weights, adapter draws and ranks).",
                       ["r_max", "baseline", "speedup per seed (0 / 1 / 2)", "rf1_ma time max/min over seeds"], body))

    (HERE / "ALL_TABLES.md").write_text("# Stage 2 tables, A100 80GB PCIe (built by build_tables.py)\n\n" + "\n".join(parts))
    print(f"wrote {len(parts)} tables from {len(rows)} rows ({len(rows) - len(good)} errors)")


if __name__ == "__main__":
    main()
