"""Rebuild tables/ and ALL_TABLES.md from raw/*.jsonl and source/ncu/*.csv (plain Python, no pandas).

    python3 build_tables.py

Speedups are time(baseline) / time(RF1-MA) on the same batch (same shape, adapters, popularity,
rank mix and tokens); summaries over several batches are geometric means with the min-max range.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW, TABLES, NCU = HERE / "raw", HERE / "tables", HERE / "source" / "ncu"
SHAPES = ["4096x4096", "q_o", "k_v", "gate_up", "down"]
SHAPE_DIMS = {"4096x4096": "4096->4096", "q_o": "896->896", "k_v": "896->128", "gate_up": "896->4864", "down": "4864->896"}
LABELS = ["decode-1", "decode-8", "decode-32", "decode-128", "decode-256", "prefill-2048", "prefill-8192"]
ADAPTERS = [1, 4, 16, 64, 256]
RF2 = ["rf2_ma_splitk", "rf2_ma_segment", "rf2_ma_bucketed"]


def load(name: str) -> list[dict]:
    rows = []
    for p in sorted(RAW.glob(f"{name}*.jsonl")):
        rows += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return [r for r in rows if not r.get("error") and r["median_us"] == r["median_us"]]


def key(r: dict, *extra) -> tuple:
    return (r["shape"], r["label"], r["num_adapters"], r["popularity"], r["dist"], *extra)


def index(rows, pages=None) -> dict:
    out = {}
    for r in rows:
        if pages is not None and r["pages"] != pages:
            continue
        out[(r["impl"], *key(r))] = r
    return out


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


def speedup_grid(idx, base: str, impl="rf1_ma", phase_labels=LABELS):
    """Per shape x adapters x label: ratios base/impl over popularity x dist."""
    cells = defaultdict(list)
    for (name, shape, label, n, pop, dist), r in idx.items():
        if name != impl:
            continue
        b = idx.get((base, shape, label, n, pop, dist))
        if b:
            cells[(shape, n, label)].append(b["median_us"] / r["median_us"])
    return cells


def main() -> None:
    parts = []
    main_rows = load("multi-main")
    idx = index(main_rows)

    # 01: RF1-MA time
    body = []
    for shape in SHAPES:
        for n in ADAPTERS:
            row = [SHAPE_DIMS[shape], n]
            for label in LABELS:
                ts = [r["median_us"] for (nm, s, l, nn, _, _), r in idx.items() if nm == "rf1_ma" and s == shape and l == label and nn == n]
                row.append(f"{gmean(ts):.1f}" if ts else "-")
            body.append(row)
    parts.append(write("01_rf1_ma_time", "01. RF1-MA GPU time (us)",
                       "CUDA-graph replay, add into the output, L2 flushed; geometric mean over popularity (uniform, Zipf 1.0) "
                       "x rank mix (uniform, skew_low, all_max). Weights in a shuffled page pool.",
                       ["shape", "adapters", *LABELS], body))

    # 02-05: speedups vs baselines
    for num, base, title in (("02", "b3_multi", "vLLM 0.32 ops, one padded copy per (adapter, rank) pair (B3-multi)"),
                             ("03", "b4_multi_copies", "S-LoRA bgmv, one adapter per (adapter, rank) pair (B4-multi copies)"),
                             ("04", "b4_multi_shared", "S-LoRA bgmv, one shared copy per adapter (B4-multi shared)"),
                             ("05", "b1_multi", "PyTorch, padded per adapter group (B1-multi)")):
        cells = speedup_grid(idx, base)
        body = [[SHAPE_DIMS[s], n, *[fmt_ratio(cells.get((s, n, l), [])) for l in LABELS]] for s in SHAPES for n in ADAPTERS]
        note = (f"time({base}) / time(rf1_ma) on the same batch; geometric mean (min-max) over popularity x rank mix. "
                "> 1: RF1-MA is faster.")
        if base.startswith("b4"):
            note += " Prefill columns: S-LoRA serves prompts with a different (Triton) kernel; these measure bgmv itself."
        parts.append(write(f"{num}_rf1_ma_vs_{base}", f"{num}. RF1-MA speedup over {title}", note,
                           ["shape", "adapters", *LABELS], body))

    # 06: RF2 variants vs RF1-MA
    body = []
    for impl in RF2:
        for phase in ("decode", "prefill"):
            for shape in SHAPES:
                cells = speedup_grid(idx, impl)  # impl / rf1 -> invert below
                ratios, best = [], None
                for (s, n, l), xs in cells.items():
                    if s != shape or not l.startswith(phase):
                        continue
                    for x in xs:
                        ratios.append(1 / x)
                for (nm, s, l, n, pop, dist), r in idx.items():
                    if nm != impl or s != shape or not l.startswith(phase):
                        continue
                    rf1 = idx.get(("rf1_ma", s, l, n, pop, dist))
                    if rf1:
                        sp = rf1["median_us"] / r["median_us"]
                        if best is None or sp > best[0]:
                            best = (sp, f"{l} N={n} {pop} {dist}")
                wins = sum(1 for x in ratios if x > 1.05)
                body.append([impl, phase, SHAPE_DIMS[shape], fmt_ratio(ratios), f"{wins}/{len(ratios)}",
                             f"{best[0]:.2f}x ({best[1]})" if best else "-"])
    parts.append(write("06_rf2_vs_rf1_ma", "06. RF2-multi variants vs RF1-MA",
                       "time(rf1_ma) / time(variant); > 1: the variant is faster. wins = batches where it is > 5% faster.",
                       ["variant", "phase", "shape", "speedup gmean (min-max)", "wins >5%", "best batch"], body))

    # 07: page-table overhead
    pages = load("multi-pages")
    pidx = {(r["impl"], r["pages"], *key(r)): r for r in pages}
    body = []
    for impl in ("rf1_ma", "rf2_ma_splitk"):
        for layout in ("contiguous", "shuffled"):
            for phase in ("decode", "prefill"):
                xs = []
                for (nm, pg, s, l, n, pop, dist), r in pidx.items():
                    if nm == impl and pg == layout and l.startswith(phase):
                        d = pidx.get((nm, "dense", s, l, n, pop, dist))
                        if d:
                            xs.append(r["median_us"] / d["median_us"])
                body.append([impl, f"{layout} / dense", phase, fmt_ratio(xs), len(xs)])
    xs = defaultdict(list)
    for (nm, s, l, n, pop, dist), r in idx.items():
        if nm == "rf1_ma" and n == 1:
            s1 = idx.get(("stage1_rf1_fused", s, l, n, pop, dist))
            if s1:
                xs[l.split("-")[0]].append(r["median_us"] / s1["median_us"])
    for phase, v in sorted(xs.items()):
        body.append(["rf1_ma (shuffled pool) / Stage 1 rf1_fused (no page table)", "1 adapter", phase, fmt_ratio(v), len(v)])
    parts.append(write("07_page_table_overhead", "07. Page-table overhead",
                       "Time ratio on the same batches (1.00x = no cost). dense = stacked weights behind an identity page "
                       "table; contiguous / shuffled = per-layer page pools with frames in order / permuted. Last rows: "
                       "RF1-MA vs Stage 1's rf1_fused, which reads one adapter with plain pointers.",
                       ["kernel", "layout", "phase", "time ratio gmean (min-max)", "batches"], body))

    # 08: few tokens per adapter (decode)
    body = []
    for shape in SHAPES:
        for label in ("decode-32", "decode-128", "decode-256"):
            for n in (16, 64, 256):
                for pop in ("uniform", "zipf"):
                    rs = [r for (nm, s, l, nn, p, _), r in idx.items() if nm == "rf1_ma" and s == shape and l == label and nn == n and p == pop]
                    if not rs:
                        continue
                    tpa = sum(r["tokens_per_active_adapter"] for r in rs) / len(rs)
                    eff = gmean([r["ideal_us"] / r["median_us"] for r in rs])
                    ratio = lambda b: fmt_ratio([idx[(b, *key(r))]["median_us"] / r["median_us"] for r in rs if (b, *key(r)) in idx])  # noqa: E731
                    body.append([SHAPE_DIMS[shape], label, n, pop, f"{tpa:.1f}",
                                 f"{gmean([r['median_us'] for r in rs]):.1f}", f"{100 * eff:.0f}%",
                                 ratio("b3_multi"), ratio("b4_multi_copies"), ratio("b4_multi_shared")])
    parts.append(write("08_few_tokens_per_adapter", "08. Decode with few tokens per adapter",
                       "RF1-MA at decode with many adapters (little reuse of a weight tile across tokens). tokens/adapter = "
                       "mean tokens per active adapter; roofline = ideal bytes (each active adapter's rows up to its group "
                       "limit, x, out read+write) at 4.8 TB/s divided by the measured time. Speedups as in 02-04.",
                       ["shape", "tokens", "adapters", "popularity", "tokens/adapter", "rf1_ma us", "of roofline",
                        "vs B3-multi", "vs B4 copies", "vs B4 shared"], body))

    # 09: ncu bytes
    body = []
    for p in sorted(NCU.glob("*.csv")):
        rows = list(csv.reader(p.open()))
        if len(rows) < 3:
            continue
        hdr = rows[0]
        data = rows[2:]  # row 1 holds units
        col = {h: i for i, h in enumerate(hdr)}
        unit = dict(zip(hdr, rows[1]))

        def total(m):
            scale = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "sector": 32, "nsecond": 1e-3, "ns": 1e-3, "usecond": 1, "us": 1, "ms": 1e3,
                     "msecond": 1e3}.get(unit.get(m, ""), 1)
            return sum(float(r[col[m]].replace(",", "")) for r in data if r[col[m]]) * scale

        impl, shape, n, req, tpr = p.stem.split("-")[:5]
        body.append([impl, shape, n, int(req) * int(tpr), len(data), f"{total('dram__bytes_read.sum') / 1e6:.2f}",
                     f"{total('dram__bytes_write.sum') / 1e6:.2f}", f"{total('lts__t_sectors.sum') / 1e6:.2f}",
                     f"{total('gpu__time_duration.sum'):.1f}"])
    parts.append(write("09_ncu_bytes", "09. Nsight Compute: DRAM and L2 traffic per call",
                       "One prepared call (caches flushed, clocks locked by ncu; compare bytes, not time, with other tables). "
                       "Summed over the call's kernels. L2 = lts__t_sectors x 32 B.",
                       ["impl", "shape", "adapters", "tokens", "kernels", "DRAM read MB", "DRAM write MB", "L2 MB",
                        "ncu time us"], body))

    sel = HERE / "tables" / "10_selector.md"
    if sel.exists():
        parts.append(sel.read_text())
    (HERE / "ALL_TABLES.md").write_text("# Stage 2 tables (built by build_tables.py)\n\n" + "\n".join(parts))
    print(f"wrote {len(parts)} tables")


if __name__ == "__main__":
    main()
