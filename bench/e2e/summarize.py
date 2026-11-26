"""Markdown tables from a results folder written by run_system.py / client.py.

    python bench/e2e/summarize.py <out dir> [--slo ttft1_tpot0.05 ...]

Reads <out>/summary.jsonl (one line per system x trace x rate; the last line of a
(tag, rate) pair wins) and <out>/*.memory.json. Prints:
  1. memory: model-loading memory, adapter memory (vs base), KV-cache memory and tokens;
  2. per trace: every system x rate (throughput, TTFT / TPOT percentiles, SLO attainment and
     goodput, mean KV usage, client lag flag);
  3. max throughput (rate inf) per system x trace;
  4. per-rank TTFT / TPOT medians for mixed-rank runs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DEFAULT_SLOS = ("ttft0.5_tpot0.05", "ttft1_tpot0.1", "ttft2_tpot0.1")


def load(out: Path):
    rows = {}
    for line in (out / "summary.jsonl").read_text().splitlines():
        if line.strip():
            s = json.loads(line)
            rows[(s["tag"], s["rate"])] = s
    mem = {}
    for f in sorted(out.glob("*.memory.json")):
        m = json.loads(f.read_text())
        mem[m["system"]] = m
    return list(rows.values()), mem


def ms(x):
    return "-" if x is None else f"{x * 1e3:.0f}" if x >= 0.1 else f"{x * 1e3:.1f}"


def fmt_rate(r):
    return "inf" if (isinstance(r, float) and math.isinf(r)) or r == "inf" else f"{r:g}"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--slo", nargs="+", default=list(DEFAULT_SLOS))
    a = p.parse_args(argv)
    out = Path(a.out)
    rows, mem = load(out)
    lines = []

    if mem:
        base = mem.get("base", {}).get("model_load_gib")
        lines += ["## Memory (from the server logs)", "",
                  "| system | model loading GiB | adapter GiB (vs base) | KV cache GiB | KV tokens | startup s |",
                  "|---|---|---|---|---|---|"]
        for sysname, m in mem.items():
            ml = m.get("model_load_gib")
            ad = None if (ml is None or base is None) else ml - base
            lines.append(f"| {sysname} | {ml} | {'-' if ad is None else f'{ad:.2f}'} | {m.get('kv_avail_gib')} | "
                         f"{m.get('kv_tokens')} | {m.get('startup_s', 0):.0f} |")
        lines.append("")

    traces = sorted({r["tag"].split("__", 1)[1] for r in rows})
    for tr in traces:
        sel = sorted((r for r in rows if r["tag"].endswith("__" + tr)),
                     key=lambda r: (r["tag"], math.inf if fmt_rate(r["rate"]) == "inf" else r["rate"]))
        lines += [f"## Trace `{tr}`", "",
                  "| system | rate | done/n | req/s | out tok/s | TTFT p50/p90/p99 ms | TPOT p50/p90/p99 ms | "
                  + " | ".join(f"{s} attain / goodput" for s in a.slo) + " | KV mean | lag |",
                  "|---|---|---|---|---|---|---|" + "---|" * len(a.slo) + "---|---|"]
        for r in sel:
            t, q = r["ttft"], r["tpot"]
            slo = [r["slo"].get(s) for s in a.slo]
            kv = (r.get("server") or {}).get("kv_cache_usage")
            kv_s = "-" if not kv else f"{kv['mean'] * 100:.0f}%"
            lines.append(
                f"| {r['tag'].split('__')[0]} | {fmt_rate(r['rate'])} | {r['completed']}/{r['n']} | {r['req_per_s']:.2f} | "
                f"{r['out_tok_per_s']:.0f} | {ms(t['p50'])}/{ms(t['p90'])}/{ms(t['p99'])} | "
                f"{ms(q['p50'])}/{ms(q['p90'])}/{ms(q['p99'])} | "
                + " | ".join("-" if v is None else f"{v['attain'] * 100:.0f}% / {v['goodput']:.2f}" for v in slo)
                + f" | {kv_s} | "
                f"{'LAG' if r.get('client_lag_flag') else 'ok'} |")
        lines.append("")

    inf_rows = [r for r in rows if fmt_rate(r["rate"]) == "inf"]
    if inf_rows:
        lines += ["## Max throughput (all requests at t = 0)", "", "| system | trace | req/s | out tok/s | total tok/s |",
                  "|---|---|---|---|---|"]
        for r in sorted(inf_rows, key=lambda r: (r["tag"].split("__")[1], -r["out_tok_per_s"])):
            s, tr = r["tag"].split("__", 1)
            lines.append(f"| {s} | {tr} | {r['req_per_s']:.2f} | {r['out_tok_per_s']:.0f} | {r['total_tok_per_s']:.0f} |")
        lines.append("")

    mixed = [r for r in rows if len(r.get("by_rank") or {}) > 1]
    if mixed:
        lines += ["## Per-rank medians (mixed-rank runs)", "", "| system / trace | rate | rank: TTFT p50 ms / TPOT p50 ms |",
                  "|---|---|---|"]
        for r in sorted(mixed, key=lambda r: (r["tag"], math.inf if fmt_rate(r["rate"]) == "inf" else r["rate"])):
            cells = ", ".join(f"r{k}: {ms(v['ttft_p50'])} / {ms(v['tpot_p50'])}" for k, v in r["by_rank"].items())
            lines.append(f"| {r['tag']} | {fmt_rate(r['rate'])} | {cells} |")
        lines.append("")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
