"""Metrics of one replayed run: throughput, latency percentiles, SLO attainment, goodput.

Definitions (per request; times measured by the client, wall clock, from the moment the HTTP
request is sent):

    TTFT      first streamed chunk that carries >= 1 output token, minus send time
    TPOT      (end - first token) / (output tokens - 1): mean time per output token after the
              first (requests with one output token have no TPOT)
    ITL       every gap between consecutive token-carrying chunks, divided evenly when one
              chunk carries k > 1 tokens (pooled over all requests)
    E2E       end of stream minus send time

Run-level:

    duration          first send to last completion (of the measured requests)
    req/s, tok/s      completed requests / duration; output tokens / duration; (prompt + output)
                      tokens / duration
    SLO attainment    fraction of ALL measured requests (failed or unfinished count as misses)
                      with TTFT <= ttft_slo AND TPOT <= tpot_slo, for every pair in the grid
    goodput           requests meeting the SLO / duration (DistServe's definition), per pair
    client lag        actual send time minus scheduled send time (p99, max): if it is large the
                      client, not the server, set the arrival process; the run is then flagged

Percentiles are p50 / p90 / p99 (numpy linear interpolation).
"""

from __future__ import annotations

import math

import numpy as np

# Default SLO grid (seconds). Override per model in the run config.
TTFT_SLOS = (0.25, 0.5, 1.0, 2.0)
TPOT_SLOS = (0.025, 0.05, 0.1)
PCTS = (50, 90, 99)
LAG_WARN_S = 0.05


def pct(x, ps=PCTS) -> dict:
    x = np.asarray([v for v in x if v is not None and not math.isnan(v)], dtype=float)
    if x.size == 0:
        return {f"p{p}": None for p in ps} | {"mean": None}
    return {f"p{p}": float(np.percentile(x, p)) for p in ps} | {"mean": float(x.mean())}


def per_request(rec: dict) -> dict:
    """ttft / tpot / e2e / itl of one client record (None when not defined)."""
    ok = rec.get("error") is None and rec.get("n_out", 0) > 0 and rec.get("first") is not None
    if not ok:
        return dict(ok=False, ttft=None, tpot=None, e2e=None, itl=[])
    ttft = rec["first"] - rec["sent"]
    e2e = rec["end"] - rec["sent"]
    n = rec["n_out"]
    tpot = (rec["end"] - rec["first"]) / (n - 1) if n > 1 else None
    itl = []
    times, toks = rec.get("chunk_t") or [], rec.get("chunk_n") or []
    for k in range(1, len(times)):
        if toks[k] > 0:
            gap = (times[k] - times[k - 1]) / toks[k]
            itl.extend([gap] * toks[k])
    return dict(ok=True, ttft=ttft, tpot=tpot, e2e=e2e, itl=itl)


def summarize(records: list[dict], ttft_slos=TTFT_SLOS, tpot_slos=TPOT_SLOS, by_rank: bool = True) -> dict:
    if not records:
        return {}
    pr = [per_request(r) for r in records]
    ok = [p for p in pr if p["ok"]]
    t0 = min(r["sent"] for r in records)
    t1 = max((r["end"] for r, p in zip(records, pr) if p["ok"]), default=t0)
    dur = max(t1 - t0, 1e-9)
    n_ok = len(ok)
    out_tok = sum(r["n_out"] for r, p in zip(records, pr) if p["ok"])
    in_tok = sum(r["prompt_len"] for r, p in zip(records, pr) if p["ok"])
    lag = [r["sent"] - r["scheduled"] for r in records if r.get("scheduled") is not None]
    s = dict(
        n=len(records), completed=n_ok, failed=len(records) - n_ok, duration_s=dur,
        req_per_s=n_ok / dur, out_tok_per_s=out_tok / dur, total_tok_per_s=(in_tok + out_tok) / dur,
        out_len_mismatch=sum(1 for r, p in zip(records, pr) if p["ok"] and r["n_out"] != r["output_len"]),
        ttft=pct([p["ttft"] for p in ok]), tpot=pct([p["tpot"] for p in ok]),
        itl=pct([g for p in ok for g in p["itl"]]), e2e=pct([p["e2e"] for p in ok]),
        client_lag=dict(p99=float(np.percentile(lag, 99)) if lag else None, max=max(lag) if lag else None),
    )
    s["client_lag_flag"] = bool(lag) and s["client_lag"]["p99"] > LAG_WARN_S
    slo = {}
    for a in ttft_slos:
        for b in tpot_slos:
            met = sum(1 for p in pr if p["ok"] and p["ttft"] <= a and (p["tpot"] is None or p["tpot"] <= b))
            slo[f"ttft{a:g}_tpot{b:g}"] = dict(attain=met / len(records), goodput=met / dur)
    for a in ttft_slos:
        met = sum(1 for p in pr if p["ok"] and p["ttft"] <= a)
        slo[f"ttft{a:g}"] = dict(attain=met / len(records), goodput=met / dur)
    s["slo"] = slo
    if by_rank:
        groups = {}
        for r, p in zip(records, pr):
            groups.setdefault(str(r.get("rank")), []).append(p)
        s["by_rank"] = {
            k: dict(n=len(v), ttft_p50=pct([p["ttft"] for p in v if p["ok"]])["p50"],
                    tpot_p50=pct([p["tpot"] for p in v if p["ok"]])["p50"])
            for k, v in sorted(groups.items(), key=lambda kv: (kv[0] == "None", int(kv[0]) if kv[0] != "None" else 0))
        }
    return s


def server_timeseries_summary(samples: list[dict]) -> dict:
    """Mean / max of the /metrics gauges sampled during a run."""
    out = {}
    for key in ("kv_cache_usage", "running", "waiting"):
        v = [s[key] for s in samples if s.get(key) is not None]
        out[key] = dict(mean=float(np.mean(v)), max=float(np.max(v))) if v else None
    for key in ("preemptions", "prefix_hits", "prefix_queries"):
        v = [s[key] for s in samples if s.get(key) is not None]
        out[key + "_delta"] = float(v[-1] - v[0]) if len(v) >= 2 else None
    return out
