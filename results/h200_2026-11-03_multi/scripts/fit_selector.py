"""Fit the CPU-side choice between rf1_ma and rf2_ma_bucketed on raw/multi-main.jsonl.

    python results/h200_2026-11-03_multi/scripts/fit_selector.py      (from the repository root)

Features are ones a scheduler has per step without touching the GPU: tokens T, h_in, and
work_ratio = sum over tokens of bucket(rank) / (tokens x bucket(batch maximum rank)), i.e. the rank
rows rf2_ma_bucketed multiplies relative to rf1_ma, whose MMA is R_PAD wide for every tile (Stage 1's
Plan.work_ratio, here BucketPlan.work_ratio).
The rule is searched over a small threshold grid to maximise the geometric-mean speedup over
always running rf1_ma, ON THE SAME DATA it is then evaluated on (no held-out set). Writes
tables/10_selector.md and prints the chosen thresholds.
"""

import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parents[1]

from rankfuse.multi import MultiWorkload, sample_requests  # noqa: E402


def bucket(r):
    return 0 if r <= 0 else max(16, 1 << (r - 1).bit_length())


def work_ratio(w: MultiWorkload) -> float:
    adapters, ranks = sample_requests(w)
    live = [r for r in ranks if r > 0]
    full = bucket(max(live)) if live else 0
    return sum(bucket(r) for r in live) / (len(live) * full) if full else 1.0


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    rows = [json.loads(line) for line in (HERE / "raw" / "multi-main.jsonl").read_text().splitlines()]
    by = {}
    for r in rows:
        if r["impl"] in ("rf1_ma", "rf2_ma_bucketed") and not r.get("error"):
            k = (r["shape"], r["label"], r["num_adapters"], r["popularity"], r["dist"])
            by.setdefault(k, {})[r["impl"]] = r
    cases = []
    for k, d in by.items():
        if len(d) < 2:
            continue
        r = d["rf1_ma"]
        w = MultiWorkload(r["num_requests"], r["tokens_per_request"], r["num_adapters"], popularity=r["popularity"],
                          dist=r["dist"], h_in=r["h_in"], h_out=r["h_out"], seed=r["seed"])
        cases.append(dict(T=r["num_tokens"], h_in=r["h_in"], wr=work_ratio(w), rf1=r["median_us"],
                          bk=d["rf2_ma_bucketed"]["median_us"], phase=r["phase"]))
    best = None
    for wr_max, t_min, t_max, h_min in itertools.product((0.3, 0.4, 0.5, 0.6, 0.7), (1, 32, 64, 128, 256),
                                                         (256, 8192), (0, 2048)):
        def pick(c):
            return c["wr"] <= wr_max and t_min <= c["T"] <= t_max and c["h_in"] >= h_min
        sp = [c["rf1"] / (c["bk"] if pick(c) else c["rf1"]) for c in cases]
        g = gmean(sp)
        if best is None or g > best[0]:
            best = (g, (wr_max, t_min, t_max, h_min), min(sp), sum(pick(c) for c in cases))
    g, (wr_max, t_min, t_max, h_min), worst, n_pick = best
    oracle = gmean([c["rf1"] / min(c["rf1"], c["bk"]) for c in cases])
    always = gmean([c["rf1"] / c["bk"] for c in cases])
    chosen = [c for c in cases if c["wr"] <= wr_max and t_min <= c["T"] <= t_max and c["h_in"] >= h_min]
    sub = gmean([c["rf1"] / c["bk"] for c in chosen]) if chosen else float("nan")
    text = f"""### 10. Selector: rf2_ma_bucketed or rf1_ma (fitted on the same data)

Rule (`rankfuse.kernels.rankfuse_multi.use_bucketed`): run rf2_ma_bucketed when work_ratio <= {wr_max},
{t_min} <= T <= {t_max} and h_in >= {h_min}; else rf1_ma. work_ratio = rank rows the bucketed kernels
multiply / rank rows rf1_ma multiplies (each token at its own bucket vs every token at the batch's bucket).
Searched over a small threshold grid on the {len(cases)} main-sweep batches and evaluated on the same
batches: an optimistic, in-sample estimate.

| policy | speedup over always rf1_ma (gmean) |
|---|---|
| always rf2_ma_bucketed | {always:.3f}x |
| selector ({n_pick}/{len(cases)} batches pick bucketed; worst batch {worst:.2f}x) | {g:.3f}x |
| speedup on the batches it picks | {sub:.3f}x |
| oracle (faster of the two per batch) | {oracle:.3f}x |
"""
    (HERE / "tables" / "10_selector.md").write_text(text)
    print(text)
    print("THRESHOLDS", wr_max, t_min, t_max, h_min)


if __name__ == "__main__":
    main()
