# Final RF1 / RF2 results: loop-expand retune, A100 80GB PCIe, 15 Nov 2026

These are the final kernel results. The code is main at afc6600 plus the retuned
`rankfuse/kernels/tuned_a100.py` (tune job 15681134), which adds the expand mode (`loop_expand`,
commit f65d09a) to the per-width search. Sweep job 15684360 ran first the full test suite with the
table on (2291 passed; `tests/test_multi_adapter.py` excluded because `rankfuse.baselines.multi`
imports a missing `bgmv_forward`), then `rf2_ideal_bound`, `rf2_ideal` and `rf2_skew`. The
configs, shapes and seeds are the same as on 6 Nov (`../a100_2026-11-06/`), and B1 timed the same on
both days (new/old geomean 0.999). Settings: 4096 × 4096 fp16, add mode, CUDA-graph timing, L2
flushed, median over 3 seeds. Single-rank batches are not counted.

Files: `BY_RMAX.md` has every table with speedups. `RAW.md` and `raw_medians.csv` have every median
in µs. `COMPARE_NOV06.md` lists every workload old → new for each kernel. The raw CSVs are in
`results/raw/rf2_retuned/` (not in git).

## Main kernel: `rf2_ranges`

`rf2_ranges` assumes the scheduler gives a rank-sorted batch. It runs one RF1 launch per rank
bucket over that bucket's contiguous rows, each with its own tuned settings, on concurrent streams.

## Headline: speedup vs B1 and RF1, geomean (wins = more than 2% faster)

| r_max, phase | **`rf2_ranges` vs B1** | `rf2_ranges` vs RF1 | RF1 vs B1 | `rf2_bucketed_tuned` vs B1 | `rf2_segment` vs B1 | `rf2_index` (random) vs B1 | `rf2_ranges` vs B1, 6 Nov |
|---|---|---|---|---|---|---|---|
| 64 prefill | **1.87× (18/18)** | 1.09× (16/18) | 1.71× (18/18) | 1.79× (18/18) | 1.76× (18/18) | 1.73× (18/18) | 1.89× (18/18) |
| 64 decode | **1.96× (24/24)** | 1.11× (19/24) | 1.77× (24/24) | 1.85× (24/24) | 1.75× (24/24) | 1.71× (24/24) | 2.01× (24/24) |
| 128 prefill | **2.03× (30/30)** | 1.20× (28/30) | 1.68× (30/30) | 1.83× (30/30) | 1.80× (30/30) | 1.78× (30/30) | 1.95× (30/30) |
| 128 decode | **1.92× (48/48)** | 1.26× (42/48) | 1.52× (48/48) | 1.65× (48/48) | 1.63× (48/48) | 1.60× (48/48) | 1.89× (48/48) |
| 256 prefill | **1.76× (28/30)** | 1.75× (29/30) | 1.01× (5/30) | 1.53× (24/30) | 1.13× (22/30) | 1.12× (18/30) | 1.63× (24/30) |
| 256 decode | **1.67× (45/48)** | 1.91× (48/48) | 0.88× (12/48) | 1.32× (35/48) | 1.09× (36/48) | 1.06× (30/48) | 1.63× (41/48) |

Over all 198 sorted workloads, `rf2_ranges` beats B1 in 193 (1.85× geomean, up from 185 and 1.80×
on 6 Nov). It loses in 5 cases, all at r_max 256: `low_frac_25` (3) and `low_frac_50` (2). In those
mixes 50–75% of tokens are at full width, so there is little to skip. Those two mixes now average
1.03× and 1.10× vs B1 (6 Nov: 0.85× and 0.99×), and at 8192 tokens 1.18× and 1.24×.

One rank-256 token in 8192 (decode, `low_frac_100`):

| Kernel | µs |
|---|---|
| `rf2_ranges` | 142 |
| `rf2_bucketed_tuned` | 170 |
| `rf2_segment` | 331 |
| `rf2_index` | 336 |
| B1 | 353 |
| RF1 | 372 |

## What the retune changed

Widths 128 and 256 now use the column-loop expand in every token range (before, the tiled expand
was used above 128 tokens). RF1 with every token at width 256 and 8192 tokens went from 602 µs to
372 µs (B1: 355 µs). This makes RF1 a fairly tuned baseline: before the retune, part of RF2's gain
over RF1 at r_max 256 came from RF1 being held back by the tiled expand.

| Kernel, all sorted workloads vs B1 | 6 Nov | 15 Nov |
|---|---|---|
| `rf1_fused` | 1.12× | 1.31× |
| `rf2_ranges` | 1.80× (11 losses) | 1.85× (5 losses) |
| `rf2_bucketed_tuned` | 1.74× (12 losses) | 1.60× (14 losses) |
| `rf2_segment` | 1.48× (16 losses) | 1.44× (13 losses) |
| `rf2_index` (random order) | 1.40× (22 losses) | 1.41× (15 losses) |

`rf2_segment` and `rf2_bucketed_tuned` lose some speed on mostly-low-rank batches. Both look their
launch settings up for the whole batch, so their narrow blocks also run the width-256 loop-expand
entry. They are secondary kernels and are reported with the same table as the others.

## Tried and not kept

* **Rank chunks** (`tony/rank-chunk`, d63327f; job 15633364). The shrink splits its rank rows
  across programs, and the tiled expand walks the rank in chunks. The results were correct, but the
  tuning chose them in only one cell (+1%). Elsewhere they were 0–25% slower: each chunk re-reads
  `x`, and the loop expand already removes the register pressure they targeted.
* **Batch-dependent lookups for segment and bucketed** (job 15737074, 16 Nov; results kept
  untracked in `../a100_2026-11-16/`). Segment looked its settings up at its median token's width,
  and bucketed looked up each bucket at its own token count. Segment got worse (1.38×, 33 losses).
  Bucketed went to 1.57× with 6 losses, which is not a clear gain. Neither change was kept.

## Caveats

* Host time to build the plan is not counted: the per-bucket plan for `rf2_ranges` and
  `rf2_bucketed_tuned`, and `rf2_index`'s argsort (~690 µs over 8192 tokens).
* Per-bucket launch sizes depend on the bucket counts, so under CUDA graphs a step with different
  counts needs another graph or padding.
* The table was tuned on single-rank batches at 256, 1024, 2048 and 8192 tokens. Other token
  counts use the entry for their range (`TOKEN_RANGES` = 128 / 1024 / 4096).
