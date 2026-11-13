# RF2 sweeps after per-width tuning, A100 80GB PCIe, 6 Nov 2026

Commit 12bf0f6 (first tuned table, `rankfuse/kernels/tuned_a100.py`), add mode, CUDA-graph timing,
3 seeds, 4096 × 4096 fp16. Jobs 15253899 (2270 tests passed with the table on, then the
`rf2_ideal_bound`, `rf2_ideal` and `rf2_skew` sweeps) and 15253901 (RF1 ablation). Raw data:
`results/raw/rf2_tuned/` in the frozen worktree. Full tables with speedups: `BY_RMAX.md`;
every median in µs, one row per workload: `RAW.md`, and the same as a CSV for analysis by hand:
`raw_medians.csv`; ablation: `RF1_ABLATION.md`.
Single-rank batches (`all_max`, `all_min`) are not counted. Geomeans over mixes and batch sizes.

Not final: a retune that adds loop-vs-tiled expand (commit f65d09a, job 15336639) may change
widths 128 and 256; these sweeps will be rerun if it does.

## Kernels

| | Assumption | Launch |
|---|---|---|
| `rf2_segment` (original) | Physically sorted tokens | One kernel with a width branch |
| `rf2_index` (original) | Sorted index list | One kernel with a width branch |
| `rf2_ranges` (new) | Physically sorted tokens | One RF1 launch per bucket, concurrent streams |
| `rf2_bucketed_tuned` | Index list per bucket | One RF1 launch per bucket, concurrent streams |

Tuning (one table by width and token count, shared with RF1) and per-bucket launches are the two
changes over the original. Per-bucket launches and streams are RF2-only (they need more than one width).

## Headline: speedup vs RF1 / vs B1 (wins = more than 2% faster)

| r_max, phase | `rf2_ranges` sorted vs RF1 | vs B1 | segment sorted vs RF1 | index random vs RF1 | `rf2_bucketed_tuned` vs B1 |
|---|---|---|---|---|---|
| 64 prefill | 1.10× (18/18) | 1.89× (18/18) | 1.02× (8/18) | 1.00× (6/18) | 1.82× (18/18) |
| 64 decode | 1.12× (23/24) | 2.01× (24/24) | 1.02× (12/24) | 0.99× (3/24) | 1.90× (24/24) |
| 128 prefill | 1.27× (22/24) | 1.92× (24/24) | 1.20× (22/24) | 1.17× (22/24) | 1.85× (24/24) |
| 128 decode | 1.41× (37/39) | 1.86× (39/39) | 1.23× (38/39) | 1.19× (37/39) | 1.78× (39/39) |
| 256 prefill | 2.06× (23/24) | 1.54× (18/24) | 1.52× (22/24) | 1.38× (20/24) | 1.46× (18/24) |
| 256 decode | 2.20× (39/39) | 1.56× (32/39) | 1.57× (38/39) | 1.45× (36/39) | 1.53× (33/39) |

One rank-256 token in 8192 (decode): `rf2_ranges` 142 µs, `rf2_bucketed_tuned` 153, segment 279,
index 329, B1 355, RF1 602, ideal (all rank 1) 139. On random input `rf2_ranges` falls back to RF1 (603).

## Tuning vs per-bucket launches, speedup vs B1

Same cases for every row (two-rank mixes only at r_max 128: skewed mixes were not run there on 5 Nov).
B1 timed identically on both days (1.00×).

| | r_max 128 | r_max 256 |
|---|---|---|
| RF1 untuned → tuned | 1.09× → 1.38× | 0.58× → 0.72× |
| Original: segment, sorted, untuned (5 Nov) | 1.43× | 0.87× (loses 50/63) |
| + tuning: segment, sorted, tuned | 1.68× | 1.12× (wins 47/63) |
| + per-bucket launches: `rf2_ranges` | 1.86× | 1.55× (wins 50/63) |

Tuning contributes about 1.17× (r_max 128) and 1.29× (256); per-bucket launches 1.10× and 1.39×.
The split depends on the order the steps are applied in, so treat it as approximate.

Within per-bucket launches, the concurrency is what helps: run one after another (`rf2_ranges_seq`)
they are 0.91× / 0.88× / 1.10× of segment at r_max 64 / 128 / 256; streams add 1.20–1.26×.

## RF1 vs B1 at full width

Tuning took RF1 at r_max 256, 8192 tokens, from 805 to 602 µs; B1 is 355 µs. RF1 alone is still
about 1.4× slower than B1 at width 256.

## RF1 ablation with the tuned table

Every shared optimization still helps RF1, except the tiled expand at r_max 128: turning it off is
1.12–1.25× faster (512–8192 tokens). The tuning search did not include the loop expand; commit
f65d09a adds it.

## Caveats

* Host plan time is not in these numbers: `rf2_index`'s argsort (~690 µs over 8192 tokens), the
  per-bucket plans of `rf2_bucketed_tuned` and `rf2_ranges` (built per request by a scheduler,
  expected to be microseconds, not measured).
* Per-bucket launch sizes depend on the bucket counts, so under CUDA graphs a step with different
  counts needs another graph or padding; a device-side plan is future work.
