# RankFuse results

Start here.

| File | What it covers |
|---|---|
| [RF1_HISTORY.md](RF1_HISTORY.md) | RF1 from the one-kernel design to split-K, the tiled expand, per-width tuning and the ablation |
| [RF2_HISTORY.md](RF2_HISTORY.md) | RF2 from the first sorted kernel to `rf2_segment`, `rf2_index`, the bucketed kernels and `rf2_ranges` |
| [h200_2026-10-25/](h200_2026-10-25/README.md) | H200 tables: RF1 design steps, RF1 vs B1 and vLLM (B3), whole-model and vLLM step time, first RF2 versions |
| [a100_2026-10-26/](a100_2026-10-26/ALL_ROWS.md) | A100 sweeps at r_max 64 and 128, and the H200 shapes replayed |
| [a100_2026-11-05/](a100_2026-11-05/SUMMARY.md) | RF2 on controlled mixes at r_max 64/128/256, overhead, the two assumptions, kernel diagnostics |
| [a100_2026-11-06/](a100_2026-11-06/SUMMARY.md) | After per-width tuning: `rf2_ranges`, tables by r_max (`BY_RMAX.md`, with speedups), raw µs (`RAW.md`, `raw_medians.csv`), RF1 ablation |
| [a100_2026-11-15/](a100_2026-11-15/SUMMARY.md) | **Final.** Loop-expand retune: `rf2_ranges` as the main kernel, tables by r_max, raw µs, every workload vs 6 Nov (`COMPARE_NOV06.md`) |
| `slides/` | H200 tables formatted for slides |
| `raw/` | Raw sweep CSVs (not in git) |

## Where things stand (15 Nov, final)

* **RF2, main kernel `rf2_ranges`** (rank-sorted batch, one RF1 launch per bucket, concurrent
  streams): beats B1 in 193 of 198 mixed-rank workloads, 1.85× geomean (1.67–2.03× by r_max and
  phase). It beats RF1 by 1.09–1.26× at r_max 64 and 128 and 1.75–1.91× at 256. The 5 losses are
  at r_max 256 on `low_frac_25` / `low_frac_50`, where most tokens are at full width.
* **RF1** beats B1 by 1.7–1.8× at r_max 64 and 1.5–1.7× at 128. At r_max 256 it ties B1 on
  prefill (1.01×) and trails on decode (0.88×). With every token at full width it is 372 vs 355 µs.
* The retune that adds the loop expand is in. The rank-chunk expand and batch-dependent lookups for
  segment and bucketed were tried and not kept (see the 15 Nov summary).
* Not counted: host time to build the plan. Per-bucket launches under CUDA graphs need one graph
  per bucket layout.
