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
| `slides/` | H200 tables formatted for slides |
| `raw/` | Raw sweep CSVs (not in git) |

## Where things stand (6 Nov)

* **RF1** beats B1 by about 1.8× at r_max 64 and 1.4× at r_max 128 on every mixed-rank workload
  tested. At r_max 256 with everything at full width it is still about 1.4× slower than B1.
* **RF2** with per-bucket concurrent launches beats RF1 at every r_max and B1 on nearly every
  mixed-rank workload: `rf2_ranges` (physically sorted tokens) 155 of 168 cases, 1.5–2.0× geomean
  over B1; `rf2_bucketed_tuned` (index list per bucket) 156 of 168. The losses are at r_max 256 on
  mixes where most tokens are at full width.
* Not counted: host time to build the plan; per-bucket launches under CUDA graphs need one graph
  per bucket layout.
* Pending: a retune with the loop expand (commit f65d09a, job 15336639) may move r_max 128 and 256.
