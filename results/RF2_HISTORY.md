# RF2 results, start to finish

RF2 is RF1 plus rank skipping: each token is computed at its own rank bucket (16, 32, 64, 128 or
256 rows, the next power of two at or above its rank) instead of the batch's highest rank. This
file follows the RF2 kernels from the first versions to the current ones and what each stage
measured. Speedup = baseline time ÷ RF2 time (above 1× = RF2 faster). GPU time (CUDA graph replay),
fp16, a 4096 × 4096 layer. Single-rank batches (`all_min`, `all_max`) are not counted: they have
nothing to skip. RF1's own history is in `RF1_HISTORY.md`.

## The kernels

| Kernel | Assumption | How it skips | Added |
|---|---|---|---|
| `rf2_fused` | Tokens sorted on the GPU | One kernel, no split-K | Oct (H200) |
| `rf2_splitk` / `rf2_index` | A sorted index list (tokens in any order) | One split-K kernel; each block branches to its width at run time | Oct; restored as `rf2_index` 4 Nov |
| `rf2_segment` | Physically sorted tokens | Same, on consecutive rows, no index | Oct |
| `rf2_bucketed` / `rf2_bucketed_tuned` | An index list per bucket | One RF1 launch per bucket, concurrent streams | Oct; per-bucket settings later |
| `rf2_ranges` | Physically sorted tokens | One RF1 launch per bucket on its contiguous rows, concurrent streams, no index | 6 Nov |

Two families: **single launch** (one kernel with a run-time width branch: `rf2_segment`,
`rf2_index`) and **per-bucket launches** (separate RF1 launches per width, run at the same time:
`rf2_bucketed_tuned`, `rf2_ranges`). `rf2_ranges` falls back to one RF1 launch on unsorted input.

## Stage 1: first versions, r_max 64 (H200, Oct)

Source: `h200_2026-10-25/ALL_TABLES.md` (tables 07–09).

| Kernel | vs RF1 |
|---|---|
| `rf2_fused` (sorted, one kernel) | 2.7–4.7× slower: no split-K |
| Sorted split-K | 0.87–1.07×; only wins at 2048–8192 decode tokens, plus a 6–46 µs sort |
| `rf2_segment` | 1.06–1.11× on large prompt batches mixing low ranks with one rank-64 request; 0.92–1.02× elsewhere |
| `rf2_bucketed`, concurrent streams | 1.08–1.15× on the same batches; 0.86–0.99× when there is little to skip |
| `rf2_bucketed`, one stream | Slower than RF1 (e.g. 67 vs 63 µs) |

Conclusion then: at r_max 64 skipping helps only in one regime, by about 10%. The concurrent-stream
result already showed that per-bucket launches need to overlap to pay.

## Stage 2: A100 sweeps (26 Oct)

Source: `a100_2026-10-26/ALL_ROWS.md`, `rf2_segment`, sampled mixes.

| Sweep | Phase | RF2 vs RF1 | RF2 vs B1 |
|---|---|---|---|
| r_max 64 | decode / prefill | 0.93× / 0.92× | 1.89× / 1.69× |
| r_max 128 | decode / prefill | 0.95× / **1.07×** | 1.65× / 1.52× |
| H200 shapes | decode / prefill | 0.95× / 0.96× | 1.54× / 1.27× |

RF2 beat B1 everywhere (through RF1's shared optimizations) but was slightly slower than RF1, except
r_max 128 prefill: the first sign that skipping pays more at higher r_max.

## Stage 3: controlled mixes and diagnostics (5 Nov, A100 80GB PCIe)

Source: `a100_2026-11-05/SUMMARY.md`, `REPORT_*.md`. New workloads: two-rank mixes (25–100% of
requests at rank 1, the rest at r_max) and sampled skewed mixes, r_max 64/128/256, random and
sorted order.

| Finding | Numbers |
|---|---|
| Overhead with nothing to skip | `rf2_segment` +5–13% at 1–128 tokens; up to +13% at r_max 256 full width |
| r_max 128: clean wins | 8192 tokens, all but one at rank 1: segment 1.68× vs RF1, 1.78× vs B1 |
| r_max 64 | At most ~8% over RF1 |
| r_max 256: RF1 artifact | RF2's 1.9–2.1× over RF1 mostly recovered RF1's untuned width-256 speed; segment 0.94× vs B1 with one r_max request |
| Index cost | `rf2_index` 4–11% slower than `rf2_segment` on the same blocks, plus a ~690 µs host sort |
| Order | Matters only when high ranks are common: 90% low, segment random 799 µs vs sorted 428 µs |

What limited the single-launch kernels (one r_max 256 request in 8192 tokens; ideal 139 µs):

| Kernel | µs | Difference |
|---|---|---|
| `rf2_segment`, sorted | 379 | |
| `rf2_bucketed` | 205 | Each launch compiled for one width: no run-time branch (~1.85×) |
| `rf2_bucketed_tuned` | 151 | Bigger blocks for narrow buckets (~1.36×) |

## Stage 4: per-width tuning and `rf2_ranges` (6 Nov, A100 80GB PCIe)

Source: `a100_2026-11-06/SUMMARY.md`, `BY_RMAX.md`. Two changes:

1. **Per-width tuning**, one settings table shared with RF1 (see `RF1_HISTORY.md`, stage 4).
2. **`rf2_ranges`**: per-bucket concurrent launches on physically sorted rows, no index list.

Speedup, geomean (wins = cases more than 2% faster):

| r_max, phase | `rf2_ranges` sorted vs RF1 | vs B1 | `rf2_bucketed_tuned` vs B1 | segment sorted vs RF1 | index random vs RF1 |
|---|---|---|---|---|---|
| 64 prefill | 1.10× (18/18) | 1.89× (18/18) | 1.82× (18/18) | 1.02× | 1.00× |
| 64 decode | 1.12× (23/24) | 2.01× (24/24) | 1.90× (24/24) | 1.02× | 0.99× |
| 128 prefill | 1.27× (22/24) | 1.92× (24/24) | 1.85× (24/24) | 1.20× | 1.17× |
| 128 decode | 1.41× (37/39) | 1.86× (39/39) | 1.78× (39/39) | 1.23× | 1.19× |
| 256 prefill | 2.06× (23/24) | 1.54× (18/24) | 1.46× (18/24) | 1.52× | 1.38× |
| 256 decode | 2.20× (39/39) | 1.56× (32/39) | 1.53× (33/39) | 1.57× | 1.45× |

One r_max 256 request in 8192 tokens: `rf2_ranges` **142 µs** (ideal 139), `rf2_bucketed_tuned` 153,
segment 279, index 329, B1 355, RF1 602.

Where the gain over B1 comes from (same cases; r_max 128 two-rank mixes only):

| | r_max 128 | r_max 256 |
|---|---|---|
| Original: `rf2_segment`, sorted, untuned | 1.43× | 0.87× (loses 50/63) |
| + tuning | 1.68× | 1.12× (wins 47/63) |
| + per-bucket concurrent launches (`rf2_ranges`) | 1.86× | 1.55× (wins 50/63) |

Within per-bucket launches, the overlap is what pays: launched one after another (`rf2_ranges_seq`)
they are 0.91× / 0.88× / 1.10× of segment at r_max 64 / 128 / 256; concurrent streams add 1.20–1.26×.

## Where RF2 stands

| Assumption | Best kernel | Result |
|---|---|---|
| Physically sorted tokens | `rf2_ranges` | Beats RF1 at every r_max and B1 in 155 of 168 mixed-rank cases |
| Index list per bucket, any order | `rf2_bucketed_tuned` | About 8% behind `rf2_ranges`; beats B1 in 156 of 168 |
| No help from the scheduler (random order) | `rf2_index` | Beats RF1 at r_max 128 and 256 (1.17–1.45×); ties at 64 |

* The remaining losses to B1 are at r_max 256, on mixes where most tokens are at full width: there
  RF1's own width-256 gap (`RF1_HISTORY.md`, stage 4) shows through.
* Not counted in these times: building the plan on the host (the sort for `rf2_index`, about 690 µs
  over 8192 tokens; per-bucket plans built per request by a scheduler, not measured).
* Per-bucket launch sizes depend on the bucket counts, so under CUDA graphs a step with different
  counts needs another graph or padding. A device-side plan is future work.
* These are the numbers with the first tuned table. The loop-expand retune (commit f65d09a, job
  15336639) may move r_max 128 and 256 slightly.
