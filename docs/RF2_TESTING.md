# RF2 testing: kernels, assumptions and why each test exists

This explains how RF1 and the RF2 variants differ, which scheduler assumptions we test, and why
each sweep is needed. Measured numbers so far are from one NVIDIA A100 80GB PCIe (the 5 Nov
sweep at commit 7871e06, before the tuned launch table of 12bf0f6); every new sweep is pinned to
that GPU type.

## What every kernel computes
For each token t with rank r_t: `out[t] (+)= scale(r_t) · (x[t] · A_t[:r_t]ᵀ) · B[:r_t]`, using one
shared adapter copy (`A_t [r_max, h_in]`, `B [r_max, h_out]`; rank r is the first r rows of each).

### RF1 (`rf1_fused`): one width for the whole batch
Every request is padded to the **batch's** maximum rank, then masked back down to its own rank.
```
W = next_pow2(max rank in batch), at least 16        # ranks [8, 8, 200] → W = 256
for each block of BM consecutive tokens (batch order):
    shrink: v = x_block · A_t[:W]ᵀ                      # every token pays W rows, split-K over h_in
    expand: v[t, r_t:] = 0                             # token t keeps only its own r_t columns
            out_block (+)= scale · v · B[:W]           # still W rows of B for every token
```
The mask makes the result correct, but a rank-8 token still does the work of rank 256.

### RF2: a width per block
Each block reads only as many rows of `A_t` and `B` as its widest token needs:
```
for each block:
    W_b = next_pow2(max rank in this block), at least 16
    shrink: v = x_block · A_t[:W_b]ᵀ
    expand: v[t, r_t:] = 0; out_block (+)= scale · v · B[:W_b]
```
A block can still mix ranks, so tokens below `W_b` are masked, as in RF1. The variants differ in
where the rank order comes from and how `W_b` is chosen:

| Kernel | What it receives | How it forms blocks | How it picks `W_b` |
|---|---|---|---|
| `rf2_segment` | `x` and ranks in memory order; **no list**. Assumes the scheduler already put requests in rank order in memory | Next BM consecutive rows | Reads the block's ranks in the kernel, takes the max, branches to the matching compiled width |
| `rf2_index` | `x` and ranks in batch order, **plus a sorted index list** | Next BM entries of the list; looks up each token's row, writes the result back to that row | Same run-time read and branch as segment |
| `rf2_bucketed` | `x`, ranks, and a CPU plan (**one index list per bucket**) | Separate launch per bucket over that bucket's list | Fixed per launch at compile time, so no read or branch |
| `rf2_ranges` | Physically sorted `x` and ranks, and a CPU plan (**one row range per bucket**) | Separate launch per bucket over that bucket's contiguous rows | Fixed per launch at compile time, so no read or branch |

With sorted input, `rf2_segment` and `rf2_index` form the same blocks at the same widths.
`rf2_segment` gets the order for free from the memory layout; `rf2_index` pays one index load per
row, and an argsort once per step, so that no data has to move. Neither sorts inside the kernel.
On unsorted input without a list, `rf2_segment` still runs correctly but its blocks mix ranks, so
it skips less.

Example, ranks `[8, 200, 8, 8, 8, 8, 8, 200]`, BM = 4 (real blocks are 16–64 tokens):

| Kernel | Blocks | Widths |
|---|---|---|
| RF1 | all 8 tokens | 256, 256 |
| `rf2_segment`, this memory order | `[8,200,8,8]`, `[8,8,8,200]` | 256, 256: no gain |
| `rf2_segment`, physically sorted | `[8,8,8,8]`, `[8,8,200,200]` | 16, 256 |
| `rf2_index`, sorted list | positions 0,2,3,4 then 5,6,1,7 | 16, 256 |
| `rf2_bucketed` | bucket 16: six tokens; bucket 256: two tokens | 16, 256 (two launches) |
| `rf2_ranges`, physically sorted | rows 0–5 at width 16, rows 6–7 at width 256 | 16, 256 (two launches) |

### Why per-bucket launches
On a sorted batch, sorting by rank also sorts by bucket, so each bucket is already one contiguous
run of rows; the breakpoints come from counting tokens per bucket. `rf2_segment` sees nearly the
same groups, but runs them all in one kernel compiled with five width arms, choosing a width per
block at run time, with the block size set by the batch's width. `rf2_ranges` instead runs
`rf1_fused`'s split-K pass once per run, on the row views `x[start:end]`, `out[start:end]`, at
exactly that bucket's width and with that width's launch settings. Each bucket is a plain RF1 call
on its own rows: no index list, no rank read, no branch, no unused arms. On unsorted input a bucket
spans several runs, so the plan falls back to one launch at the batch's width, which is `rf1_fused`.

Each RF2 kernel has a `_fw` control that runs the same binary with every block forced to the
batch's width, so `kernel` vs `kernel_fw` is the effect of skipping alone.

## The shared kernel
RF1 and every split-K RF2 variant launch the same two kernels through `splitk_pass`
(`rankfuse/kernels/rankfuse_fused.py`), so these optimizations are identical across them:

| Optimization | What it does |
|---|---|
| Split-K shrink | h_in split into S slices (about one wave of SMs, at most 16, at least 2 above 128 tokens); fp32 partials in a workspace |
| Fused expand | Sum partials, rank mask, per-rank scale, cast and expand in one kernel; `v` stays in registers |
| Add into output | `add=True` adds the delta in the expand, with no separate pass over the output |
| Two expand modes | Up to 128 tokens: column loop, about 4 programs per SM. Above: one tile of up to 256 columns per program, 8 warps |
| Aligned stores | A `tl.multiple_of` hint vectorizes output stores (67 → 45 µs at 8192 tokens on the H200) |
| Block size rule | `_config(width, T)`: by default `_block_m` (16 tokens up to 128 tokens, else 64/32/16 by width, keeping `v` within 32 KiB of registers), replaced by the tuned table's entry on its GPU (see Per-width launch settings) |
| Shrink tiles | 128-wide h_in chunks, 3 pipeline stages, reduced only if shared memory runs out |
| Width rounding | Width rounded up to a power of two of at least 16 (`tl.dot` needs ≥ 16); extra rows load as zero |
| Fused slices | q/k/v or gate/up projections in one launch |
| Graph-safe | With the batch maximum rank passed in: no host syncs, workspace from the graph's memory pool |

## Where the variants differ
| | Width | Token access | Extra work per block | Launches |
|---|---|---|---|---|
| `rf1_fused` | Batch width, compiled | Batch order | None (the expand reads ranks only for the mask) | 1 pair |
| `rf2_segment` | Block's max rank, chosen at run time | Batch order | Shrink rank read, max, branch over up to 5 compiled widths; zero fill for rank-0 blocks in write mode | 1 pair |
| `rf2_index` | Same as segment | Through the sorted index, result written to each token's row | Same as segment, plus one index load per row; argsort once per step | 1 pair |
| `rf2_bucketed` | One compiled width per launch | Through each bucket's index | None at run time; CPU plan per step | 1 pair per bucket, on concurrent streams |
| `rf2_ranges` | One compiled width per launch | Batch order, one contiguous range per bucket | None at run time; CPU plan per step | 1 pair per bucket, on concurrent streams |

### Asymmetries to keep in mind
1. **Narrow blocks inherit the batch's block size (works against RF2).**
   A program handles BM tokens and keeps their `v [BM, width]` in fp32 registers. To keep that
   within 32 KiB, `_block_m` picks BM from the **batch's** width, once per call: 64 tokens at
   width 64, 32 at 128, 16 at 256. Width 16 could use 64 tokens (only 4 KiB). The tuned table
   changes those values on its GPU but is still looked up at the batch's width for one-launch
   kernels.

   With one r_max 256 request, the batch width is 256, so every block gets BM = 16, including the
   rank-16 blocks that make up most of the batch. Compared with an all-low batch (BM = 64), those
   blocks become four times as many programs with a quarter of the tokens each. Each program
   re-reads its rows of `A_t` and `B`, so the same work costs more reads and smaller matrix
   multiplies. The narrow blocks therefore run slower than RF1 on an all-low batch, which is the
   time used as the ideal bound, so RF2 can't fully reach that bound.

   This comes from a choice in the shared kernel, not something inherent. `_block_m` is a
   register-budget rule written for RF1, where every block has the same width; RF2 inherited it.
   Triton fixes BM per launch at compile time, so it can't change per block within one launch.
   `rf2_ranges` and `rf2_bucketed_tuned` avoid it by launching once per bucket.

2. **Unused width arms cost time (works against RF2).**
   RF2's kernel has one code path ("arm") per possible width (16, 32, 64, 128, 256), all compiled
   into a single kernel; each block runs only the arm for its width. At r_max 256 with every block
   at full width, adding the other four arms made the kernel about 6–7% slower even though they
   never ran (`rf2_probe_branch` → `rf2_segment_fw`).

   The likely reason, not yet confirmed: a compiled kernel reserves registers for its largest
   path, so fewer programs fit on each SM at once. That would also slow narrow blocks in an
   outlier batch, whose kernel is compiled with all five arms. That case is unmeasured, because
   `all_min` batches compile only the 16-row arm.

3. **RF1 was tuned at r_max 64 only (works for RF2), in the 5 Nov results.** There, going from
   r_max 128 to 256 cost RF1 2.7–3.1×, not 2×, so part of any r_max 256 win could be RF1 being
   slow. The A100 tuned table (below) now tunes every width, including RF1's, so those results
   predate the fix; every skew result is also compared with `b1_padded`.

## Per-width launch settings (tuning)
Block size, warps and tile settings are looked up per **(width, token count range)** in one table,
`rankfuse_fused._config`, filled from `rankfuse/kernels/tuned_a100.py` on the GPU it was tuned on
(`RANKFUSE_TUNED=0` turns it off for before/after runs). `bench/tune_blocks.py` builds it:

* **Inputs:** single-rank batches (every token at width W) at 256, 1024, 2048 and 8192 tokens,
  4096 × 4096 fp16, add mode, graph timing. These batches are tuning inputs only, never results.
* **Search:** BM (16–128) × shrink warps × expand tile × expand warps, then shrink chunk × pipeline
  stages at the best of those. Combinations holding `v [BM, W]` above 64 KiB of fp32 registers
  are skipped. Up to 128 tokens (decode, loop expand) keeps the default rules.
* **Choice:** per range, the per-size winners and the default are re-timed on every size in the
  range over 3 seeds; the best geometric mean replaces the default only if it is more than 2%
  faster (the repeat noise).

Every kernel uses the same table; only the width it looks up differs. `rf1_fused`, `rf2_segment`
and `rf2_index` run one launch, so they use the batch's width. `rf2_bucketed_tuned` and
`rf2_ranges` launch per bucket, so they use each bucket's width. `rf2_bucketed` uses the batch's
width, as the "RF1 settings" control.

**Parity rule:** RF1 gets every optimization that doesn't depend on running blocks at different
widths. Tuned block sizes, warps and tiles are shared through the table. Only per-bucket block
sizes, concurrent streams across buckets, the host bucket plan and the run-time rank read and
branch are RF2-only, because they only exist when a batch runs at more than one width.

## Two scheduler assumptions
| Assumption | What the scheduler provides | Kernel | Benchmark input | Cost reported separately |
|---|---|---|---|---|
| Physically sorted tokens | Requests ordered by rank when the batch is built | `rf2_segment` | `request_order: sorted` | Scheduler reordering (not measured) |
| Sorted index list | A rank-sorted list of token positions; tokens stay in batch order | `rf2_index` | `request_order: random` | argsort, once per step (`prepare_us`) |

The index list is the weaker assumption and closer to vLLM, which already builds
`token_indices_sorted_by_lora_ids` each step; rank would become a secondary sort key. Physically
sorting needs no data movement if done when the batch is built, but means changing how vLLM
orders requests in its batch. Whether vLLM can do either is unverified and left to future work.

## Why each test is needed
| Change or sweep | Why it's needed |
|---|---|
| `rf2_index` restored | Without it, results only cover the physically sorted assumption. `rf2_index` vs `rf2_segment` on sorted input isolates the index cost, the price of the weaker assumption |
| Paired flush timing | Small-batch repeats differed by up to 6.4%, about the size of the overhead being measured (6–8%). Timing the L2 flush right before each repetition cut that to 2.5% (`bench/flush_timing_check.py`), so small-batch overhead numbers are defensible |
| `zipf_1`, `zipf_2` mixes | Existing mixes are uniform-like or exact two-rank splits (`low_frac`). Traffic skewed to low ranks with a long tail of high ranks sits between them, and is where RF1 pays full width for little reason |
| `rf2_overhead.yaml` | States what RF2 costs when it can't skip (1–128 tokens and single-rank batches), so the report can say where RF2 doesn't help, with a number |
| `rf2_skew.yaml` | The search for wins: r_max 256, low-rank-heavy mixes, both orders and both assumptions, with all-low and all-high rows bounding each size |
| `b1_padded` in skew runs | Guards against asymmetry 3: a win over RF1 at r_max 256 counts only if it also holds against B1 |
| `rf2_bucketed_tuned` in skew runs | `rf2_bucketed_tuned` vs `rf2_bucketed` estimates what larger blocks for narrow buckets would gain (asymmetry 1) |
| Analysis tables (`analysis/rf2_eval.py`) | One table per question: overhead in µs, net gain over RF1 and B1 under each assumption, skipping alone, share of the ideal captured |

Share of requests per width bucket for the r_max 256 mixes (nine pow2 tiers, 1 to 256):

| Mix | Bucket 16 (ranks 1–16) | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| `skew_low` | 78% | 9% | 7% | 4% | 2% |
| `zipf_1` | 81% | 6% | 5% | 4% | 4% |
| `zipf_2` | 95% | 2% | 1% | 1% | 1% |
| `low_frac_90/99/100` | 90% / 99% / all but one | – | – | – | rest |

## Future work
* **Block size for narrow blocks (asymmetry 1).** `rf2_ranges` and `rf2_bucketed_tuned` fix it by
  launching per bucket with that width's settings. Still open for the one-launch kernels: pick BM
  for narrow blocks and let wider arms walk their tokens in smaller row chunks.
* **Device-side bucket ranges** for `rf2_ranges` under CUDA graphs in serving, where the bucket
  sizes change every step: offsets on the GPU, worst-case grids, early exit.
* **Compile only the widths a batch needs (asymmetry 2)**, if the skew results show narrow blocks
  paying for unused arms.
* **Scheduler support:** check whether vLLM can provide a rank-sorted index or rank-ordered requests.
