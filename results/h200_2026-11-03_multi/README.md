# RankFuse Stage 2 results: many Matryoshka adapters (3 Nov 2026)

Stage 2 = many Matryoshka adapters in one batch; every request carries (adapter, rank). This folder
holds the measurements of the multi-adapter kernels and baselines on one H200.

* `ALL_TABLES.md`: every table. `tables/`: the same as CSV and Markdown.
* `raw/multi-main.jsonl` (main sweep, 8610 rows) and `raw/multi-pages.jsonl` (page-layout sweep, 630
  rows): one JSON object per timed (implementation, batch). `raw/superseded/`: two earlier runs of
  the same sweeps with the previous split-K rule (see "Tuning"), kept for the record, not used in tables.
* `source/ncu/*.csv`: Nsight Compute per-kernel metrics (`source/ncu/superseded/`: RF1-MA before
  the split-K change). `source/logs/`: test, sweep, ncu and tuning logs.
* `build_tables.py`: rebuilds `tables/` and `ALL_TABLES.md` from `raw/` and `source/ncu/`
  (`python3 build_tables.py`, plain Python). `scripts/fit_selector.py` rebuilds table 10.
* `scripts/ncu_multi.sh`, `scripts/ncu_all.sh`: the ncu runs.

Code (branch `pratham/rf1-splitk`, uncommitted): `rankfuse/multi.py` (interface, page tables,
metadata, workloads, registry), `rankfuse/kernels/rankfuse_multi.py` (RF1-MA, RF2-multi, selector),
`rankfuse/baselines/*_multi.py` (B0, B1, B3, B4), `bench/run_multi_sweep.py`, `bench/configs/multi_*.yaml`.

## Setup

| | |
|---|---|
| GPU | 1x NVIDIA H200 (132 SMs, 141 GB HBM3e, 4.8 TB/s), PACE ICE, driver 615.71.09 |
| Software | PyTorch 2.13.0 (CUDA 13.0), Triton 3.7.1, vLLM 0.32.0, Nsight Compute 2026.3.0 |
| Precision | fp16 for every timing; tests also bf16 (and fp32 for the RankFuse kernels and B1) |
| Shapes | 4096 x 4096 test layer; Qwen2.5-0.5B projections q/o 896->896, k/v 896->128, gate/up 896->4864, down 4864->896 (one projection per call) |
| Adapters | 1, 4, 16, 64, 256 Matryoshka adapters, r_max 64 each, tiers 1, 2, 4, ..., 64; all resident |
| Popularity | adapter of each request uniform, or Zipf with alpha 1.0 over adapter ids |
| Rank mixes | per request over the 7 tiers: uniform, skew_low (0.25/0.21/.../0.04), all_max (all 64) |
| Tokens | decode: 1, 8, 32, 128, 256 requests of 1 token; prefill: 16 x 128 = 2048 and 32 x 256 = 8192 tokens (whole prompts; at most 16 / 32 adapters active) |
| Batch order | arrival order (not sorted by adapter); a prompt's tokens are consecutive |
| Weights | RankFuse kernels read a shuffled per-layer page pool (16-row pages, frames permuted); baselines read their own layouts |

**Timing.** GPU time only, by CUDA-graph replay, as `bench/run_sweep.py --timing graph`: per-step
metadata (each implementation's `prepare`) is built outside the graph, the call is captured once,
7 repetitions of 20 back-to-back replays, median reported; a 256 MB L2 flush inside the graph keeps
weights cold and its own time is subtracted. Every call **adds into a preallocated output**
(`add=True`), as a LoRA layer adds its delta to the base projection (no implementation pays an
extra zeroing pass). One active adapter's weights are read once in every implementation's best case.

**Implementations.**

| Name | What it is |
|---|---|
| B0-multi `b0_multi` | fp32 reference, one matmul pair per distinct (adapter, rank) pair (correctness only) |
| B1-multi `b1_multi` | PyTorch: per adapter group, pad to the group's maximum rank, mask, two cuBLAS matmuls, index_add (about 7 launches per group) |
| B3-multi `b3_multi` | stock vLLM: every (adapter, rank) pair present is its own adapter, padded to r_max, in vLLM 0.32's `_lora_shrink` / `_lora_expand` (LoRAKernelMeta sorts tokens by slot); only the pairs present are built (vLLM's best case: no adapter loads) |
| B4-multi copies `b4_multi_copies` | S-LoRA's bgmv kernel, every (adapter, rank) pair its own adapter at its true rank (rounded to 8) |
| B4-multi shared `b4_multi_shared` | S-LoRA's bgmv with one shared copy per adapter (entry (a, r) shrinks at r, expands at 64) |
| RF1-MA `rf1_ma` | RankFuse: tokens grouped by adapter, split-K shrink + masked expand over a tile table, every tile at its group's rank limit, weights through the page table |
| RF2-multi `rf2_ma_splitk` / `rf2_ma_segment` / `rf2_ma_bucketed` | RF1-MA plus per-tile rank buckets: sorted by (adapter, rank) / batch order with runs of one adapter / a CPU plan with one kernel pair per rank bucket on concurrent streams |
| `rf_ma_auto` | the selector: rf2_ma_bucketed or rf1_ma per step (table 10) |
| `stage1_rf1_fused` | Stage 1's `rf1_fused` on adapter 0, plain pointers, no page table (runs at 1 adapter only) |

## Design

* **Interface** (`rankfuse/multi.py`). A separate `MULTI_REGISTRY` (`impl(x, W, token_adapter,
  token_ranks, out=, add=, **prepared)`), so Stage 1's `REGISTRY` and `tests/test_correctness.py`
  are untouched. `PagedWeights` = (A pool, B pool, `page_table [num_adapters, r_max/16]`, frame
  strides, per-slice offsets and B row strides, optional per-adapter per-rank scale table, merged
  slice sizes). Rank row j of adapter a, slice s is at `frame(a, j//16) * frame_stride + off[s] +
  (j%16) * ld`: the whole-model page layout (one frame holds every matrix, fixed offsets) and simple
  per-layer pools `[F, S, 16, h]` are both this description. `PagedWeights.dense` wraps stacked
  `[N, S, r_max, h]` tensors with an identity page table; `PagedWeights.paged` copies pages into
  pools (contiguous or shuffled frames, optionally only some pages resident, page_table -1 elsewhere).
* **Per-step metadata** (`build_meta`, host-side, once per step, outside the call): token order
  sorted by adapter (stable, as vLLM's LoRAKernelMeta), group start/length/adapter/rank limit, and a
  tile table `(adapter, first position, length, rank limit, first token)` of at most BM tokens per
  tile, each inside one group. Rank-0 / adapter -1 tokens form a last group of limit 0 (their rows
  are zeroed, or left alone with add=True). With `capacity=tile_capacity(...)` and a fixed `r_pad`
  the grid is fixed, and `meta.load_(new)` refreshes it in place, so **one captured CUDA graph
  replays later steps with other adapters and ranks** (tested).
* **RF1-MA**: rf1_fused's split-K design over the tile table. **Per-group rank limit via masking
  under a constexpr R_PAD**: the register tile v is `[BM, R_PAD]` (R_PAD = bucket of the batch's
  maximum rank, or the caller's r_max for graphs); each tile's runtime limit masks the A and B row
  loads, and masked loads issue no memory request, so a group reads exactly its limit's rows (ncu:
  16.0 MB read where the ideal is 16.8 MB, table 09). The MMA always runs R_PAD wide, which is cheap
  here, and one compiled variant serves every rank mix. Rejected: a runtime loop over 16-row chunks
  (Triton cannot index a register tile by a runtime chunk; it would need per-chunk accumulators or
  re-reading x per chunk); bucketed compiled variants are what the RF2 variants test.
* **Page table**: each program loads the frame of every rank row it may read once, before its
  h_in (shrink) or column (expand) loop, from an L1-resident table of r_max/16 int32 per adapter;
  nothing is added per weight element. Offsets are int32 unless a pool has >= 2^31 elements (int64
  cost ~1 us at decode), and alignment hints keep the gathered row loads vectorized.
* **Graph safety**: no host reads in the call; fixed max grid (tile capacity), empty tiles exit at
  once; ws comes from the graph pool during capture.
* **Two small latency fixes found while measuring** (against Stage 1's rf1_fused at one adapter):
  int32 offsets (above) and a "tile tokens are consecutive" column that lets prompt tiles skip the
  dependent gather through the token order. Together they took the one-adapter gap from 10-25% to
  the 4-11% in table 07.
* **Tuning**: with tens to hundreds of tiles (many-adapter decode) rf1_fused's split-K rule (1
  program per SM) leaves one long serial h_in loop per program. For h_in >= 2048 and <= 256 tokens
  the shrink now aims at 4 programs per SM (at most 8 slices), never fewer than rf1_fused's rule:
  e.g. 4096 x 4096, 128 tokens over 256 adapters 38 -> 29 us; down projection 31 -> 19 us
  (`source/logs/tune_splitk.out`, the same H200; a first version applied to prompts too and cost
  ~8% there, so it is limited to <= 256 tokens). Fitted on one experiment; the sweeps were then rerun.
* **RF2 variants**: rf2_ma_splitk sorts by (adapter, rank) so tile limits are tight and branches to a
  per-bucket compiled tile; rf2_ma_segment keeps the batch order (tiles = runs of one adapter); 
  rf2_ma_bucketed (CPU `BucketPlan`) splits tokens by bucket (16/32/64), groups each bucket by adapter,
  one kernel pair per bucket compiled for exactly R rows, buckets on concurrent streams (fork/join,
  capturable). **Its graph is only valid for steps with the same bucket set and tile counts**: the
  number of launches and their grids are baked in (BucketPlan takes per-bucket capacities to fix them).

## Headline results

All speedups are time(other) / time(RF1-MA) on the same batch; ranges over popularity x rank mix.

| Question | Result | Table |
|---|---|---|
| RF1-MA vs vLLM's way (B3-multi) | faster on **all 1050** batches: geometric mean **2.04x** at decode (1.04-7.5x), **2.84x** on prompts (1.11-12.3x). Grows with adapters: 4096 x 4096 at 32-256 decode tokens 1.4-2.0x with 1 adapter, 2.9-4.2x with 64-256 | 02 |
| ... because of bytes | 16 adapters, 128 tokens, 4096 x 4096: RF1-MA reads **16.0 MB** from DRAM (ideal ~16.8 MB), B3-multi **79.8 MB** (74 (adapter, rank) pairs, each a padded 64-row copy), B4 copies 26.8 MB, B4 shared 17.4 MB | 09 |
| RF1-MA vs S-LoRA bgmv (B4-multi) at decode | geometric mean **1.50x** (copies) / **1.53x** (shared) over 750 decode batches; **1.10x** at 1-8 tokens (0.90-1.35x; 24 of the 25 batches below 0.95x are here), **1.21x** at 32 tokens (0.93-1.87x), **2.3x** at 128-256 tokens (1.08-6.0x) | 03, 04 |
| ... on prompts | 13-29x (bgmv has no weight reuse across tokens; S-LoRA uses a different kernel for prompts, so this is bgmv itself, not S-LoRA's prompt path) | 03, 04 |
| RF1-MA vs PyTorch per group (B1-multi) | 1.8-249x; 10.8x geometric mean at decode (B1 launches ~7 kernels per adapter group) | 05 |
| Page-table cost | shuffled vs dense (identity table): **1.00x** at decode (0.90-1.08) and prefill (0.96-1.03); contiguous vs dense 0.99-1.00x | 07 |
| Multi-adapter machinery vs Stage 1 at 1 adapter | RF1-MA is 1.11x slower at decode (1.02-1.29x; ~1 us, mostly the tile-table load ahead of the first x load), 1.04x at prefill | 07 |
| Few tokens per adapter (the risk) | still faster than both B4 modes at 32-256 tokens: e.g. 4096 x 4096, 128 tokens over 256 adapters (96 active, 1.3 tokens each): 32 us, 1.38x B4 copies, 1.66x B4 shared, 3.0x B3; but only **13-38% of the byte roofline** on 4096 x 4096 and 2-21% on the 896-input layers (prompts: 37-62% on 4096 x 4096); small decode batches are latency-bound with any number of adapters | 08 |
| RF2-multi vs RF1-MA | no variant pays on average (per shape and phase, geometric means): splitk 0.97-1.02x, segment 0.81-1.02x (0.31x worst: decode, where runs of one adapter are ~1 token), bucketed 0.96-1.01x; best single batches +33% (bucketed, 128 decode tokens, 256 adapters, Zipf, skew_low), +20% (splitk, decode), +11% (segment, prompts) | 06 |
| Selector (fitted on the same data) | rf2_ma_bucketed when its rank work <= 70% of RF1-MA's and h_in >= 2048: **+0.4%** over always RF1-MA (oracle +2.2%), worst pick 0.76x. Recommendation: RF1-MA alone | 10 |
| Correctness | full suite on the H200: **8389 passed, 54 skipped, 0 failed**; new files 6798 passed (24 skipped: B4 with merged slices); the pre-existing files unchanged at 1531 passed / 29 skipped (+60 / +1 from the parallel session's new vLLM test files) | below, `source/logs/full_suite_per_file.txt` |

### Correctness

`tests/test_multi_correctness.py` and `tests/test_multi_graphs.py` (new files only): every
multi-adapter implementation vs B0-multi in fp16 and bf16 on 4096 x 4096 and the four Qwen shapes,
1 / 5 / 64 adapters (64 with Zipf popularity), decode T = 1, 17, 64, 128 and prompts 3 x 50,
2 x 300, rank mixes uniform / skew_low / all_max / 30% rank 0; shuffled and contiguous page pools,
partially resident pages (page_table -1, spare frames), add mode (rank-0 rows untouched exactly),
overwrite mode (rank-0 rows exactly 0), merged slices (q/k/v 896/128/128 and gate/up, with scales),
per-adapter scale tables, odd ranks and adapter -1, all rank 0, fp32, unaligned sizes (200 -> 100),
8192 tokens over 64 adapters; metadata invariants; CUDA-graph capture and replay of all 9
implementations, and one graph replayed over new steps after `meta.load_`.

## Caveats

* One GPU, fp16 timings, r_max 64, one projection per call (no merged-slice timings, no whole model,
  no vLLM integration; Stage 1's vLLM work is in the other session).
* All adapters are resident: no adapter loading, eviction or page faults are measured. B3-multi is
  given a slot for every (adapter, rank) pair present, its best case (real vLLM would also need
  max_loras >= distinct pairs, i.e. about 7x the slots per adapter, or pay CPU->GPU loads).
* B4 uses S-LoRA's decode kernel everywhere; its prompt rows are not S-LoRA's prompt kernel.
* Batch composition is synthetic (seeded); prompts are 16 x 128 and 32 x 256 tokens, so at most 16 /
  32 adapters are active there whatever the adapter count. Mixed prefill + decode batches are not
  timed (correctness covers any token pattern).
* The split-K rule for many-adapter decode and the selector were fitted on the same H200 data they
  are shown on. The selector's gain is within noise; it is kept to quantify the RF2 opportunity.
* ncu times are with caches flushed and clocks locked per kernel: compare bytes, not time. DRAM
  writes show as ~0 because the small outputs stay in L2 until after the kernel ends.
* RF1-MA's many-adapter decode is far from the byte roofline (13-38% on 4096 x 4096): with ~1 token per adapter
  every tile is mostly padding and each program streams one adapter's rows; a decode-specialised
  path (several adapters per program, or a GEMV-style kernel for 1-token groups) is the obvious next
  step and is not done.
* Results from `raw/superseded/` are not used; tables come from the final code only (the
  RF1-MA-specific ncu CSVs were re-profiled after the split-K change; B3/B4 ncu runs are unchanged code).
