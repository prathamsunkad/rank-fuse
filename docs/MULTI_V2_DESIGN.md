# Multi-adapter v2: RF1-MA and RF2-MA on the shared split-K kernels

Status: design for Phase B on `pratham/newmultiadapter`. Nothing here is implemented yet.

## 1. Goal

Today the RankFuse code has two separate split-K kernel families:

* **Tony's shared pair** (`rankfuse/kernels/rankfuse_fused.py`, `_splitk_shrink_kernel` /
  `_splitk_expand_kernel`, launched through `splitk_pass`). Every single-adapter RF1 and RF2
  variant runs on it, with constexprs `INDEXED`, `RANK_SKIP`, `R`, `LOOP` and `RANK_PROBE`, and with
  launch settings from `_config` (the default rules plus `tuned_a100.py` on the A100 80GB PCIe).
* **Our multi-adapter pair** (`rankfuse/kernels/rankfuse_multi.py`, `_ma_shrink_kernel` /
  `_ma_expand_kernel`). It uses the same design but reads a tile table and a page table, and its
  launch rules are its own copies (`_split_k`, `_expand_geometry`).

Phase B makes the multi-adapter path one more mode of the shared pair. Then:

* single-adapter and multi-adapter results come from one kernel source;
* tuning (`_config`, `tuned_a100.py`) applies to both paths;
* the paper can say "RF1-MA is RF1 plus a tile table and a page table" and show it in code.

The Python API stays as it is: `rankfuse/multi.py` (`build_meta`, `MultiMeta`, `MultiMeta.load_`,
`PagedWeights`, `MULTI_REGISTRY`, `tile_capacity`), so the paging session's work plugs in unchanged.

## 2. What the multi mode adds to each kernel

A new constexpr `MULTI` (default `False`) is added to both kernels. With `MULTI=False`, every
existing path compiles to the same code as today. Section 6 gives the check.

| Need | Today in `_ma_*` | In the shared kernels with `MULTI=True` |
|---|---|---|
| Per-block tile entry | `tiles[i] = (adapter, start, length, limit, first)` int32 ×5 | Same table. Program axis 0 is the tile index, not `m * BM`. `pos = start + arange(BM)`, `row_ok = arange(BM) < length`. |
| Tokens through the sorted order | `_tile_tokens`: `first >= 0` means consecutive tokens (no `order` load), else `order[pos]` | Same helper. It replaces the `INDEXED` branch when `MULTI`. |
| Workspace row | `pos` (position in the sorted order) | Same. `ws_row0 = 0`. The workspace is still `[S, split_k, T, R_PAD]`. |
| Weight addressing | `_frame_rows`: `frame(a, k // 16) * FRAME + off[s] + (k % 16) * LD`, int32 when the pool is below 2^31 elements (`OFF64`), `multiple_of(16)` when aligned | Same helper. A_t / B base pointers become pool pointers. `A_SLICE` / `B_SLICE` are replaced by the per-slice `a_off` / `b_off` / `b_ld` from `W.slice_table()`. |
| Rank limit per group | `k_ok = k < limit`. `limit` is the tile's group limit, masked loads only. | `r_limit` becomes per tile: `limit = tiles[i, 3]`. Same masking. |
| `RANK_SKIP` per group | `BUCKETED`: branch on `limit` to R = 16/32/64/128/256 | `RANK_SKIP` with `MULTI`: branch on the tile's `limit` instead of `max(ranks)`. They are equal for `rank_sorted` tiles. For `group` tiles, `limit` is the group's maximum. `force_rank` still works as the full-width control. |
| Per-token rank mask | `v = where(k < ranks[t], v, 0)` | Unchanged. |
| Scale | `scale_table[a * (r_max + 1) + r_t]` | `HAS_SCALE` with `MULTI`: index by `adapter * SCALE_LD + ranks`. `SCALE_LD = 0` gives back the single-adapter table. |
| Separate A per q/k/v slice | grid axis 2 = slice, `a_off[s]` | Same. The single-adapter kernels already use axis 2 for slices. |
| Write-back to the original row | `out[tok]` | `out[tok]`. This already holds for `INDEXED`. |
| Rank-0 / no-LoRA tile | `limit == 0`: zero-fill when `add=False`, else skip | Same, through the existing `_splitk_zero_part`. |
| Empty tiles past the step's count | `length == 0`: exit | Same: early exit before any load. |
| Fixed capacity (one graph across steps) | grid axis 0 = `meta.capacity`. `MultiMeta.load_` copies a new step's tables in place. | Unchanged. Only the grid's first axis changes (capacity instead of `cdiv(T, BM)`). |

### Kernel structure

The shrink currently computes `tok`, `a_s`, `ws_k` and then dispatches on
`RANK_SKIP` / `RANK_PROBE`. The plan is to split it into:

1. a prologue `_splitk_block(...)` that returns
   `(tok, pos, row_ok, ranks?, limit, a_base, a_rows_fn args)`;
2. the existing width dispatch.

With `MULTI=False` the prologue is exactly today's code, inlined at compile time.
`_splitk_shrink_part` / `_splitk_expand_part` take a row-offset vector (`a_rows` / `b_rows`) in
place of `k * H_IN` / `k * N_MAX`. With `MULTI=False` the caller passes `k * H_IN`, so the generated
address arithmetic is the same expression.

**Risk:** if Triton produces different PTX for an equivalent expression, the single-adapter
timings could move. Mitigation: keep the `MULTI=False` path textually unchanged inside
`if MULTI: ... else: <old code>`. This duplicates some code, but bitwise equality and identical
timing are then guaranteed by construction. Check the generated PTX hashes (section 6).

## 3. Host side

* `multi_lora(x, W, token_ranks, meta, ...)` becomes a wrapper over a new
  `splitk_pass(..., multi=MultiArgs)`, or over a sibling `splitk_pass_multi` that shares the expand
  tile rules. The grid's axis 0 is `meta.capacity`. `grid_tiles` is `capacity * S` for the split
  rule.
* **Launch settings:** `_config(width = meta.r_pad, num_tokens = T)` gives bm, warps, bh, stages
  and bn, as in RF1. `meta.bm` must then come from `_config` too. `build_meta` takes `bm`, so
  `rf1_ma`'s prepare passes `_config(r_pad, T)["bm"]`. The tile capacity depends on bm
  (`tile_capacity(T, bm, N)`), so a captured graph remains valid as long as bm is fixed per
  (T, width).
* **Many-adapter split-K rule:** today `MULTI_SPLITK_*` lives in `rankfuse_multi.py` and applies at
  `h_in >= 2048`, `T <= 256`. Keep it only where it measures better than `_split_count` on the
  A100. It is an extra `split_k` override passed into the plan, so `_split_count` itself is
  untouched.
* **Fast launch:** the multi mode goes through the same `_SplitKPlan` / `_launch` caching (the plan
  key adds capacity, `aligned`, `off64`, the page-table width and the scale layout). The metadata
  tensors are arguments whose alignment bits enter the kernel key.
* **Registry:**
  * `rf1_ma` → the new path;
  * `rf1_ma_legacy` → today's `multi_lora` over `_ma_*`, unchanged, for comparison;
  * `rf2_ma_*` names stay.

  Anagha's `rf1_ma` / `rf1_ma_flat` in `rankfuse_ma.py` are not registered and are left as they are.

## 4. RF2-MA on Tony's strategies

Strategy (i): `RANK_SKIP` inside multi-adapter tiles.
* Tiles are groups of (adapter, width bucket): `build_meta(mode="rank_sorted")` already sorts by
  (adapter, rank) and cuts tiles at bucket boundaries. The `limit` of each tile is its own maximum.
  With `RANK_SKIP` and `MULTI`, each program branches to the bucket width. This is `rf2_ma_splitk`
  rebuilt on the shared kernels. `force_rank = r_pad` gives the full-width control
  (`rf2_ma_splitk_fw`), as `rf2_segment_fw` does.
* Variant over the batch order (`mode="segment"`): `rf2_ma_segment`.

Strategy (ii): per-width group launches on concurrent streams.
* As in `rf2_bucketed_tuned` / `rf2_ranges`: split tokens by width bucket, then group by adapter
  within each bucket. Each bucket gets one launch at a fixed `R` (no runtime branch) with that
  bucket's `_config(R, count)` settings. Buckets run on concurrent streams (fork/join, capturable).
  This is `rf2_ma_bucketed` rebuilt on the shared kernels, with per-bucket tuned settings.
* For graphs: per-bucket capacities (`BucketPlan(capacities=...)`) fix the grids. A graph is then
  valid for steps with the same bucket set. That limit is documented, as it is today.

The selector `rf_ma_auto` stays, refit on a training split of the A100 sweep, never on the
evaluation batches.

## 5. Validation plan

1. Every `tests/test_multi_*.py` test passes unchanged.
2. A new `tests/test_multi_v2.py` checks the new path:
   * against `b0_multi` (fp32 reference): dense and paged layouts, shuffled frames, merged
     q/k/v (896, 128, 128) and gate|up, per-adapter scales, int32 and int64 offsets (a pool above
     2^31 elements, or a forced `OFF64`), rank-0 tokens, adapter −1, empty tiles past the step's
     count, T = 1..8192, 1..256 adapters, r_max 64 / 128 / 256, fp16 / bf16 / fp32;
   * CUDA-graph capture with `capacity`, then `MultiMeta.load_` of new steps and replay;
   * `add=True` / `False`, and `out` as a column view.
3. With `RANKFUSE_TUNED=0` (so both paths use the default rules), the new `rf1_ma` against
   `rf1_ma_legacy`: report bitwise equality or the maximum difference per case. Bitwise equality
   is expected when bm, bh, split_k and the expand tile agree, since the reduction order is the
   same. Where the legacy many-adapter split rule differs from `_split_count`, the split-K
   partial sums differ in order, so report the maximum difference.
4. Single-adapter RF1 / RF2 with `MULTI=False`:
   * bitwise equality against origin/main's code (the merged branch now has
     `tests/test_launch_plan_vs_splitk_pass.py`, which compares against origin/main's
     `_splitk` + `splitk_pass`; add a pinned copy of the kernels at 8264f8f for this check);
   * H200 timings within noise on the Stage 1 grid (`bench/run_sweep.py`, graph replay, same
     seeds);
   * identical PTX for the `MULTI=False` specializations, by comparing the Triton cache
     artifacts.
5. RF2-MA (i) and (ii): correctness against `b0_multi` on the same grid as item 2, plus the
   full-width control against `rf1_ma`.

## 6. Timing plan (A100 80GB PCIe, final numbers)

* **GPU:** `-p ice-gpu --gres=gpu:a100:1 -C A100-80GB`. Coordinate with the A100 measurement
  session; develop on the H200 / H100.
* **Implementations:** new `rf1_ma`, the RF2-MA variants (i) and (ii), `rf1_ma_legacy`, `b3_multi`,
  `b4_slora_multi` copies and shared.
* **Workloads:**
  * 1 to 256 adapters;
  * popularity: uniform, and Zipf with α 1.0 and 1.2;
  * rank mixes: uniform, skew_low, all_max, and the low_frac mixes;
  * decode at 1–256 tokens, prefill at 2048 and 8192;
  * r_max 64, 128 and 256.
* **Shapes:**
  * 4096²;
  * the four Qwen2.5-0.5B projections;
  * Qwen2.5-7B: 3584→3584, 3584→512, 3584→18944, 18944→3584.
* **Method:** CUDA-graph replay with metadata built outside timing, an L2 flush, and the median of
  7 reps × 20 replays with p20/p80. 3 seeds. Nsight Compute byte counts for a few cases.
* **Output:** `results/a100_<date>_multi_v2/` with a README giving the exact GPU name string,
  torch / Triton / CUDA versions, the commit and a diff hash, seeds, and the timing method.
* **Run size:** the grid is large (about 2–3k batches × ~7 impls × 3 seeds). Run it as one sbatch
  with `--resume`, split by shape so a 16 h job limit is not an issue.

## 7. Open questions for the main session

* Should `rf1_ma` move to `_config`'s block size, which changes `tile_capacity` (and so the graph
  shapes the paging session captures)? Proposed: yes, `build_meta`'s default bm becomes
  `_config(r_pad, T)["bm"]` when no bm is passed.
* Should the many-adapter split-K rule stay if the A100 shows no gain? Proposed: keep it only for
  the (shape, T) ranges where it wins, as an entry next to `tuned_a100.py`.
