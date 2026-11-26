# Prefix paging on the A100 80GB PCIe: copy timings and residency-policy simulation

Date: 2026-11-23/24. The code is uncommitted work in the `wt/paging` worktree, based on `31adb60`
(`pratham/rf1-splitk`) plus Anagha's `rankfuse/paging/` from `origin/main`.

## Contents

| path | what |
|---|---|
| `source/h2d.{json,csv}` | Pinned H2D copy timings: a size sweep, real page and adapter sizes, vLLM-style many-copy loads, and loads through `ModelPageManager` (job 7263530). |
| `source/overlap.{json,csv}`, `source/ready.{json,csv}` | Overlap and time-to-ready from the first run (job 7263530). Superseded by the rerun. |
| `source/rerun_overlap_ready/` | Rerun of overlap and time-to-ready (job 7284990), after two measurement fixes (below). The tables use this. |
| `source/env.json` | Device, driver, torch/triton/CUDA versions, host, git HEAD. |
| `source/sim_results{,_rate5,_rate80}.{json,csv}` | The full simulator grid: 6912 rows per arrival rate. |
| `tables/h2d.md`, `tables/overlap.md`, `tables/ready.md` | GPU measurements. |
| `tables/hit_rate.md`, `gb_copied.md`, `resident.md`, `wait.md` (+ `_rate5`, `_rate80`) | Simulator headline tables: 1000 adapters, Zipf 1.0, uniform tiers. |
| `tables/summary*.md` | `rf_prefix` against each baseline over the whole grid, and the smallest budget reaching 50% / 80% hit rate. |
| `figures/*.png` | Hit rate, GB copied, mean wait and adapters resident against budget (6 panels: model x r_max), for uniform and skew_low tiers. |

## Conditions

**GPU runs.**
* Hardware: ICE `atl1-1-01-005-13-0`, NVIDIA A100 80GB PCIe, driver 615.71.09, PCIe Gen4 x16,
  SM clock max 1410 MHz.
* Jobs: 7263530 (all three experiments plus `tests/test_paging_bridge.py` and
  `tests/test_model_pages.py`: 35 passed) and 7284990 (overlap and ready rerun). Both ran on
  partition coc-gpu with `--gres=gpu:a100:1 -C A100-80GB`, 8 CPUs, 96 GB host RAM.
* Software: torch 2.13.0+cu130, Triton 3.7.1, CUDA 13.0, fp16 weights.
* Timing: CUDA events, median (and p95) after warm-up. Host buffers are pinned with
  `cudaHostAlloc` via torch.
* Command:
  `python bench/paging_timing.py --out results/a100_2026-11-23_paging/source[/rerun_overlap_ready] [--only overlap ready]`.
* Tables: `python3 bench/paging_report.py --src ... --tables ...`.

**Page sizes (fp16, whole model: all layers, q/k/v/o/gate/up/down, A and B).**
* Qwen2.5-0.5B: 1,099,776 B per rank row; one 16-row page is 17.6 MB; a rank-64 adapter is 70.4 MB.
* Qwen2.5-7B: 5,046,272 B per rank row; one page is 80.7 MB; a rank-64 adapter is 323 MB, rank 256
  is 1.29 GB.

**Simulator.**
* Command: `python bench/paging_sim_sweep.py --out results/a100_2026-11-23_paging --bw 26.28 --call-us 2.04 [--rate 5|80 --tag _rate5|_rate80]`.
  It runs on the login node and is CPU-only.
* Copy cost: `t = calls * 2.04 us + bytes / 26.28 GB/s`, fitted from `source/h2d.json`.
  * The bandwidth is the slope of the one-copy size sweep for sizes >= 64 KiB.
  * The per-call cost is the median extra time of the vLLM-style per-module copies over the same
    bytes in one copy.
* Requests and trace:
  * Each request has an adapter (Zipf over N adapters, ids shuffled) and a rank tier drawn from
    a mix over the tiers 1, 2, 4, .., r_max.
  * Mixes: `uniform`; `skew_low`/`skew_high` (linear weights toward low/high tiers); `all_max`.
  * Arrivals are Poisson at 20 req/s (main), 5 and 80 req/s (sensitivity).
  * The trace has 30,000 requests. The first 10,000 only warm the cache, and metrics count the
    other 20,000. Seed 0, one trace per (N, alpha, mix, r_max).
* Grid: models {0.5B, 7B} x r_max {64, 128, 256} x budget {0.5, 1, 2, 4, 8, 12, 16, 20} GB x
  N {100, 300, 1000} x alpha {0.8, 1.0, 1.2} x 4 mixes x 4 policies.
* Policies (`rankfuse/paging/policy_sim.py`; all get the same byte budget):
  * `vllm`: stock vLLM with each Matryoshka tier registered as its own LoRA. Slots are padded to
    `max_lora_rank = r_max`, with one slot per (adapter, tier) and whole-slot LRU. A load copies
    the tier's rows as 2 x 7 x layers copies.
  * `slora`: S-LoRA-style. Each (adapter, tier) is stored at its true rank, with row granularity
    and no padding, and whole-adapter LRU by bytes. A load is 2 x layers copies. S-LoRA's sharing
    with the KV cache is not modelled.
  * `rf_whole`: RankFuse with one copy per adapter at r_max that serves every tier, and whole-slot
    LRU. A load is r_max/16 page copies.
  * `rf_prefix`: RankFuse prefix paging with 16-row whole-model pages. A request loads only its
    missing pages, and eviction takes the top page of the least-recently-used other adapter.
* Wait model: one FIFO copy engine. A request waits until its pages or item are loaded,
  including loads queued before it and in-flight loads it hits on.
* Not modelled:
  * request duration, so an item can be evicted while a running request uses it (LRU makes that
    rare);
  * the host tier (every adapter is assumed to be in pinned host memory);
  * compute.

## Results

### 1. Copy time and bandwidth (`tables/h2d.md`)

The fit gives **t = 14.0 us + n / 26.28 GB/s**. Copies of 16 MB and more reach 25.7–26.3 GB/s.

| | 0.5B | 7B |
|---|---:|---:|
| 1 page | 0.68 ms (17.6 MB) | 3.09 ms (80.7 MB) |
| adapter r64, one copy | 2.69 ms | 12.30 ms |
| adapter r128, one copy | 5.37 ms | 24.59 ms |
| adapter r256, one copy | 10.73 ms | 49.18 ms |
| adapter r64, `ModelPageManager.ensure` (4 page copies into scattered frames + page-table copy) | 2.79 ms | 12.39 ms |
| adapter r256 via `ensure` (16 page copies) | 11.02 ms | 49.44 ms |
| adapter r64, vLLM-style (336 / 392 copies) | 3.96 ms | 13.10 ms |

* Loading through the manager costs 1–4% more than one contiguous copy.
* The vLLM-style many-copy load pays about 2 us per extra copy (+1.3 ms for a 0.5B r64 adapter,
  +0.8 ms for a 7B one). This is a lower bound for real vLLM, whose `set_lora` also zeroes the slot
  and runs a Python loop per module.

### 2. Does a concurrent copy slow RF1-MA? (`tables/overlap.md`)

The setup:
* One CUDA graph runs RF1-MA on every (layer, q|k|v / o / gate|up / down) of the model: 192
  launches for 0.5B, 224 for 7B.
* Ranks are 16/32/64/128, and every adapter is resident at r 128 in the whole-model pool.
* "With copy": a second stream streams whole-model pages over the whole timed window.

| model | tokens | step alone | step with copy | slowdown |
|---|---:|---:|---:|---:|
| 0.5B | 64 | 2.874 ms | 2.967 ms | 1.032 |
| 0.5B | 256 | 4.192 ms | 4.291 ms | 1.024 |
| 7B | 16 | 3.604 ms | 3.663 ms | 1.016 |
| 7B | 64 | 7.376 ms | 7.590 ms | 1.029 |
| 7B | 256 | 10.627 ms | 10.848 ms | 1.021 |

* **A page copy slows the RF1-MA kernels by 1.6–3.2%**, and the copy keeps its full 26.3 GB/s while
  the kernels run. Loads can run behind decode steps at almost no cost.
* The 0.5B, 16-token row is excluded. In both runs the copy chain had finished before the timed
  steps did ("copy covered window: False"), so it did not measure overlap. The rerun used 8-page
  copies (about 120 queued calls instead of about 1000), which ruled out a full stream queue as the
  cause. The cause is not known.

### 3. Time until a rank-r request on a cold adapter can run (`tables/ready.md`, r_max 256)

| rank | 0.5B prefix | 0.5B whole | 7B prefix | 7B whole | 7B speed-up |
|---:|---:|---:|---:|---:|---:|
| 16 | 1.64 ms (1 page) | 14.3 ms* | 3.14 ms | 49.4 ms | 15.8x |
| 32 | 1.42 ms | 10.98 ms | 6.22 ms | 49.4 ms | 7.9x |
| 64 | 2.79 ms | 10.97 ms | 12.40 ms | 49.4 ms | 4.0x |
| 128 | 5.52 ms | 10.98 ms | 24.72 ms | 49.4 ms | 2.0x |
| 256 | 10.98 ms | 10.97 ms | 49.38 ms | 49.4 ms | 1.0x |

* Time to ready scales with the pages needed, ceil(r/16), not the adapter.
* `ensure()` costs 12 us of host time per step when everything is resident.
* \*Anomaly: the first two 0.5B configs measured (r16 prefix and r16 whole) ran about 2x slower
  than the copy rate in both runs (expected 0.68 / 11.0 ms), although the rerun had 3 warm-up
  cycles. Every later row matches the h2d rate within 2%.

### 4. Policy simulation (`tables/summary.md`, `figures/hit_rate_uniform.png`)

Over every config of the grid where both policies fit (1620 per pair, 20 req/s):

| prefix vs | prefix hit rate >= baseline | median hit-rate gain | max gain | median misses ratio | median bytes ratio | median wait ratio |
|---|---:|---:|---:|---:|---:|---:|
| stock vLLM | 1620 / 1620 | +31.5 pts | +74.5 | 1.54x | 1.18x | 1.62x |
| S-LoRA-style | 1620 / 1620 | +15.8 pts | +37.4 | 1.34x | 1.05x | 1.13x |
| RankFuse whole | 1588 / 1620 | +7.0 pts | +37.7 | 1.16x | 2.99x | 3.19x |

The headline slice is 1000 adapters, Zipf 1.0 and uniform tiers (`tables/hit_rate.md`).
* Hit rates at 4 GB:
  * 0.5B r64: 17.6 / 36.6 / 47.7 / **56.1%** (vllm / slora / rf_whole / rf_prefix).
  * 7B r64: 4.8 / 14.5 / 23.2 / **32.1%**.
  * 7B r256 at 8 GB: 2.0 / 8.0 / 14.3 / **26.0%**.
* Smallest budget for a 50% hit rate at 1000 adapters (`summary.md`):
  * 0.5B r64, Zipf 1.0: prefix 4 GB, whole 8 GB, S-LoRA 12 GB, vLLM >20 GB.
  * 7B r64, Zipf 1.2: prefix 4 GB, whole 8 GB, S-LoRA 12 GB.
* Adapters resident (`resident.md`), 7B r64 at 8 GB: prefix 53, S-LoRA 64, whole 24, vLLM 20.
  S-LoRA keeps more items, but they are (adapter, tier) copies and a hit needs the exact tier.
* Mean wait per request, 7B r256 (`wait.md`): rf_whole 88–573 ms against 10–17 ms for the others.
  rf_whole loads 1.29 GB per miss, which saturates the copy engine at 20 req/s.

Caveats:
* **Page granularity costs bytes on tiny ranks.** A rank-1 to rank-8 request still loads one
  16-row page, while S-LoRA-style loads exactly r rows. With uniform tiers, 4 of the 7 tiers for
  r_max 64 are at most 8.
  * For 7B r64 at budgets up to 12 GB, S-LoRA-style therefore copies fewer bytes than prefix.
    At 0.5 GB it is 1798 GB against 2350 GB; at 12 GB, 1293 GB against 1322 GB. Its mean wait is
    also lower or equal: 3.8 ms against 4.8 ms at 0.5 GB.
  * Prefix still has 1.6–3.4x its hit rate there.
  * At 16 and 20 GB, prefix copies the fewest bytes of all policies.
  * A finer page (8 rows) or rank-aware eviction would close this gap. The simulator can test
    both.
* **Growth misses.** In 32 of 1620 configs, rf_whole hits slightly more often than prefix, by at
  most 0.17 points. All 32 have N=100 and no memory pressure. Prefix loads a low-rank prefix
  first, then misses once when the same adapter later asks for a higher rank.
* **Saturation.** Waits are not steady state when the copy engine is saturated (`copy_util` > 1 in
  the CSV, e.g. rf_whole for 7B r256). They then grow with trace length, and should be read as
  "this policy cannot keep up".
* **Rate sensitivity.** Hit rate and bytes do not depend on the arrival rate. Waits at 5 and 80
  req/s are in `tables/wait_rate5.md` / `wait_rate80.md`. At 80 req/s for 7B r64 at 0.5 GB:
  prefix 6.5 ms, vLLM 6.5 ms, S-LoRA 5.2 ms, rf_whole 143 ms.

## Earlier dev check (not results)

An H100 80GB HBM3 run with `--quick` (job 7263531; output in
`~/scratch/rankfuse/scratch_vllm/paging/h100_quick/`) measured about 55 GB/s pinned H2D and a
1–2% kernel slowdown under copy. It also found a bug: the copy buffer was 1 GiB and clipped 7B
r256 copies. That was fixed before the A100 run.

## Fixes between the two A100 runs (`bench/paging_timing.py`)

* overlap: the copy stream now issues 8-page copies instead of 1-page copies, so the stream queue
  stays short.
* ready: 3 full load/evict warm-up cycles, 9 repeats instead of 5, and the minimum is also
  recorded.
