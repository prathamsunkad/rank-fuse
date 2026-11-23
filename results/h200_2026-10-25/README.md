# RankFuse results (as of 25 Oct 2026)

Everything measured so far that can go into the presentation and the report, in one place.

* `ALL_TABLES.md`: every table, with what was measured and where it came from.
* `tables/`: the same tables as CSV (for plots and slides) and as Markdown.
* `source/`: the raw measurements the tables are built from (JSON, ncu CSVs, job logs).
* `figures/`: the results figures (CUDA-graph timings only), PNG for slides and PDF for the report.
* `build_tables.py`: rebuilds `tables/` and `ALL_TABLES.md` from `source/` (`python3 build_tables.py`);
  `make_figures.py` redraws `figures/` from `tables/` (needs matplotlib).
* `scripts/b2_grouped_runs.py`: the B2 timing runs (the other experiments' scripts live outside the repo).

How the kernels work: [docs/ALGORITHMS.md](../../docs/ALGORITHMS.md). Code: this repository, branch
`pratham/rf1-splitk`; the prototypes behind tables 03-09 live outside the repo (`~/scratch/rankfuse/checks`
on PACE) and were ported into `rankfuse/kernels/` and `rankfuse/baselines/vllm_ops.py`.

## Setup

| | |
|---|---|
| GPU | 1x NVIDIA H200 (132 SMs, 141 GB HBM3e at 4.8 TB/s, ~50 MB L2), PACE ICE |
| Software | PyTorch 2.13.0 (CUDA 13.0), Triton 3.7.1, vLLM 0.32.0, Nsight Compute 2026.3 |
| Precision | fp16 throughout |
| One-layer tests | a 4096 x 4096 projection (Llama-3-8B hidden size), one Matryoshka adapter with r_max 64, rank tiers 1, 2, 4, 8, 16, 32, 64 |
| Whole-model tests | Qwen2.5-0.5B-Instruct, all 168 projections (q, k, v, o, gate, up, down x 24 layers) carrying one generated rank-64 Matryoshka adapter |
| Rank mixes | uniform over the 7 tiers; skew_low (mostly low ranks); all_max (every token at 64); plus hand-built batches, e.g. 15 requests at rank 4 + 1 at rank 64 |
| Decode / prefill | decode = one new token per request; prefill = whole prompts (many tokens per request) |

**Timing.** Unless a table says otherwise, times are GPU time only: each call is captured in a
CUDA graph and replayed (median of 7 runs of 50 replays). This is how a serving engine runs decode
steps. Eager timing (a normal Python call) adds 70-110 us of Python and launch overhead to every
call on these nodes, which hides the kernels at decode sizes; tables 11 and 14 show it separately.

**Implementations compared.**

| Name | What it is |
|---|---|
| B0 | fp32 reference (correctness only) |
| B1 | plain PyTorch: pad every token to the batch's highest rank, mask, two cuBLAS matmuls |
| B2 | plain PyTorch: group tokens by rank, one gather + two matmuls + scatter per distinct rank (no wasted rank work, but more, smaller launches) |
| B3 | stock vLLM's way: every rank tier is a separate adapter padded to rank 64 (7 copies), run through vLLM 0.32's own LoRA kernels (`_lora_shrink` / `_lora_expand`, Punica's design) |
| RF1 | RankFuse `rf1_fused`: split-K shrink + fused mask/scale/expand, two Triton kernels, one copy of the adapter. `rf1_single` = its first, one-kernel design |
| RF2 | RF1 plus skipping unused rank rows per token block: sorted (`rf2_fused`, `rf2_splitk`), segment (`rf2_segment`), bucketed (`rf2_bucketed`) |

## Headline results

| Claim | Result | Table |
|---|---|---|
| Fusing shrink + mask + expand beats PyTorch | RF1 is **2.1-2.5x** faster than B1 at 1-128 tokens, 1.3-1.7x at 256-2048, 1.05-1.07x at 4096, even at 8192 | 01 |
| ... and more when adding into the base output | **1.4-1.8x** faster than B1 + add at 512-8192 tokens | 02 |
| Grouping by rank (B2) is the slowest option for mixed ranks | RF1 is **4.9-12.6x** faster than B2 when the batch holds 4-7 distinct ranks (8-8192 tokens), 1.8-2.6x when it holds one; RF2 split-K is 2.3-12.0x faster than B2. B2 launches 34-35 kernels per call at 7 ranks | 01-04, 07-09 |
| One shared copy beats vLLM's padded copies | RF1 is **1.8-2.5x** faster than B3 at decode and **2.2-2.8x** on prompts; 1.2-1.5x even when B3 uses a single copy | 03 |
| ... because it reads far less memory | at 32 tokens: RF1 reads **1.72 MB** from DRAM, B3 **7.92 MB** (its 7 padded copies), B1 2.04 MB; RF1 launches 2 kernels, B3 3, B1 8 | 04 |
| The design steps each paid off | split-K RF1 is 2.5-3.7x faster than the one-kernel RF1; the tiled expand took long prompts from 0.73x to parity with cuBLAS | 01, 05, 06 |
| Skipping unused ranks (RF2) helps only in one regime | +6-11% (segment) and +8-15% (bucketed) on large prompt batches mixing low ranks with a rank-64 request; neutral or slower elsewhere; the sorted version never pays | 07, 08, 09 |
| Gains survive in a whole model | Qwen2.5-0.5B forward with CUDA graphs: RF1 **1.38-1.47x** faster than B1 and **1.13-1.32x** than B3 at decode, 1.18-1.26x and 1.21-1.30x at prefill; LoRA's share of the step drops from 47-51% (B1) and 36-45% (B3) to 24-29% (RF1). Against B2: 1.85-4.9x with mixed ranks (B2's LoRA share 57-85%), 1.2-1.5x when every request has the same rank | 10 |
| ... but not yet without graphs | eager: RF1 is 0.80-0.87x of B1 (Python launch path ~108 us per call), still 1.15-1.23x faster than B3 | 11 |
| LoRA matters in a real server | stock vLLM, Qwen2.5-0.5B, graphs on: LoRA is **52 / 44 / 32 / 8%** of a decode step at batch 1 / 8 / 32 / 128 and 3-63% of a prefill; rank 16 costs almost as much as rank 64 | 12 |
| Everything is correct | 1020 tests, every prototype and the whole model checked against the fp32 reference | 13 |

### 13. Correctness

| Check | Result | Source |
|---|---|---|
| Repo test suite (every registered implementation vs B0: fp32/fp16/bf16, rank mixes, decode and prefill shapes, rank 0, odd sizes, r_max 128/256, add mode, output slices, CUDA-graph capture) | **1020 passed**, 29 skipped (B3 in fp32: vLLM's kernels take fp16/bf16 only); the B2 subset re-run after giving b2_grouped its per-step groups: 148 passed | `source/logs/full_test_suite_and_ncu_repo.out`, `source/logs/b2_timing_runs.out` |
| Prototype checks vs B0 before each timing run | RF1 split-K 264/264 and 492/492; sorted split-K RF2 82/82; segment RF2 99/99; bucketed RF2 102/102 (concurrent, single-stream and auto each); B3 53/53 (write and add mode each) | `source/logs/*_checks.out` |
| Whole Qwen2.5-0.5B, every backend (incl. B2 and RF2 sorted split-K) vs an fp32 reference (prefill logits and one decode step, 3 batches incl. rank 0, >128-token tiled paths, a mostly-rank-4 batch) | all pass: mean logit error 0.004-0.007 against a LoRA effect of 0.09-0.10, top-1 agreement >= 99.5%; a request gives the same logits alone and inside a mixed batch | `source/logs/whole_model_check_and_timing_repo.out`, `source/logs/b2_whole_model_check.out`, `source/logs/rf2sk_whole_model_check_and_timing.out` |
| Stock vLLM loads the generated adapters | full adapter and all 7 rank copies load; the rank-64 copy gives identical tokens to the full adapter | `source/logs/vllm_adapter_check.out` |

## All tables

| # | Table | Use it for |
|---|---|---|
| 01 | RF1 vs B1 and B2, one layer, 1-8192 tokens, two rank mixes | main operator result; latency-vs-tokens plot |
| 02 | RF1 vs B1 with the add into the base output | fusion of the residual add |
| 03 | RF1 vs B3 (vLLM's kernels), decode and prompts | Claim 1: one shared copy vs padded copies |
| 04 | DRAM / L2 traffic and kernel counts (ncu) at 32 and 2048 tokens | why RF1 wins: bytes read, launches |
| 05 | long-prompt ncu profile before the tiled expand | design story: what the profile showed |
| 06 | RF1 before vs after the tiled expand | design story: the fix |
| 07 | sorted RF2 (one kernel and split-K) vs RF1 | negative result: sorting costs what skipping saves |
| 08 | segment RF2 vs RF1 | RF2 helps on mixed-rank prompt batches |
| 09 | bucketed RF2 vs RF1, with the CPU rule that picks it | best RF2; per-batch choice |
| 10 | whole Qwen2.5-0.5B forward, CUDA graphs, all backends | end-to-end at the model level |
| 11 | the same, eager | the launch-overhead problem still open |
| 12 | LoRA share of a stock vLLM step | motivation / ceiling for end-to-end gains |
| 14 | every registered implementation from the committed repo, graph and eager | reproducibility check of the final code |

## Figures (CUDA-graph results only)

| File | Shows | From |
|---|---|---|
| `fig1_latency_vs_tokens` | one layer, time vs tokens: B1, B2, RF1 one kernel, RF1 split-K | table 01 |
| `fig2_speedup_vs_baselines` | RF1's speedup over B1, B2 and vLLM's kernels (B3), per workload | table 03 |
| `fig3_memory_traffic` | DRAM read and kernel launches per call at 32 and 2048 tokens | table 04 |
| `fig4_design_steps` | RF1 one kernel, then split-K, then tiled expand, against B1 | tables 01, 06 |
| `fig5_rf2_vs_rf1` | RF2 versions' speedup over RF1 per batch type, and RF1 time vs the batch's highest rank | tables 07-09 |
| `fig6_whole_model` | whole Qwen2.5-0.5B forward split into base model + LoRA, per backend | table 10 |
| `fig7_vllm_lora_share` | LoRA's share of a stock vLLM step, decode and prefill | table 12 |

## Caveats to state in the presentation

* One GPU (H200), one adapter (Stage 1), fp16. Layer tests use one shape (4096 x 4096); the
  whole model is small (0.5B, hidden size 896), which overstates LoRA's share compared with 7-8B.
* Single-layer RF2 and B3 numbers (tables 03, 07-09) were measured with the prototypes, which
  are the same kernels now in the repo; table 14 re-runs the committed code on a smaller grid
  (with an L2 flush before each call, so its numbers sit a little above tables 01-03).
* ncu times (tables 04, 05) are longer than graph-replay times: ncu flushes caches and locks
  clocks for every kernel. Compare bytes and kernel counts across implementations, not ncu time
  with the other tables.
* The CPU rule that picks bucketed RF2 (table 09) was fitted on the same data it is shown on.
* Every batch is pure decode (one token per request) or pure prefill (whole prompts). Mixed batches, which
  vLLM's chunked prefill (on by default in 0.32) produces, are not measured; correctness is unaffected
  (every kernel takes any per-token rank pattern), only their timing is untested.
* B2 was timed later (26 Oct) on exactly the same batches and with the same method; only its columns come
  from that run. Timing B2 right before another kernel disturbed the next measurement by up to ~30% on
  large batches (its gathers allocate large temporaries), so in the runs used here B2 is timed last; RF1
  re-timed in those runs agrees with the tables within a median of 1.8% (max 5.8%).
* Whole-model tests run the model in PyTorch, not inside vLLM; RankFuse is not integrated into
  vLLM yet (table 12 is stock vLLM only).

## Not included

Superseded development runs (eager timings from before graph replay was adopted, early split-K
prototypes and tuning sweeps) stay on PACE in `~/scratch/rankfuse/logs/`: `rf1_tune.json`, `rf1_splitk*.json`,
`rf_splitk_sweep.json`, `model_bench.json` (an earlier copy of table 10/11 without the RF2
columns; `model_bench_rf2.json` is the prototype run that table 10/11's repo re-run reproduces).
