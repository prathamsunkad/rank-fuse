# RF1 results, start to finish

RF1 (`rf1_fused`) computes a Matryoshka LoRA delta for a batch where every token carries its own
rank: one shared copy of the adapter, every token computed at the batch's highest rank with the
unused rows masked. This file follows how it got to its current form and what each stage measured.
Speedup = baseline time ÷ RF1 time, so above 1× means RF1 is faster. All times are GPU time (CUDA
graph replay), fp16, a 4096 × 4096 layer, unless noted. Single-rank batches (`all_min`, `all_max`)
are left out of summaries except where a row says so.

Baselines: **B1** pads every token to the batch's highest rank and runs two cuBLAS matmuls (plain
PyTorch); **B3** is stock vLLM 0.32's LoRA kernels with every rank tier as a separate padded adapter.

## Stage 1: one kernel, then split-K (H200, 23–25 Oct)

Source: `h200_2026-10-25/ALL_TABLES.md` (tables 01–06), r_max 64.

| Design | Result |
|---|---|
| One fused kernel (`rf1_single`) | Correct, but slow: one program per token block does the whole shrink |
| Split-K shrink + fused mask/scale/expand (two kernels), now `rf1_fused` | 2.5–3.7× faster than the one-kernel design |
| Long prompts, strip expand | Expand ~4× slower than cuBLAS at 2048–8192 tokens (ncu, table 05); RF1 0.73× of B1 at 8192 |
| Tiled expand (one output tile per program) + vectorised stores | 8192 tokens: 0.73× → 0.99× of B1; 1024 tokens: 1.11× → 1.66× |

Final H200 RF1 against the baselines:

| Comparison | Result |
|---|---|
| vs B1, output written | 2.1–2.5× at 1–128 tokens, 1.3–1.7× at 256–2048, ~1.0–1.07× at 4096–8192 |
| vs B1, delta added into the base output | 1.4–1.8× at 512–8192 tokens |
| vs B3 (vLLM's kernels) | 1.8–2.5× at decode, 2.2–2.8× on prompts; reads 1.72 MB of DRAM vs B3's 7.92 MB at 32 tokens |
| Whole Qwen2.5-0.5B forward, CUDA graphs | 1.38–1.47× faster than B1 at decode, 1.18–1.26× at prefill |
| Same, eager (no graphs) | 0.80–0.87× of B1: Python launch overhead (~108 µs per call) dominates |

## Stage 2: A100 sweeps (26 Oct)

Source: `a100_2026-10-26/ALL_ROWS.md`. Sampled rank mixes; geomean over cases.

| Sweep | Phase | Cases | RF1 vs B1 (range) |
|---|---|---|---|
| r_max 64, tiers 8–64 (A100 40GB) | decode | 15 | 2.03× (1.47–2.58) |
| | prefill | 15 | 1.83× (1.16–3.08) |
| r_max 128, tiers 16–128 (A100 40GB) | decode | 15 | 1.73× (1.11–2.50) |
| | prefill | 15 | 1.42× (0.80–2.73) |
| H200 shapes replayed (A100 80GB) | decode | 90 | 1.62× (1.09–2.57) |
| | prefill | 54 | 1.32× (1.05–1.88) |

RF1 beat B1 almost everywhere at r_max 64 and 128; the one loss (0.80×) was a large r_max 128 prefill.

## Stage 3: r_max 256 exposes a tuning gap (5 Nov, A100 80GB PCIe)

Source: `a100_2026-11-05/SUMMARY.md`. With every token at r_max 256 and 8192 tokens, RF1 took
**812 µs against B1's 358 µs** (B1 2.3× faster). The cause: RF1's block size had been tuned at
r_max 64 only, and at width 256 it fell back to 16-token blocks. At r_max 128 RF1 was still slightly
faster than B1 (305 vs 324 µs at 8192 tokens), and at r_max 64 about 1.7× faster.

## Stage 4: per-width tuning (6 Nov, A100 80GB PCIe)

Source: `a100_2026-11-06/SUMMARY.md`, `rankfuse/kernels/tuned_a100.py`, tuning data in
`raw/tune/a100.csv`. One table of launch settings (tokens per block, shrink and expand warps, shrink
tile and pipeline depth, expand tile), looked up by width and token count, searched on single-rank
batches and kept only when more than 2% faster than the default over 3 seeds. The same table is used
by every RF2 kernel.

Gain over the default settings, single-rank tuning batches:

| Width | 256–1024 tokens | 2048 | 8192 |
|---|---|---|---|
| 16 | 1.09× | 1.02× | default |
| 32 | 1.14× | default | default |
| 64 | 1.15× | default | 1.03× |
| 128 | 1.07× | 1.27× | 1.43× |
| 256 | 1.16× | 1.25× | 1.34× |

On the mixed-rank sweeps, against the 5 Nov run:

| r_max | RF1 tuned vs untuned | RF1 vs B1, untuned → tuned |
|---|---|---|
| 64 | 1.05× | 1.68× → 1.77× |
| 128 | 1.27× | 1.09× → 1.38× |
| 256 | 1.24× | 0.58× → 0.72× |

Full width, r_max 256, 8192 tokens: **805 → 602 µs**; B1 355 µs. RF1 alone still loses to B1 at
width 256 by about 1.4×.

## Stage 5: what each RF1 optimization is worth (6 Nov ablation)

Source: `a100_2026-11-06/RF1_ABLATION.md`. Each row removes one optimization from the tuned kernel;
"slower by" at 8192 tokens.

| Removed | r_max 64 | r_max 128 |
|---|---|---|
| Tuned block size (forced to 16 tokens) | 1.39× | 1.90× |
| Fused add into the output | 1.60× | 1.45× |
| Aligned-store hint | 1.40× | 1.34× |
| Split-K (at 32 tokens) | 2.25× | 1.91× |
| Tiled expand | 1.07× | **0.89×** (removing it is faster) |

The tiled expand loses to the loop expand at width 128 (by 11–20% at 512–8192 tokens). Commit
f65d09a makes the expand mode a tuning option; that retune (job 15336639) had not run when this was
written.

## Where RF1 stands

* Beats B1 at r_max 64 (about 1.8×) and r_max 128 (about 1.4×) on every mixed-rank workload tested.
* Loses to B1 at r_max 256 when everything runs at full width (0.72×). This is the one regime where
  RF1 alone is not enough; RF2 (see `RF2_HISTORY.md`) more than recovers it on mixed-rank batches.
* Next step if RF1 itself must match B1 at width 256: an expand that loops over the rank in chunks.
