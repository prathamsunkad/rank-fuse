### RF1 vs B4 on prompt-sized batches (us per call) -- NOT how S-LoRA serves prompts

**S-LoRA runs bgmv for decode only; for prompts (>= 200 tokens) it uses a separate Triton kernel. These rows run the bgmv kernel on 2048- and 8192-token batches (4 or 16 requests x 512 tokens, one rank per request) to show how the kernel itself scales, not S-LoRA's prefill time.** GPU time only: each call is captured in a CUDA graph with the per-step metadata its `prepare` builds (pools, token -> entry map, zeroed buffers) and replayed (bench/run_sweep.py --timing graph; median of 50 repetitions of 10 replays; L2 flushed before each call, flush time subtracted), except B1 (b1_padded_graphed), which replays its own graph and is timed with CUDA events around the call. One NVIDIA H200, fp16, r_max 64 (tiers 1, 2, 4, ..., 64), each call returns the LoRA delta [T, h_out]. Shapes: the 4096 x 4096 test layer and Qwen2.5-0.5B's projections. B4 copies = S-LoRA's bgmv kernel serving each rank tier as its own adapter at its true rank (ranks 1/2/4 stored padded to 8: 144 pool rows per matrix); B4 shared = the same kernel with A_t and B^T stored once (64 rows), shrink at each token's true rank, expand over all 64 columns. B3 = vLLM 0.32's shrink/expand with 7 copies padded to 64; B1 = PyTorch padded to the batch max rank. Speedup = baseline us / RF1 us.

| shape | workload | tokens | max rank | mean rank | B1 us | B3 us | RF1 us | B4 copies us | B4 shared us | RF1 speedup vs B4 copies | RF1 speedup vs B4 shared | RF1 speedup vs B3 | RF1 speedup vs B1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4096x4096 | prompts 4 x 512, uniform | 2048 | 2 | 1.75 | 39.07 | 52.41 | 16.77 | 408.5 | 407.4 | 24.36 | 24.29 | 3.13 | 2.33 |
| 4096x4096 | prompts 4 x 512, skew_low | 2048 | 1 | 1.00 | 41.12 | 52.25 | 16.53 | 408 | 406.9 | 24.68 | 24.62 | 3.16 | 2.49 |
| 4096x4096 | prompts 4 x 512, all_max | 2048 | 64 | 64.00 | 33.58 | 52.33 | 22.56 | 570 | 569.9 | 25.26 | 25.26 | 2.32 | 1.49 |
| 4096x4096 | prompts 16 x 512, uniform | 8192 | 32 | 7.12 | 64.90 | 183.1 | 52.78 | 1727 | 1742 | 32.72 | 33.01 | 3.47 | 1.23 |
| 4096x4096 | prompts 16 x 512, skew_low | 8192 | 8 | 2.81 | 67.95 | 190.8 | 48.79 | 1715 | 1728 | 35.15 | 35.41 | 3.91 | 1.39 |
| 4096x4096 | prompts 16 x 512, all_max | 8192 | 64 | 64.00 | 63.55 | 192.8 | 60.23 | 2371 | 2374 | 39.36 | 39.42 | 3.20 | 1.06 |
| q/o 896x896 | prompts 4 x 512, uniform | 2048 | 2 | 1.75 | 30.21 | 26.38 | 8.48 | 163.2 | 163.3 | 19.24 | 19.25 | 3.11 | 3.56 |
| q/o 896x896 | prompts 4 x 512, skew_low | 2048 | 1 | 1.00 | 25.57 | 25.76 | 7.85 | 162.5 | 163.1 | 20.71 | 20.78 | 3.28 | 3.26 |
| q/o 896x896 | prompts 4 x 512, all_max | 2048 | 64 | 64.00 | 23.68 | 25.65 | 10.37 | 179.4 | 179.4 | 17.30 | 17.31 | 2.47 | 2.28 |
| q/o 896x896 | prompts 16 x 512, uniform | 8192 | 32 | 7.12 | 30.34 | 79.25 | 16.69 | 633.6 | 631.8 | 37.97 | 37.86 | 4.75 | 1.82 |
| q/o 896x896 | prompts 16 x 512, skew_low | 8192 | 8 | 2.81 | 31.25 | 79.81 | 15.42 | 632.3 | 631.7 | 41.00 | 40.96 | 5.17 | 2.03 |
| q/o 896x896 | prompts 16 x 512, all_max | 8192 | 64 | 64.00 | 31.06 | 79.41 | 19.29 | 694.3 | 693.4 | 36.00 | 35.95 | 4.12 | 1.61 |
| k/v 896x128 | prompts 4 x 512, uniform | 2048 | 2 | 1.75 | 26.30 | 22.80 | 7.33 | 96.04 | 95.76 | 13.11 | 13.07 | 3.11 | 3.59 |
| k/v 896x128 | prompts 4 x 512, skew_low | 2048 | 1 | 1.00 | 23.20 | 23.34 | 7.55 | 95.58 | 96.06 | 12.66 | 12.72 | 3.09 | 3.07 |
| k/v 896x128 | prompts 4 x 512, all_max | 2048 | 64 | 64.00 | 22.69 | 23.56 | 9.91 | 112.3 | 112.7 | 11.33 | 11.37 | 2.38 | 2.29 |
| k/v 896x128 | prompts 16 x 512, uniform | 8192 | 32 | 7.12 | 25.92 | 66.48 | 11.60 | 363.1 | 363.4 | 31.29 | 31.31 | 5.73 | 2.23 |
| k/v 896x128 | prompts 16 x 512, skew_low | 8192 | 8 | 2.81 | 26.48 | 68.07 | 10.91 | 363 | 363.1 | 33.26 | 33.27 | 6.24 | 2.43 |
| k/v 896x128 | prompts 16 x 512, all_max | 8192 | 64 | 64.00 | 27.07 | 67.54 | 13.18 | 423.2 | 423.5 | 32.11 | 32.12 | 5.12 | 2.05 |
| gate/up 896x4864 | prompts 4 x 512, uniform | 2048 | 2 | 1.75 | 37.95 | 41.42 | 14.04 | 504.9 | 506.7 | 35.96 | 36.10 | 2.95 | 2.70 |
| gate/up 896x4864 | prompts 4 x 512, skew_low | 2048 | 1 | 1.00 | 38.56 | 41.97 | 14.28 | 504.4 | 505.9 | 35.32 | 35.43 | 2.94 | 2.70 |
| gate/up 896x4864 | prompts 4 x 512, all_max | 2048 | 64 | 64.00 | 31.34 | 42.24 | 21.22 | 521.7 | 522.4 | 24.58 | 24.61 | 1.99 | 1.48 |
| gate/up 896x4864 | prompts 16 x 512, uniform | 8192 | 32 | 7.12 | 50.99 | 156.5 | 43.14 | 2077 | 2088 | 48.15 | 48.39 | 3.63 | 1.18 |
| gate/up 896x4864 | prompts 16 x 512, skew_low | 8192 | 8 | 2.81 | 52.72 | 158.1 | 38.02 | 2084 | 2091 | 54.82 | 54.98 | 4.16 | 1.39 |
| gate/up 896x4864 | prompts 16 x 512, all_max | 8192 | 64 | 64.00 | 52.59 | 160.6 | 51.29 | 2149 | 2145 | 41.90 | 41.82 | 3.13 | 1.03 |
| down 4864x896 | prompts 4 x 512, uniform | 2048 | 2 | 1.75 | 38.02 | 45.68 | 14.14 | 164.9 | 164.3 | 11.66 | 11.61 | 3.23 | 2.69 |
| down 4864x896 | prompts 4 x 512, skew_low | 2048 | 1 | 1.00 | 31.41 | 44.37 | 14.12 | 164.3 | 164.1 | 11.63 | 11.62 | 3.14 | 2.22 |
| down 4864x896 | prompts 4 x 512, all_max | 2048 | 64 | 64.00 | 30.53 | 44.23 | 17.18 | 374.9 | 375.2 | 21.82 | 21.83 | 2.57 | 1.78 |
| down 4864x896 | prompts 16 x 512, uniform | 8192 | 32 | 7.12 | 50.30 | 135.1 | 36.43 | 679 | 681.2 | 18.64 | 18.70 | 3.71 | 1.38 |
| down 4864x896 | prompts 16 x 512, skew_low | 8192 | 8 | 2.81 | 51.34 | 138.9 | 34.76 | 659.8 | 659.6 | 18.98 | 18.98 | 4.00 | 1.48 |
| down 4864x896 | prompts 16 x 512, all_max | 8192 | 64 | 64.00 | 50.96 | 140.9 | 38.88 | 1497 | 1512 | 38.50 | 38.89 | 3.62 | 1.31 |
