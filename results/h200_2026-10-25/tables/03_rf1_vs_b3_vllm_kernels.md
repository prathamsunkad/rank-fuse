### RF1 vs B3 (vLLM 0.32's own LoRA kernels), delta added into the output (us per call)

GPU time only: each call captured in a CUDA graph and replayed (median of 7 runs of 50 replays). One NVIDIA H200, fp16, 4096 x 4096 layer, r_max 64 (tiers 1, 2, 4, ..., 64). B3 = how stock vLLM serves a Matryoshka adapter: every rank tier is a separate adapter padded to rank 64 (7 copies), computed with vLLM's _lora_shrink/_lora_expand; 'copies used' = distinct ranks in the batch. all_max = every token at rank 64 (B3 then reads one copy). Source: b3_vllm.json. B2 = group tokens by rank, one PyTorch matmul pair per distinct rank (b2_grouped's body, groups built once per step outside the graph), timed later on the same batches with the same method (b2_grouped.json); speedups over B2 use this table's own RankFuse numbers.

| workload | tokens | B3 copies used | max rank | mean rank | B1 us | B2 us | B3 us | RF1 us | RF2 segment us | RF1 speedup vs B3 | RF1 speedup vs B1 | RF1 speedup vs B2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decode 1 x 1, uniform | 1 | 1 | 1 | 1.00 | 14.35 | 14.13 | 11.49 | 6.12 | 6.44 | 1.88 | 2.34 | 2.31 |
| decode 1 x 1, skew_low | 1 | 1 | 1 | 1.00 | 14.65 | 13.99 | 11.16 | 6.10 | 6.41 | 1.83 | 2.40 | 2.29 |
| decode 1 x 1, all_max | 1 | 1 | 64 | 64.00 | 15.16 | 14.35 | 11.18 | 7.75 | 7.49 | 1.44 | 1.96 | 1.85 |
| decode 8 x 1, uniform | 8 | 4 | 16 | 5.00 | 15.36 | 48.80 | 12.71 | 6.15 | 6.65 | 2.07 | 2.50 | 7.94 |
| decode 8 x 1, skew_low | 8 | 4 | 8 | 2.40 | 14.60 | 47.37 | 14.37 | 5.78 | 6.46 | 2.48 | 2.52 | 8.19 |
| decode 8 x 1, all_max | 8 | 1 | 64 | 64.00 | 17.79 | 16.62 | 9.64 | 7.55 | 7.57 | 1.28 | 2.35 | 2.20 |
| decode 32 x 1, uniform | 32 | 7 | 64 | 17.50 | 18.59 | 93.13 | 15.33 | 7.85 | 7.70 | 1.95 | 2.37 | 11.87 |
| decode 32 x 1, skew_low | 32 | 7 | 64 | 9.20 | 18.66 | 95.51 | 15.71 | 7.97 | 7.88 | 1.97 | 2.34 | 11.99 |
| decode 32 x 1, all_max | 32 | 1 | 64 | 64.00 | 18.51 | 14.04 | 9.54 | 7.79 | 7.79 | 1.22 | 2.38 | 1.80 |
| decode 128 x 1, uniform | 128 | 7 | 64 | 19.40 | 19.16 | 105.8 | 16.13 | 8.36 | 8.75 | 1.93 | 2.29 | 12.66 |
| decode 128 x 1, skew_low | 128 | 7 | 64 | 10.60 | 19.40 | 108.7 | 15.89 | 8.52 | 8.66 | 1.86 | 2.28 | 12.75 |
| decode 128 x 1, all_max | 128 | 1 | 64 | 64.00 | 19.28 | 17.25 | 12.94 | 8.58 | 8.50 | 1.51 | 2.25 | 2.01 |
| decode 512 x 1, uniform | 512 | 7 | 64 | 18.40 | 22.46 | 104.2 | 22.59 | 12.93 | 13.36 | 1.75 | 1.74 | 8.05 |
| decode 2048 x 1, uniform | 2048 | 7 | 64 | 18.30 | 41.21 | 160.8 | 52.42 | 22.76 | 23.53 | 2.30 | 1.81 | 7.07 |
| decode 8192 x 1, uniform | 8192 | 7 | 64 | 18.10 | 110.0 | 413.6 | 161.7 | 79.38 | 82.04 | 2.04 | 1.39 | 5.21 |
| prompts 4 x 512, uniform | 2048 | 2 | 2 | 1.80 | 47.11 | 107.0 | 43.86 | 15.54 | 16.50 | 2.82 | 3.03 | 6.89 |
| prompts 16 x 512, skew_low | 8192 | 4 | 8 | 2.80 | 114.8 | 391.1 | 169.9 | 64.39 | 66.38 | 2.64 | 1.78 | 6.07 |
| prompts 4 x 2048, uniform | 8192 | 2 | 2 | 1.80 | 134.8 | 391.3 | 171.6 | 64.57 | 66.73 | 2.66 | 2.09 | 6.06 |
| prompts 15 x 512 at rank 4 + 1 x 512 at rank 64 | 8192 | 2 | 64 | 7.80 | 110.1 | 400.2 | 173.7 | 79.64 | 75.69 | 2.18 | 1.38 | 5.02 |
