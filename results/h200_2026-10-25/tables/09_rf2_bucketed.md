### RF2 bucketed (one kernel pair per rank bucket, concurrent streams) vs RF1 (us per call)

GPU time only: each call captured in a CUDA graph and replayed (median of 7 runs of 50 replays). One NVIDIA H200, fp16, 4096 x 4096 layer, r_max 64 (tiers 1, 2, 4, ..., 64). work ratio = bucketed rank work / RF1's (lower = more to skip). '1 stream' = buckets run one after another. 'use_rf2 picks' applies the rule now in the repo (graphs on, > 128 tokens, work ratio <= 0.45, and >= 4096 tokens or >= 64 tokens per request); the rule was fitted on this same data. Source: rf2_bucketed.json. B2 = group tokens by rank, one PyTorch matmul pair per distinct rank (b2_grouped's body, groups built once per step outside the graph), timed later on the same batches with the same method (b2_grouped.json); speedups over B2 use this table's own RankFuse numbers.

| batch | tokens | max rank | mean rank | work ratio | B2 us | RF1 us | RF2 segment us | RF2 bucketed us | RF2 bucketed, 1 stream us | RF1 / RF2 bucketed | use_rf2 picks | RF1 / picked | RF2 bucketed speedup vs B2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 15 x 512 @r4 + 1 x 512 @r64 | 8192 | 64 | 7.80 | 0.30 | 272.2 | 62.60 | 55.82 | 54.47 | 67.34 | 1.15 | RF2 bucketed | 1.15 | 5.00 |
| 7 x 1024 @r8 + 1 x 1024 @r64 | 8192 | 64 | 15.00 | 0.34 | 246.3 | 62.23 | 57.02 | 55.69 | 66.41 | 1.12 | RF2 bucketed | 1.12 | 4.42 |
| 31 x 64 @r4 + 1 x 64 @r64 | 2048 | 64 | 5.90 | 0.27 | 83.13 | 20.10 | 17.91 | 17.49 | 26.11 | 1.15 | RF2 bucketed | 1.15 | 4.75 |
| 3 x 2048 @r16 + 1 x 2048 @r64 | 8192 | 64 | 28.00 | 0.44 | 243.2 | 62.36 | 58.13 | 56.46 | 67.19 | 1.10 | RF2 bucketed | 1.10 | 4.31 |
| mix 1/2/4/8/16/32/64 x 1024 | 7168 | 64 | 18.10 | 0.39 | 267.4 | 56.33 | 50.78 | 50.59 | 74.40 | 1.11 | RF2 bucketed | 1.11 | 5.28 |
| 16 x 512 skew_low | 8192 | 8 | 2.80 | 1.00 | 275.3 | 50.21 | 52.66 | 51.89 | 52.15 | 0.97 | RF1 | 1.00 | 5.31 |
| 16 x 512 uniform | 8192 | 32 | 7.10 | 0.53 | 286.0 | 53.94 | 53.38 | 54.76 | 67.72 | 0.99 | RF1 | 1.00 | 5.22 |
| 32 x 256 skew_low | 8192 | 64 | 9.20 | 0.30 | 309.5 | 61.91 | 55.39 | 56.84 | 81.07 | 1.09 | RF2 bucketed | 1.09 | 5.45 |
| 32 x 256 uniform | 8192 | 64 | 17.50 | 0.39 | 294.2 | 61.96 | 56.55 | 56.55 | 80.95 | 1.10 | RF2 bucketed | 1.10 | 5.20 |
| 64 x 128 skew_low | 8192 | 64 | 9.80 | 0.31 | 307.9 | 62.17 | 55.01 | 56.48 | 80.39 | 1.10 | RF2 bucketed | 1.10 | 5.45 |
| 64 x 128 uniform | 8192 | 64 | 17.90 | 0.40 | 300.6 | 61.95 | 55.91 | 57.42 | 81.19 | 1.08 | RF2 bucketed | 1.08 | 5.24 |
| 16 x 128 skew_low | 2048 | 8 | 2.80 | 1.00 | 112.5 | 13.72 | 14.54 | 15.48 | 15.27 | 0.89 | RF1 | 1.00 | 7.26 |
| 16 x 128 uniform | 2048 | 32 | 7.10 | 0.53 | 129.8 | 15.46 | 15.04 | 17.90 | 25.43 | 0.86 | RF1 | 1.00 | 7.25 |
| 8 x 1024 skew_low | 8192 | 8 | 2.40 | 1.00 | 281.3 | 50.30 | 52.69 | 52.15 | 51.99 | 0.96 | RF1 | 1.00 | 5.39 |
| 8 x 1024 uniform | 8192 | 16 | 5.00 | 1.00 | 265.2 | 50.60 | 52.74 | 52.30 | 52.45 | 0.97 | RF1 | 1.00 | 5.07 |
| 32 x 1 skew_low | 32 | 64 | 9.20 | 0.30 | 91.99 | 7.49 | 7.29 | 9.04 | 16.49 | 0.83 | RF1 | 1.00 | 10.17 |
| 32 x 1 uniform | 32 | 64 | 17.50 | 0.39 | 89.67 | 7.49 | 7.54 | 8.98 | 16.47 | 0.83 | RF1 | 1.00 | 9.98 |
| 128 x 1 skew_low | 128 | 64 | 10.60 | 0.32 | 94.89 | 8.37 | 8.19 | 9.75 | 17.41 | 0.86 | RF1 | 1.00 | 9.74 |
| 128 x 1 uniform | 128 | 64 | 19.40 | 0.41 | 96.98 | 8.48 | 8.35 | 9.20 | 17.91 | 0.92 | RF1 | 1.00 | 10.55 |
| 2048 x 1 skew_low | 2048 | 64 | 8.70 | 0.29 | 145.7 | 19.91 | 20.10 | 20.79 | 36.87 | 0.96 | RF1 | 1.00 | 7.01 |
| 2048 x 1 uniform | 2048 | 64 | 18.30 | 0.39 | 140.5 | 19.55 | 20.40 | 20.12 | 34.96 | 0.97 | RF1 | 1.00 | 6.98 |
| 8192 x 1 skew_low | 8192 | 64 | 8.80 | 0.29 | 309.2 | 62.14 | 62.70 | 56.22 | 82.24 | 1.11 | RF2 bucketed | 1.11 | 5.50 |
| 8192 x 1 uniform | 8192 | 64 | 18.10 | 0.39 | 300.4 | 62.26 | 64.43 | 57.06 | 84.34 | 1.09 | RF2 bucketed | 1.09 | 5.27 |
