### RF2 bucketed (one kernel pair per rank bucket, concurrent streams) vs RF1 (us per call)

GPU time only: each call captured in a CUDA graph and replayed (median of 7 runs of 50 replays). One NVIDIA H200, fp16, 4096 x 4096 layer, r_max 64 (tiers 1, 2, 4, ..., 64). work ratio = bucketed rank work / RF1's (lower = more to skip). '1 stream' = buckets run one after another. 'use_rf2 picks' applies the rule now in the repo (graphs on, > 128 tokens, work ratio <= 0.45, and >= 4096 tokens or >= 64 tokens per request); the rule was fitted on this same data. Source: rf2_bucketed.json.

| batch | tokens | max rank | mean rank | work ratio | RF1 us | RF2 segment us | RF2 bucketed us | RF2 bucketed, 1 stream us | RF1 / RF2 bucketed | use_rf2 picks | RF1 / picked |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 15 x 512 @r4 + 1 x 512 @r64 | 8192 | 64 | 7.80 | 0.30 | 62.60 | 55.82 | 54.47 | 67.34 | 1.15 | RF2 bucketed | 1.15 |
| 7 x 1024 @r8 + 1 x 1024 @r64 | 8192 | 64 | 15.00 | 0.34 | 62.23 | 57.02 | 55.69 | 66.41 | 1.12 | RF2 bucketed | 1.12 |
| 31 x 64 @r4 + 1 x 64 @r64 | 2048 | 64 | 5.90 | 0.27 | 20.10 | 17.91 | 17.49 | 26.11 | 1.15 | RF2 bucketed | 1.15 |
| 3 x 2048 @r16 + 1 x 2048 @r64 | 8192 | 64 | 28.00 | 0.44 | 62.36 | 58.13 | 56.46 | 67.19 | 1.10 | RF2 bucketed | 1.10 |
| mix 1/2/4/8/16/32/64 x 1024 | 7168 | 64 | 18.10 | 0.39 | 56.33 | 50.78 | 50.59 | 74.40 | 1.11 | RF2 bucketed | 1.11 |
| 16 x 512 skew_low | 8192 | 8 | 2.80 | 1.00 | 50.21 | 52.66 | 51.89 | 52.15 | 0.97 | RF1 | 1.00 |
| 16 x 512 uniform | 8192 | 32 | 7.10 | 0.53 | 53.94 | 53.38 | 54.76 | 67.72 | 0.99 | RF1 | 1.00 |
| 32 x 256 skew_low | 8192 | 64 | 9.20 | 0.30 | 61.91 | 55.39 | 56.84 | 81.07 | 1.09 | RF2 bucketed | 1.09 |
| 32 x 256 uniform | 8192 | 64 | 17.50 | 0.39 | 61.96 | 56.55 | 56.55 | 80.95 | 1.10 | RF2 bucketed | 1.10 |
| 64 x 128 skew_low | 8192 | 64 | 9.80 | 0.31 | 62.17 | 55.01 | 56.48 | 80.39 | 1.10 | RF2 bucketed | 1.10 |
| 64 x 128 uniform | 8192 | 64 | 17.90 | 0.40 | 61.95 | 55.91 | 57.42 | 81.19 | 1.08 | RF2 bucketed | 1.08 |
| 16 x 128 skew_low | 2048 | 8 | 2.80 | 1.00 | 13.72 | 14.54 | 15.48 | 15.27 | 0.89 | RF1 | 1.00 |
| 16 x 128 uniform | 2048 | 32 | 7.10 | 0.53 | 15.46 | 15.04 | 17.90 | 25.43 | 0.86 | RF1 | 1.00 |
| 8 x 1024 skew_low | 8192 | 8 | 2.40 | 1.00 | 50.30 | 52.69 | 52.15 | 51.99 | 0.96 | RF1 | 1.00 |
| 8 x 1024 uniform | 8192 | 16 | 5.00 | 1.00 | 50.60 | 52.74 | 52.30 | 52.45 | 0.97 | RF1 | 1.00 |
| 32 x 1 skew_low | 32 | 64 | 9.20 | 0.30 | 7.49 | 7.29 | 9.04 | 16.49 | 0.83 | RF1 | 1.00 |
| 32 x 1 uniform | 32 | 64 | 17.50 | 0.39 | 7.49 | 7.54 | 8.98 | 16.47 | 0.83 | RF1 | 1.00 |
| 128 x 1 skew_low | 128 | 64 | 10.60 | 0.32 | 8.37 | 8.19 | 9.75 | 17.41 | 0.86 | RF1 | 1.00 |
| 128 x 1 uniform | 128 | 64 | 19.40 | 0.41 | 8.48 | 8.35 | 9.20 | 17.91 | 0.92 | RF1 | 1.00 |
| 2048 x 1 skew_low | 2048 | 64 | 8.70 | 0.29 | 19.91 | 20.10 | 20.79 | 36.87 | 0.96 | RF1 | 1.00 |
| 2048 x 1 uniform | 2048 | 64 | 18.30 | 0.39 | 19.55 | 20.40 | 20.12 | 34.96 | 0.97 | RF1 | 1.00 |
| 8192 x 1 skew_low | 8192 | 64 | 8.80 | 0.29 | 62.14 | 62.70 | 56.22 | 82.24 | 1.11 | RF2 bucketed | 1.11 |
| 8192 x 1 uniform | 8192 | 64 | 18.10 | 0.39 | 62.26 | 64.43 | 57.06 | 84.34 | 1.09 | RF2 bucketed | 1.09 |
