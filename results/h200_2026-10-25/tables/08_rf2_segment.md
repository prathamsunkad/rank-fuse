### RF2 segment (consecutive-token blocks, no sort) vs RF1 (us per call, output written)

GPU time only: each call captured in a CUDA graph and replayed (median of 7 runs of 50 replays). One NVIDIA H200, fp16, 4096 x 4096 layer, r_max 64 (tiers 1, 2, 4, ..., 64). 'A x L @rR' = A requests of L tokens at rank R; 'skew_low'/'uniform' = one rank per request drawn from that mix. contig = fixed blocks (kept); aligned = blocks never cross a request (slow at decode). >1 in 'RF1 / RF2 segment' = segment RF2 faster. Source: rf2_segment.json. B2 = group tokens by rank, one PyTorch matmul pair per distinct rank (b2_grouped's body, groups built once per step outside the graph), timed later on the same batches with the same method (b2_grouped.json); speedups over B2 use this table's own RankFuse numbers.

| batch | max rank | mean rank | B1 us | B2 us | RF1 split-K us | RF2 sorted us | RF2 segment (contig) us | RF2 segment (aligned) us | RF1 / RF2 segment | RF2 segment speedup vs B2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 15 x 512 @r4 + 1 x 512 @r64 | 64 | 7.80 | 59.44 | 272.2 | 61.43 | 58.24 | 55.95 | 55.98 | 1.10 | 4.87 |
| 7 x 1024 @r8 + 1 x 1024 @r64 | 64 | 15.00 | 61.62 | 246.3 | 61.38 | 59.48 | 57.37 | 57.25 | 1.07 | 4.29 |
| 31 x 64 @r4 + 1 x 64 @r64 | 64 | 5.90 | 25.75 | 83.13 | 19.36 | 18.21 | 17.98 | 18.06 | 1.08 | 4.62 |
| 3 x 2048 @r16 + 1 x 2048 @r64 | 64 | 28.00 | 61.58 | 243.2 | 61.49 | 60.43 | 58.03 | 58.48 | 1.06 | 4.19 |
| mix 1/2/4/8/16/32/64 x 1024 | 64 | 18.10 | 56.08 | 267.4 | 55.06 | 52.95 | 50.39 | 50.80 | 1.09 | 5.31 |
| 16 x 512 skew_low | 8 | 2.80 | 63.22 | 275.3 | 50.06 | 53.73 | 52.00 | 52.38 | 0.96 | 5.29 |
| 16 x 512 uniform | 32 | 7.10 | 62.74 | 286.0 | 53.64 | 54.73 | 52.77 | 53.01 | 1.02 | 5.42 |
| 32 x 256 skew_low | 64 | 9.20 | 61.44 | 309.5 | 61.30 | 57.81 | 55.42 | 55.24 | 1.11 | 5.58 |
| 32 x 256 uniform | 64 | 17.50 | 60.77 | 294.2 | 61.37 | 59.57 | 56.19 | 56.37 | 1.09 | 5.24 |
| 64 x 128 skew_low | 64 | 9.80 | 61.44 | 307.9 | 61.42 | 58.05 | 55.25 | 55.17 | 1.11 | 5.57 |
| 64 x 128 uniform | 64 | 17.90 | 61.04 | 300.6 | 61.36 | 59.58 | 55.92 | 55.59 | 1.10 | 5.38 |
| 16 x 128 skew_low | 8 | 2.80 | 25.14 | 112.5 | 13.80 | 14.67 | 14.98 | 14.70 | 0.92 | 7.51 |
| 16 x 128 uniform | 32 | 7.10 | 25.37 | 129.8 | 15.17 | 15.75 | 15.09 | 14.88 | 1.01 | 8.61 |
| 8 x 1024 skew_low | 8 | 2.40 | 64.10 | 281.3 | 49.75 | 53.53 | 52.26 | 52.31 | 0.95 | 5.38 |
| 8 x 1024 uniform | 16 | 5.00 | 63.22 | 265.2 | 50.22 | 53.85 | 52.71 | 52.38 | 0.95 | 5.03 |
| 32 x 1 skew_low | 64 | 9.20 | 16.97 | 91.99 | 7.53 | 7.96 | 7.36 | 9.55 | 1.02 | 12.51 |
| 32 x 1 uniform | 64 | 17.50 | 17.12 | 89.67 | 7.42 | 7.96 | 7.37 | 10.00 | 1.01 | 12.17 |
| 128 x 1 skew_low | 64 | 10.60 | 17.87 | 94.89 | 8.12 | 8.53 | 8.44 | 21.39 | 0.96 | 11.24 |
| 128 x 1 uniform | 64 | 19.40 | 17.53 | 96.98 | 8.36 | 8.79 | 8.54 | 23.16 | 0.98 | 11.36 |
| 2048 x 1 skew_low | 64 | 8.70 | 25.87 | 145.7 | 19.53 | 18.13 | 19.72 | 361.3 | 0.99 | 7.39 |
| 2048 x 1 uniform | 64 | 18.30 | 26.95 | 140.5 | 19.81 | 18.99 | 20.58 | 389.0 | 0.96 | 6.83 |
| 8192 x 1 skew_low | 64 | 8.80 | 61.54 | 309.2 | 61.26 | 57.97 | 62.52 | 1432.4 | 0.98 | 4.95 |
| 8192 x 1 uniform | 64 | 18.10 | 61.84 | 300.4 | 61.52 | 59.60 | 64.30 | 1535.3 | 0.96 | 4.67 |
