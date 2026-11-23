### RF1 vs B1 and B2, one layer, output written (us per call)

GPU time only: each call captured in a CUDA graph and replayed (median of 7 runs of 50 replays). One NVIDIA H200, fp16, 4096 x 4096 layer, r_max 64 (tiers 1, 2, 4, ..., 64). Decode-shaped batches (T requests, 1 token each); uniform and skew_low rank mixes. B1 = PyTorch pad-to-batch-max + mask (b1_padded_graphed's body). RF1 one kernel = rf1_single (first design); RF1 split-K = rf1_fused (final). Source: rf1_final_sweep.json. B2 = group tokens by rank, one PyTorch matmul pair per distinct rank (b2_grouped's body, groups built once per step outside the graph), timed later on the same batches with the same method (b2_grouped.json); speedups over B2 use this table's own RankFuse numbers. Check: RF1 re-timed alongside B2 differs from this table's RF1 by a median of 1.8% (max 5.8%).

| tokens | rank mix | B1 us | B2 us | RF1 one-kernel us | RF1 split-K us | speedup vs B1 | RF1 speedup vs B2 | speedup vs one-kernel |
|---|---|---|---|---|---|---|---|---|
| 1 | uniform | 12.10 | 13.72 | 19.27 | 5.79 | 2.09 | 2.37 | 3.33 |
| 8 | uniform | 13.84 | 52.01 | 20.22 | 5.77 | 2.40 | 9.01 | 3.50 |
| 32 | uniform | 16.84 | 89.15 | 23.74 | 7.46 | 2.26 | 11.95 | 3.18 |
| 128 | uniform | 17.86 | 96.28 | 25.99 | 8.36 | 2.14 | 11.52 | 3.11 |
| 256 | uniform | 18.38 | 98.45 | 30.65 | 11.08 | 1.66 | 8.89 | 2.77 |
| 512 | uniform | 19.50 | 100.8 | 32.79 | 11.87 | 1.64 | 8.49 | 2.76 |
| 1024 | uniform | 21.63 | 109.8 | 36.65 | 13.00 | 1.66 | 8.45 | 2.82 |
| 2048 | uniform | 26.40 | 144.7 | 62.39 | 19.63 | 1.34 | 7.37 | 3.18 |
| 4096 | uniform | 40.12 | 199.9 | 109.9 | 37.34 | 1.07 | 5.36 | 2.94 |
| 8192 | uniform | 61.05 | 300.5 | 155.0 | 61.78 | 0.99 | 4.86 | 2.51 |
| 1 | skew_low | 13.19 | 15.21 | 19.15 | 5.87 | 2.25 | 2.59 | 3.26 |
| 8 | skew_low | 13.73 | 51.71 | 19.94 | 5.41 | 2.54 | 9.56 | 3.68 |
| 32 | skew_low | 17.08 | 91.35 | 23.77 | 7.26 | 2.35 | 12.59 | 3.28 |
| 128 | skew_low | 18.03 | 95.34 | 25.89 | 8.05 | 2.24 | 11.85 | 3.22 |
| 256 | skew_low | 18.64 | 98.81 | 30.46 | 11.36 | 1.64 | 8.70 | 2.68 |
| 512 | skew_low | 19.18 | 100.7 | 32.75 | 11.96 | 1.60 | 8.42 | 2.74 |
| 1024 | skew_low | 21.38 | 112.8 | 36.72 | 13.16 | 1.63 | 8.58 | 2.79 |
| 2048 | skew_low | 26.13 | 145.3 | 63.56 | 19.89 | 1.31 | 7.30 | 3.19 |
| 4096 | skew_low | 40.12 | 200.4 | 104.9 | 38.03 | 1.05 | 5.27 | 2.76 |
| 8192 | skew_low | 61.67 | 309.0 | 155.6 | 61.76 | 1.00 | 5.00 | 2.52 |
