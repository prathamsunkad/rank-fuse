### RF1 vs B1, one layer, output written (us per call)

GPU time only: each call captured in a CUDA graph and replayed (median of 7 runs of 50 replays). One NVIDIA H200, fp16, 4096 x 4096 layer, r_max 64 (tiers 1, 2, 4, ..., 64). Decode-shaped batches (T requests, 1 token each); uniform and skew_low rank mixes. B1 = PyTorch pad-to-batch-max + mask (b1_padded_graphed's body). RF1 one kernel = rf1_single (first design); RF1 split-K = rf1_fused (final). Source: rf1_final_sweep.json.

| tokens | rank mix | B1 us | RF1 one-kernel us | RF1 split-K us | speedup vs B1 | speedup vs one-kernel |
|---|---|---|---|---|---|---|
| 1 | uniform | 12.10 | 19.27 | 5.79 | 2.09 | 3.33 |
| 8 | uniform | 13.84 | 20.22 | 5.77 | 2.40 | 3.50 |
| 32 | uniform | 16.84 | 23.74 | 7.46 | 2.26 | 3.18 |
| 128 | uniform | 17.86 | 25.99 | 8.36 | 2.14 | 3.11 |
| 256 | uniform | 18.38 | 30.65 | 11.08 | 1.66 | 2.77 |
| 512 | uniform | 19.50 | 32.79 | 11.87 | 1.64 | 2.76 |
| 1024 | uniform | 21.63 | 36.65 | 13.00 | 1.66 | 2.82 |
| 2048 | uniform | 26.40 | 62.39 | 19.63 | 1.34 | 3.18 |
| 4096 | uniform | 40.12 | 109.9 | 37.34 | 1.07 | 2.94 |
| 8192 | uniform | 61.05 | 155.0 | 61.78 | 0.99 | 2.51 |
| 1 | skew_low | 13.19 | 19.15 | 5.87 | 2.25 | 3.26 |
| 8 | skew_low | 13.73 | 19.94 | 5.41 | 2.54 | 3.68 |
| 32 | skew_low | 17.08 | 23.77 | 7.26 | 2.35 | 3.28 |
| 128 | skew_low | 18.03 | 25.89 | 8.05 | 2.24 | 3.22 |
| 256 | skew_low | 18.64 | 30.46 | 11.36 | 1.64 | 2.68 |
| 512 | skew_low | 19.18 | 32.75 | 11.96 | 1.60 | 2.74 |
| 1024 | skew_low | 21.38 | 36.72 | 13.16 | 1.63 | 2.79 |
| 2048 | skew_low | 26.13 | 63.56 | 19.89 | 1.31 | 3.19 |
| 4096 | skew_low | 40.12 | 104.9 | 38.03 | 1.05 | 2.76 |
| 8192 | skew_low | 61.67 | 155.6 | 61.76 | 1.00 | 2.52 |
