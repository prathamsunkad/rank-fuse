### RF1 before and after the tiled expand (us per call, output written)

GPU time only: each call captured in a CUDA graph and replayed (median of 7 runs of 50 replays). One NVIDIA H200, fp16, 4096 x 4096 layer, r_max 64 (tiers 1, 2, 4, ..., 64). 'before' = rf1_fused on 24 Oct with the strip expand (expand_prefill.json); 'after' = final rf1_fused with the one-tile-per-program expand and vectorised stores (rf1_final_sweep.json, uniform). Speedups are against B1 measured in the same run.

| tokens | B1 us | RF1 before us | RF1 after us | before: speedup vs B1 | after: speedup vs B1 |
|---|---|---|---|---|---|
| 32 | 15.85 | 7.68 | 7.46 | 2.06 | 2.26 |
| 128 | 17.97 | 8.95 | 8.36 | 2.01 | 2.14 |
| 512 | 19.60 | 14.30 | 11.87 | 1.37 | 1.64 |
| 1024 | 21.62 | 19.47 | 13.00 | 1.11 | 1.66 |
| 2048 | 25.54 | 24.83 | 19.63 | 1.03 | 1.34 |
| 4096 | 41.95 | 45.97 | 37.34 | 0.91 | 1.07 |
| 8192 | 61.60 | 83.90 | 61.78 | 0.73 | 0.99 |
