### 09. Nsight Compute: DRAM and L2 traffic per call

One prepared call (caches flushed, clocks locked by ncu; compare bytes, not time, with other tables). Summed over the call's kernels. L2 = lts__t_sectors x 32 B.

| impl | shape | adapters | tokens | kernels | DRAM read MB | DRAM write MB | L2 MB | ncu time us |
|---|---|---|---|---|---|---|---|---|
| b3_multi | 4096x4096 | 16 | 128 | 3 | 79.80 | 3.83 | 161.17 | 67.6 |
| b3_multi | 4096x4096 | 16 | 32 | 3 | 29.94 | 0.00 | 52.89 | 43.8 |
| b3_multi | down | 16 | 128 | 3 | 56.13 | 2.64 | 121.82 | 51.4 |
| b3_multi | down | 16 | 32 | 3 | 21.06 | 0.00 | 36.08 | 36.1 |
| b4_multi_copies | 4096x4096 | 16 | 128 | 3 | 26.80 | 0.00 | 74.36 | 41.5 |
| b4_multi_copies | 4096x4096 | 16 | 32 | 3 | 11.96 | 0.00 | 23.89 | 18.7 |
| b4_multi_copies | down | 16 | 128 | 3 | 18.86 | 0.01 | 57.49 | 24.1 |
| b4_multi_copies | down | 16 | 32 | 3 | 8.42 | 0.00 | 17.50 | 13.8 |
| b4_multi_shared | 4096x4096 | 16 | 128 | 3 | 17.37 | 0.00 | 113.92 | 41.0 |
| b4_multi_shared | 4096x4096 | 16 | 32 | 3 | 10.20 | 0.00 | 34.45 | 18.7 |
| b4_multi_shared | down | 16 | 128 | 3 | 11.47 | 0.00 | 58.99 | 22.2 |
| b4_multi_shared | down | 16 | 32 | 3 | 6.26 | 0.00 | 17.95 | 13.2 |
| rf1_ma | 4096x4096 | 16 | 128 | 2 | 16.02 | 0.00 | 29.94 | 14.4 |
| rf1_ma | 4096x4096 | 16 | 32 | 2 | 8.37 | 0.00 | 15.43 | 13.2 |
| rf1_ma | down | 16 | 128 | 2 | 11.35 | 0.00 | 21.00 | 13.6 |
| rf1_ma | down | 16 | 32 | 2 | 5.91 | 0.00 | 10.74 | 12.2 |
