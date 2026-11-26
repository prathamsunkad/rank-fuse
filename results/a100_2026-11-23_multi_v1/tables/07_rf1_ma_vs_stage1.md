### 07. RF1-MA (shuffled page pool) vs Stage 1 rf1_fused at one adapter

time(rf1_ma) / time(stage1_rf1_fused) on the same batch (1.00x = no cost of the multi-adapter machinery and page table).

| r_max | phase | time ratio gmean (min-max) | batches |
|---|---|---|---|
| 64 | decode | 1.09x (0.62-1.87) | 810 |
| 64 | prefill | 1.03x (0.97-1.08) | 324 |
| 128 | decode | 1.13x (0.53-2.02) | 810 |
| 128 | prefill | 1.05x (0.91-1.16) | 324 |
| 256 | decode | 1.18x (0.70-2.49) | 810 |
| 256 | prefill | 1.08x (0.94-1.15) | 324 |
