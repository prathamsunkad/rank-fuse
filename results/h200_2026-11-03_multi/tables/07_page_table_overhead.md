### 07. Page-table overhead

Time ratio on the same batches (1.00x = no cost). dense = stacked weights behind an identity page table; contiguous / shuffled = per-layer page pools with frames in order / permuted. Last rows: RF1-MA vs Stage 1's rf1_fused, which reads one adapter with plain pointers.

| kernel | layout | phase | time ratio gmean (min-max) | batches |
|---|---|---|---|---|
| rf1_ma | contiguous / dense | decode | 0.99x (0.91-1.09) | 75 |
| rf1_ma | contiguous / dense | prefill | 1.00x (0.94-1.04) | 30 |
| rf1_ma | shuffled / dense | decode | 1.00x (0.90-1.08) | 75 |
| rf1_ma | shuffled / dense | prefill | 1.00x (0.96-1.03) | 30 |
| rf2_ma_splitk | contiguous / dense | decode | 1.00x (0.93-1.07) | 75 |
| rf2_ma_splitk | contiguous / dense | prefill | 1.00x (0.95-1.03) | 30 |
| rf2_ma_splitk | shuffled / dense | decode | 1.01x (0.93-1.06) | 75 |
| rf2_ma_splitk | shuffled / dense | prefill | 1.01x (0.95-1.03) | 30 |
| rf1_ma (shuffled pool) / Stage 1 rf1_fused (no page table) | 1 adapter | decode | 1.11x (1.02-1.29) | 150 |
| rf1_ma (shuffled pool) / Stage 1 rf1_fused (no page table) | 1 adapter | prefill | 1.04x (0.99-1.11) | 60 |
