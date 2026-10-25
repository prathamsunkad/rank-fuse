### Long-prompt profile that led to the tiled expand (ncu, output written)

Development-stage profile (23 Oct): at 2048-8192 tokens the split-K shrink matched cuBLAS but the expand was ~4x slower, which motivated the one-tile-per-program expand now in rf1_fused (table 06). Same ncu settings as table 04, uniform ranks. Source: ncu/*-prefill.csv.

| tokens | implementation | kernels | kernel time us (ncu) | DRAM read MB | DRAM write MB | L2 traffic MB | DRAM % peak |
|---|---|---|---|---|---|---|---|
| 2048 | B1 (PyTorch, graphed) | 7 | 34.40 | 20.49 | 0.00 | 72.50 | 12.40 |
| 2048 | RF1 one kernel, tuned | 1 | 36.10 | 17.86 | 1.84 | 132.9 | 11.40 |
| 2048 | RF1 split-K (prototype, before tiled expand) | 2 | 30.50 | 19.95 | 0.01 | 75.70 | 13.60 |
| 8192 | B1 (PyTorch, graphed) | 6 | 57.60 | 72.05 | 42.87 | 259.8 | 41.50 |
| 8192 | RF1 one kernel, tuned | 1 | 113.1 | 68.23 | 40.31 | 243.9 | 20.00 |
| 8192 | RF1 split-K (prototype, before tiled expand) | 2 | 103.7 | 72.42 | 42.70 | 260.7 | 23.10 |
