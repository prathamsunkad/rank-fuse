### Memory traffic per call, Nsight Compute (add-into-output mode)

ncu 2026.3, one H200, 4096 x 4096, r_max 64, uniform ranks (T=32 uses all 7 ranks), fp16. Only the last of three calls profiled; sums over that call's kernels. ncu flushes caches and locks clocks to base for every kernel, so times are longer than the graph-replay numbers. 'DRAM % peak' = (read + write) / time against 4.8 TB/s; writes that stay in L2 until the kernel ends are not counted as DRAM writes. B3's extra reads are its 7 padded adapter copies; B2 launches several small kernels per rank group. Source: ncu/b3cmp-*.csv (B2 profiled later, same settings and batch).

| tokens | implementation | kernels | kernel time us (ncu) | DRAM read MB | DRAM write MB | L2 traffic MB | DRAM % peak |
|---|---|---|---|---|---|---|---|
| 32 | B1 (PyTorch) | 8 | 28.50 | 2.04 | 0.00 | 6.20 | 1.50 |
| 32 | B2 (PyTorch, grouped) | 35 | 141.7 | 3.51 | 0.00 | 12.30 | 0.50 |
| 32 | B3 (vLLM kernels) | 3 | 19.90 | 7.92 | 0.00 | 16.40 | 8.30 |
| 32 | RF1 split-K | 2 | 11.60 | 1.72 | 0.00 | 5.40 | 3.10 |
| 32 | RF2 segment | 2 | 11.60 | 1.71 | 0.00 | 5.20 | 3.10 |
| 2048 | B1 (PyTorch) | 8 | 48.20 | 54.05 | 0.13 | 150.7 | 23.40 |
| 2048 | B2 (PyTorch, grouped) | 34 | 216.5 | 70.35 | 0.00 | 225.0 | 6.80 |
| 2048 | B3 (vLLM kernels) | 3 | 52.40 | 42.00 | 0.00 | 211.9 | 16.70 |
| 2048 | RF1 split-K | 2 | 26.30 | 36.73 | 0.00 | 130.7 | 29.10 |
| 2048 | RF2 segment | 2 | 28.30 | 36.77 | 0.00 | 129.5 | 27.10 |
