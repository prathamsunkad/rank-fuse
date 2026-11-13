### Memory Traffic and Hardware Profiling (NCU)

> **Environment:** NVIDIA H200, PACE ICE | PyTorch 2.13.0+cu130, Triton 3.7.1, vLLM 0.32.0 | git commit 506dad2ece50ec5de14604cc07ed22b9ef758a5e

Nsight Compute (ncu) profiling metrics for decode workloads on H200 (4096 x 4096, r_max 64, fp16). Measures DRAM read/write volume, L2 cache traffic, tensor pipe activity, and theoretical rows read. Measured DRAM read traffic: rf1_ma read 15.37 MB vs b3_ma 21.29 MB (27.8% reduction). At t32-n16, rf1_ma shows rows_read (240) = 2x ideal (120) because physical page allocation granularity is P=16, so active rank-8 adapters load a 16-rank page (ceil(8/16)*16 = 16 = 2*8), whereas rf1_ma_flat loads exact contiguous unpadded rows.

| config | impl | kernels | kernel time us | DRAM read MB | DRAM write MB | L2 traffic MB | tensor pipe % peak | rows read | ideal rows read |
|---|---|---|---|---|---|---|---|---|---|
| t32-n16 | rf1_ma | 25 | 72.80 | 15.37 | 0.00 | 29.79 | 5.70 | 208 | 104 |
| t32-n16 | rf1_ma_flat | 25 | 71.10 | 15.37 | 0.00 | 29.77 | 5.90 | 104 | 104 |
| t32-n16 | b1_ma_padded | 85 | 269.7 | 9.27 | 0.00 | 40.82 | 5.60 | 104 | 104 |
| t32-n16 | b3_ma | 4 | 49.80 | 21.29 | 0.03 | 39.42 | 6.60 | 832 | 104 |
| t32-n16 | b4_ma_copies | 4 | 143.0 | 8.41 | 0.00 | 15.79 | 0.20 | 256 | 104 |
| t32-n16 | b4_ma_shared | 4 | 142.0 | 7.63 | 0.00 | 15.26 | 0.20 | 256 | 104 |
| t128-n16 | rf1_ma | 25 | 95.70 | 18.26 | 0.00 | 34.86 | 7.70 | 256 | 128 |
| t128-n16 | rf1_ma_flat | 25 | 92.90 | 18.25 | 0.00 | 34.80 | 8.20 | 128 | 128 |
| t128-n16 | b1_ma_padded | 97 | 334.0 | 16.15 | 0.00 | 59.28 | 5.40 | 128 | 128 |
| t128-n16 | b3_ma | 4 | 40.10 | 30.51 | 0.03 | 68.72 | 6.40 | 1024 | 128 |
| t128-n16 | b4_ma_copies | 4 | 160.4 | 15.77 | 0.00 | 44.73 | 0.30 | 1024 | 128 |
| t128-n16 | b4_ma_shared | 4 | 161.2 | 14.20 | 0.00 | 43.86 | 0.30 | 1024 | 128 |
| t128-n64 | rf1_ma | 25 | 96.60 | 60.20 | 0.71 | 98.21 | 8.40 | 912 | 456 |
| t128-n64 | rf1_ma_flat | 25 | 95.90 | 60.20 | 0.84 | 98.20 | 9.20 | 456 | 456 |
| t128-n64 | b1_ma_padded | 337 | 1091.2 | 35.15 | 0.02 | 160.3 | 5.50 | 456 | 456 |
| t128-n64 | b3_ma | 4 | 112.0 | 80.85 | 3.47 | 166.0 | 5.50 | 3648 | 456 |
| t128-n64 | b4_ma_copies | 4 | 161.7 | 31.23 | 0.10 | 58.32 | 0.30 | 1024 | 456 |
| t128-n64 | b4_ma_shared | 4 | 160.9 | 28.61 | 0.00 | 56.96 | 0.30 | 1024 | 456 |
