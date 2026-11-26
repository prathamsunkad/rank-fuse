### 06. RF2-multi variants vs RF1-MA

time(rf1_ma) / time(variant) over all shapes; > 1: the variant is faster. wins = batches where it is > 5% faster.

| r_max | variant | phase | speedup gmean (min-max) | wins >5% | best batch |
|---|---|---|---|---|---|
| 64 | rf2_ma_splitk | decode | 0.98x (0.53-1.26) | 360/4050 | 1.26x (gate_up7 decode-256 N=256 uniform skew_low seed 2) |
| 64 | rf2_ma_splitk | prefill | 1.02x (0.90-1.15) | 312/1620 | 1.15x (k_v prefill-8192 N=256 uniform skew_low seed 1) |
| 64 | rf2_ma_segment | decode | 0.85x (0.24-1.15) | 133/4050 | 1.15x (q_o decode-1 N=64 uniform uniform seed 0) |
| 64 | rf2_ma_segment | prefill | 1.02x (0.89-1.17) | 356/1620 | 1.17x (down prefill-8192 N=1 zipf skew_low seed 1) |
| 64 | rf2_ma_bucketed | decode | 0.98x (0.54-1.27) | 487/4050 | 1.27x (gate_up7 decode-256 N=256 uniform skew_low seed 2) |
| 64 | rf2_ma_bucketed | prefill | 1.01x (0.83-1.21) | 351/1620 | 1.21x (k_v prefill-8192 N=256 uniform skew_low seed 1) |
| 128 | rf2_ma_splitk | decode | 1.03x (0.51-2.21) | 1092/4050 | 2.21x (gate_up7 decode-256 N=1 zipf skew_low seed 0) |
| 128 | rf2_ma_splitk | prefill | 1.17x (0.89-1.93) | 964/1620 | 1.93x (gate_up7 prefill-8192 N=1 uniform skew_low seed 1) |
| 128 | rf2_ma_segment | decode | 0.87x (0.21-1.60) | 429/4050 | 1.60x (gate_up7 decode-256 N=1 zipf skew_low seed 0) |
| 128 | rf2_ma_segment | prefill | 1.17x (0.89-1.96) | 967/1620 | 1.96x (gate_up7 prefill-8192 N=1 zipf skew_low seed 1) |
| 128 | rf2_ma_bucketed | decode | 1.01x (0.49-2.20) | 815/4050 | 2.20x (gate_up7 decode-256 N=1 uniform skew_low seed 2) |
| 128 | rf2_ma_bucketed | prefill | 1.22x (0.75-2.34) | 907/1620 | 2.34x (gate_up7 prefill-8192 N=1 zipf skew_low seed 1) |
| 256 | rf2_ma_splitk | decode | 1.08x (0.57-2.67) | 1865/4050 | 2.67x (gate_up7 decode-256 N=1 uniform skew_low seed 1) |
| 256 | rf2_ma_splitk | prefill | 1.34x (0.95-2.20) | 996/1620 | 2.20x (down prefill-8192 N=1 zipf skew_low seed 1) |
| 256 | rf2_ma_segment | decode | 0.85x (0.16-1.89) | 872/4050 | 1.89x (gate_up7 decode-256 N=1 uniform skew_low seed 1) |
| 256 | rf2_ma_segment | prefill | 1.34x (0.95-2.68) | 991/1620 | 2.68x (down prefill-8192 N=1 zipf skew_low seed 1) |
| 256 | rf2_ma_bucketed | decode | 1.08x (0.54-4.10) | 1316/4050 | 4.10x (gate_up7 decode-256 N=1 uniform skew_low seed 1) |
| 256 | rf2_ma_bucketed | prefill | 1.74x (0.87-5.53) | 972/1620 | 5.53x (4096x4096 prefill-8192 N=1 zipf skew_low seed 1) |
