### 06. RF2-multi variants vs RF1-MA

time(rf1_ma) / time(variant); > 1: the variant is faster. wins = batches where it is > 5% faster.

| variant | phase | shape | speedup gmean (min-max) | wins >5% | best batch |
|---|---|---|---|---|---|
| rf2_ma_splitk | decode | 4096->4096 | 0.98x (0.88-1.19) | 8/150 | 1.19x (decode-256 N=256 zipf skew_low) |
| rf2_ma_splitk | decode | 896->896 | 0.97x (0.86-1.11) | 5/150 | 1.11x (decode-256 N=256 uniform skew_low) |
| rf2_ma_splitk | decode | 896->128 | 0.97x (0.86-1.07) | 1/150 | 1.07x (decode-1 N=16 zipf skew_low) |
| rf2_ma_splitk | decode | 896->4864 | 0.98x (0.84-1.20) | 7/150 | 1.20x (decode-256 N=256 uniform skew_low) |
| rf2_ma_splitk | decode | 4864->896 | 0.98x (0.85-1.13) | 4/150 | 1.13x (decode-256 N=256 uniform skew_low) |
| rf2_ma_splitk | prefill | 4096->4096 | 1.01x (0.96-1.05) | 4/60 | 1.05x (prefill-8192 N=1 zipf uniform) |
| rf2_ma_splitk | prefill | 896->896 | 1.00x (0.93-1.06) | 1/60 | 1.06x (prefill-2048 N=4 zipf uniform) |
| rf2_ma_splitk | prefill | 896->128 | 0.97x (0.92-1.02) | 0/60 | 1.02x (prefill-8192 N=1 uniform skew_low) |
| rf2_ma_splitk | prefill | 896->4864 | 1.02x (0.96-1.11) | 7/60 | 1.11x (prefill-2048 N=1 zipf skew_low) |
| rf2_ma_splitk | prefill | 4864->896 | 1.00x (0.95-1.03) | 0/60 | 1.03x (prefill-2048 N=16 zipf skew_low) |
| rf2_ma_segment | decode | 4096->4096 | 0.83x (0.31-1.06) | 1/150 | 1.06x (decode-1 N=64 uniform skew_low) |
| rf2_ma_segment | decode | 896->896 | 0.91x (0.54-1.10) | 10/150 | 1.10x (decode-32 N=4 zipf uniform) |
| rf2_ma_segment | decode | 896->128 | 0.95x (0.75-1.11) | 10/150 | 1.11x (decode-256 N=4 zipf skew_low) |
| rf2_ma_segment | decode | 896->4864 | 0.81x (0.32-1.08) | 5/150 | 1.08x (decode-8 N=256 uniform skew_low) |
| rf2_ma_segment | decode | 4864->896 | 0.88x (0.44-1.06) | 3/150 | 1.06x (decode-1 N=16 uniform skew_low) |
| rf2_ma_segment | prefill | 4096->4096 | 1.02x (0.97-1.08) | 9/60 | 1.08x (prefill-2048 N=4 zipf skew_low) |
| rf2_ma_segment | prefill | 896->896 | 1.00x (0.93-1.04) | 0/60 | 1.04x (prefill-2048 N=64 uniform skew_low) |
| rf2_ma_segment | prefill | 896->128 | 0.97x (0.92-1.04) | 0/60 | 1.04x (prefill-2048 N=1 zipf skew_low) |
| rf2_ma_segment | prefill | 896->4864 | 1.02x (0.96-1.11) | 16/60 | 1.11x (prefill-2048 N=1 zipf skew_low) |
| rf2_ma_segment | prefill | 4864->896 | 1.00x (0.95-1.03) | 0/60 | 1.03x (prefill-8192 N=1 zipf uniform) |
| rf2_ma_bucketed | decode | 4096->4096 | 1.01x (0.76-1.33) | 35/150 | 1.33x (decode-128 N=256 zipf skew_low) |
| rf2_ma_bucketed | decode | 896->896 | 0.99x (0.74-1.22) | 31/150 | 1.22x (decode-256 N=256 uniform skew_low) |
| rf2_ma_bucketed | decode | 896->128 | 0.99x (0.85-1.12) | 26/150 | 1.12x (decode-256 N=4 zipf skew_low) |
| rf2_ma_bucketed | decode | 896->4864 | 1.00x (0.75-1.25) | 35/150 | 1.25x (decode-128 N=256 zipf skew_low) |
| rf2_ma_bucketed | decode | 4864->896 | 1.01x (0.77-1.21) | 27/150 | 1.21x (decode-256 N=4 uniform skew_low) |
| rf2_ma_bucketed | prefill | 4096->4096 | 1.00x (0.92-1.08) | 7/60 | 1.08x (prefill-8192 N=1 zipf uniform) |
| rf2_ma_bucketed | prefill | 896->896 | 0.99x (0.88-1.09) | 8/60 | 1.09x (prefill-8192 N=256 zipf uniform) |
| rf2_ma_bucketed | prefill | 896->128 | 0.96x (0.88-1.04) | 0/60 | 1.04x (prefill-2048 N=256 uniform all_max) |
| rf2_ma_bucketed | prefill | 896->4864 | 1.00x (0.88-1.10) | 12/60 | 1.10x (prefill-8192 N=4 zipf uniform) |
| rf2_ma_bucketed | prefill | 4864->896 | 1.01x (0.96-1.06) | 3/60 | 1.06x (prefill-2048 N=16 zipf skew_low) |
