### RF1 vs B1 and B2, delta added into the base output (us per call)

GPU time only: each call captured in a CUDA graph and replayed (median of 7 runs of 50 replays). One NVIDIA H200, fp16, 4096 x 4096 layer, r_max 64 (tiers 1, 2, 4, ..., 64). out += delta, as inside a real layer: RF1 adds inside its expand kernel; B1 needs a separate add; B2 adds each group with index_add_. Uniform ranks. Source: rf1_final_sweep.json. B2 = group tokens by rank, one PyTorch matmul pair per distinct rank (b2_grouped's body, groups built once per step outside the graph), timed later on the same batches with the same method (b2_grouped.json); speedups over B2 use this table's own RankFuse numbers.

| tokens | B1 + add us | B2 (index_add) us | RF1 split-K (add) us | speedup | RF1 speedup vs B2 |
|---|---|---|---|---|---|
| 512 | 22.61 | 103.3 | 12.62 | 1.79 | 8.19 |
| 2048 | 41.28 | 163.8 | 22.70 | 1.82 | 7.21 |
| 4096 | 65.27 | 255.4 | 46.67 | 1.40 | 5.47 |
| 8192 | 109.3 | 413.0 | 79.13 | 1.38 | 5.22 |
