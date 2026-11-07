### Every registered implementation, repo code (us per call)

bench/run_sweep.py on the committed repo (25 Oct), one H200, 4096 x 4096, r_max 64, uniform ranks, fp16, L2 flushed before each call. 'graph' column = --timing graph (GPU time; b1_padded_graphed replays its own graph so it is timed eagerly); 'eager' column = --timing eager (includes ~70-110 us of Python launch overhead per call, B3 and RF2 bucketed also build their per-step metadata per call). Sources: repo_sweep_graph.csv, repo_sweep_eager.csv.

| tokens | implementation | graph us | graph timing used | eager us |
|---|---|---|---|---|
| 1 | b1_padded | 13.20 | graph | 86.40 |
| 1 | b1_padded_graphed | 19.20 | eager (graphs itself) | 16.70 |
| 1 | b3_vllm | 12.70 | graph | 484.3 |
| 1 | rf1_fused | 6.20 | graph | 108.9 |
| 1 | rf2_fused | 32.60 | graph | 118.6 |
| 1 | rf2_splitk | 7.20 | graph | 125.2 |
| 1 | rf2_segment | 7.00 | graph | 142.4 |
| 1 | rf2_bucketed | 6.40 | graph | 229.3 |
| 16 | b1_padded | 18.30 | graph | 89.80 |
| 16 | b1_padded_graphed | 22.90 | eager (graphs itself) | 23.00 |
| 16 | b3_vllm | 16.90 | graph | 480.3 |
| 16 | rf1_fused | 7.20 | graph | 109.2 |
| 16 | rf2_fused | 46.50 | graph | 144.9 |
| 16 | rf2_splitk | 7.90 | graph | 134.3 |
| 16 | rf2_segment | 8.10 | graph | 141.3 |
| 16 | rf2_bucketed | 7.60 | graph | 387.6 |
| 32 | b1_padded | 19.10 | graph | 90.00 |
| 32 | b1_padded_graphed | 23.00 | eager (graphs itself) | 22.90 |
| 32 | b3_vllm | 18.00 | graph | 485.6 |
| 32 | rf1_fused | 8.50 | graph | 110.5 |
| 32 | rf2_fused | 47.50 | graph | 147.5 |
| 32 | rf2_splitk | 9.10 | graph | 132.0 |
| 32 | rf2_segment | 8.70 | graph | 143.9 |
| 32 | rf2_bucketed | 8.00 | graph | 488.8 |
| 128 | b1_padded | 16.90 | graph | 87.80 |
| 128 | b1_padded_graphed | 20.80 | eager (graphs itself) | 21.60 |
| 128 | b3_vllm | 21.20 | graph | 478.3 |
| 128 | rf1_fused | 6.80 | graph | 110.9 |
| 128 | rf2_fused | 43.80 | graph | 139.3 |
| 128 | rf2_splitk | 7.50 | graph | 131.2 |
| 128 | rf2_segment | 7.70 | graph | 143.4 |
| 128 | rf2_bucketed | 6.80 | graph | 243.2 |
| 2048 | b1_padded | 29.30 | graph | 91.30 |
| 2048 | b1_padded_graphed | 33.20 | eager (graphs itself) | 33.30 |
| 2048 | b3_vllm | 58.50 | graph | 483.6 |
| 2048 | rf1_fused | 18.40 | graph | 105.1 |
| 2048 | rf2_fused | 77.40 | graph | 168.4 |
| 2048 | rf2_splitk | 19.00 | graph | 125.2 |
| 2048 | rf2_segment | 19.00 | graph | 143.8 |
| 2048 | rf2_bucketed | 20.30 | graph | 463.2 |
| 4096 | b1_padded | 39.10 | graph | 92.00 |
| 4096 | b1_padded_graphed | 43.10 | eager (graphs itself) | 44.40 |
| 4096 | b3_vllm | 97.40 | graph | 504.5 |
| 4096 | rf1_fused | 37.40 | graph | 112.2 |
| 4096 | rf2_fused | 111.7 | graph | 202.9 |
| 4096 | rf2_splitk | 35.10 | graph | 129.5 |
| 4096 | rf2_segment | 33.60 | graph | 146.6 |
| 4096 | rf2_bucketed | 32.40 | graph | 655.3 |
