### RF1 vs B1, delta added into the base output (us per call)

GPU time only: each call captured in a CUDA graph and replayed (median of 7 runs of 50 replays). One NVIDIA H200, fp16, 4096 x 4096 layer, r_max 64 (tiers 1, 2, 4, ..., 64). out += delta, as inside a real layer: RF1 adds inside its expand kernel; B1 needs a separate add. Uniform ranks. Source: rf1_final_sweep.json.

| tokens | B1 + add us | RF1 split-K (add) us | speedup |
|---|---|---|---|
| 512 | 22.61 | 12.62 | 1.79 |
| 2048 | 41.28 | 22.70 | 1.82 |
| 4096 | 65.27 | 46.67 | 1.40 |
| 8192 | 109.3 | 79.13 | 1.38 |
