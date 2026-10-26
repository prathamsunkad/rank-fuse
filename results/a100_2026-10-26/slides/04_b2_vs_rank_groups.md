### Why B2 is slow: cost grows with the number of rank groups

NVIDIA A100-PCIE-40GB, rf_vs_b1 r_max 64 + 128 (seed 0), all mixes pooled. B2g = grouping built once, then gather + 2 matmuls + scatter per group replayed from a CUDA graph; B2 = the same in eager PyTorch, regrouping (with host syncs) every call.

| batch size | rank groups | workloads | B2g vs B1 (median) | B2g us | B1 us | B2 eager us |
|---|---|---|---|---|---|---|
| <=64 tokens | 1 | 44 | 0.84x | 22.5 | 18.7 | 221 |
| <=64 tokens | 2 | 4 | 0.47x | 39.4 | 18.9 | 337 |
| <=64 tokens | 3 | 4 | 0.38x | 54.3 | 20.6 | 438 |
| <=64 tokens | 4 | 8 | 0.30x | 77.3 | 23.3 | 551 |
| 256-4096 tokens | 1 | 16 | 0.67x | 54.8 | 35.3 | 232 |
| 256-4096 tokens | 3 | 4 | 0.41x | 64.0 | 26.5 | 445 |
| 256-4096 tokens | 4 | 20 | 0.37x | 119.3 | 44.4 | 564 |
