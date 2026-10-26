### Decode, r_max 64: baselines vs RankFuse

NVIDIA A100 80GB PCIe, 4096 x 4096, fp16, 7 tiers (1..64), uniform mix, median of 3 seeds (max rank and rank groups listed per seed when they differ).

| tokens | max rank | rank groups | B1 us | B2g us | RF1 us | RF2 us | RF1 vs B1 | B2g vs B1 | RF2 vs RF1 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1/2 | 1 | 12.9 | 20.5 | 6.3 | 7.0 | 2.03x | 0.63x | 0.90x |
| 8 | 16/32/64 | 4/5/6 | 17.1 | 90.1 | 7.6 | 7.9 | 2.25x | 0.19x | 0.96x |
| 32 | 64 | 7 | 21.1 | 120.8 | 8.7 | 9.5 | 2.42x | 0.17x | 0.91x |
| 128 | 64 | 7 | 22.3 | 160.8 | 11.7 | 11.7 | 1.91x | 0.14x | 1.00x |
| 512 | 64 | 7 | 31.8 | 175.1 | 20.3 | 21.9 | 1.57x | 0.18x | 0.93x |
| 2048 | 64 | 7 | 55.9 | 215.0 | 42.9 | 44.4 | 1.30x | 0.26x | 0.97x |
| 8192 | 64 | 7 | 142.8 | 524.3 | 131.2 | 138.8 | 1.09x | 0.27x | 0.94x |

### Prefill (long prompts), r_max 64: baselines vs RankFuse

NVIDIA A100 80GB PCIe, 4096 x 4096, fp16, 7 tiers, uniform mix, median of 3 seeds. With 1-16 requests the max rank depends on the draw, so RF1 is sometimes already narrow.

| requests x tokens | tokens | max rank | rank groups | B1 us | B2g us | RF1 us | RF2 us | RF1 vs B1 | B2g vs B1 | RF2 vs RF1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 x 2048 | 2048 | 1/2 | 1 | 52.0 | 133.1 | 31.9 | 33.3 | 1.63x | 0.39x | 0.96x |
| 4 x 512 | 2048 | 2/32/64 | 2/4 | 56.3 | 167.9 | 35.4 | 36.0 | 1.59x | 0.34x | 0.98x |
| 16 x 128 | 2048 | 32/64 | 6/7 | 55.9 | 212.0 | 42.7 | 43.2 | 1.31x | 0.26x | 0.99x |
| 4 x 2048 | 8192 | 2/32/64 | 2/4 | 142.8 | 499.7 | 105.5 | 106.0 | 1.35x | 0.29x | 1.00x |
| 8 x 1024 | 8192 | 16/32/64 | 4/5/6 | 128.7 | 506.9 | 105.3 | 105.6 | 1.22x | 0.25x | 1.00x |
| 16 x 512 | 8192 | 32/64 | 6/7 | 141.5 | 517.1 | 134.2 | 137.2 | 1.05x | 0.27x | 0.98x |

### r_max 64 vs 128 at the same batch shapes

NVIDIA A100-PCIE-40GB, 4096 x 4096, fp16, tiers r_max/8, /4, /2, full, seed 0. all_max = nothing to skip (control). Prefill here = 16-token requests.

| phase | requests x tokens | tokens | r_max | mix | max rank | rank groups | B1 us | B2g us | RF1 us | RF2 us | RF1 vs B1 | B2g vs B1 | RF2 vs RF1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decode | 64 x 1 | 64 | 64 | uniform | 64 | 4 | 22.4 | 78.8 | 10.6 | 11.2 | 2.11x | 0.28x | 0.95x |
| decode | 64 x 1 | 64 | 64 | skew_low | 64 | 4 | 23.0 | 84.0 | 10.4 | 11.0 | 2.21x | 0.27x | 0.95x |
| decode | 64 x 1 | 64 | 64 | all_max | 64 | 1 | 22.9 | 24.6 | 10.4 | 11.6 | 2.20x | 0.93x | 0.90x |
| decode | 64 x 1 | 64 | 128 | uniform | 128 | 4 | 24.1 | 75.8 | 15.6 | 15.2 | 1.55x | 0.32x | 1.02x |
| decode | 64 x 1 | 64 | 128 | skew_low | 128 | 4 | 24.4 | 79.9 | 15.3 | 15.3 | 1.59x | 0.31x | 1.00x |
| decode | 64 x 1 | 64 | 128 | all_max | 128 | 1 | 23.3 | 25.6 | 15.4 | 16.9 | 1.51x | 0.91x | 0.91x |
| decode | 256 x 1 | 256 | 64 | uniform | 64 | 4 | 27.3 | 84.0 | 18.4 | 18.0 | 1.48x | 0.33x | 1.02x |
| decode | 256 x 1 | 256 | 64 | skew_low | 64 | 4 | 27.5 | 81.9 | 18.7 | 19.1 | 1.47x | 0.34x | 0.98x |
| decode | 256 x 1 | 256 | 64 | all_max | 64 | 1 | 26.0 | 33.8 | 18.5 | 19.7 | 1.41x | 0.77x | 0.94x |
| decode | 256 x 1 | 256 | 128 | uniform | 128 | 4 | 29.8 | 81.9 | 26.0 | 24.2 | 1.15x | 0.36x | 1.08x |
| decode | 256 x 1 | 256 | 128 | skew_low | 128 | 4 | 30.0 | 81.9 | 27.0 | 23.5 | 1.11x | 0.37x | 1.15x |
| decode | 256 x 1 | 256 | 128 | all_max | 128 | 1 | 29.9 | 35.8 | 26.9 | 28.3 | 1.11x | 0.83x | 0.95x |
| prefill | 16 x 16 | 256 | 64 | uniform | 32 | 3 | 25.5 | 62.5 | 13.1 | 14.4 | 1.96x | 0.41x | 0.90x |
| prefill | 16 x 16 | 256 | 64 | skew_low | 32 | 3 | 25.5 | 63.5 | 14.4 | 14.3 | 1.77x | 0.40x | 1.01x |
| prefill | 16 x 16 | 256 | 64 | all_max | 64 | 1 | 27.1 | 33.8 | 18.7 | 19.4 | 1.45x | 0.80x | 0.97x |
| prefill | 16 x 16 | 256 | 128 | uniform | 64 | 3 | 27.4 | 64.5 | 18.7 | 19.1 | 1.46x | 0.42x | 0.98x |
| prefill | 16 x 16 | 256 | 128 | skew_low | 64 | 3 | 27.5 | 65.5 | 18.6 | 19.0 | 1.48x | 0.42x | 0.98x |
| prefill | 16 x 16 | 256 | 128 | all_max | 128 | 1 | 30.1 | 35.8 | 27.1 | 27.8 | 1.11x | 0.84x | 0.97x |
| prefill | 64 x 16 | 1024 | 64 | uniform | 64 | 4 | 44.3 | 116.7 | 29.7 | 30.3 | 1.49x | 0.38x | 0.98x |
| prefill | 64 x 16 | 1024 | 64 | skew_low | 64 | 4 | 44.6 | 118.8 | 30.0 | 29.3 | 1.49x | 0.38x | 1.02x |
| prefill | 64 x 16 | 1024 | 64 | all_max | 64 | 1 | 44.6 | 79.9 | 30.2 | 35.2 | 1.48x | 0.56x | 0.86x |
| prefill | 64 x 16 | 1024 | 128 | uniform | 128 | 4 | 44.5 | 119.8 | 50.6 | 40.2 | 0.88x | 0.37x | 1.26x |
| prefill | 64 x 16 | 1024 | 128 | skew_low | 128 | 4 | 44.7 | 120.8 | 50.3 | 38.3 | 0.89x | 0.37x | 1.31x |
| prefill | 64 x 16 | 1024 | 128 | all_max | 128 | 1 | 45.7 | 79.9 | 50.6 | 52.2 | 0.90x | 0.57x | 0.97x |
| prefill | 256 x 16 | 4096 | 64 | uniform | 64 | 4 | 91.8 | 277.5 | 78.8 | 82.8 | 1.16x | 0.33x | 0.95x |
| prefill | 256 x 16 | 4096 | 64 | skew_low | 64 | 4 | 92.4 | 273.4 | 78.9 | 80.0 | 1.17x | 0.34x | 0.99x |
| prefill | 256 x 16 | 4096 | 64 | all_max | 64 | 1 | 91.8 | 267.3 | 79.3 | 97.0 | 1.16x | 0.34x | 0.82x |
| prefill | 256 x 16 | 4096 | 128 | uniform | 128 | 4 | 113.7 | 284.7 | 141.2 | 102.8 | 0.80x | 0.40x | 1.37x |
| prefill | 256 x 16 | 4096 | 128 | skew_low | 128 | 4 | 113.8 | 275.5 | 141.6 | 90.0 | 0.80x | 0.41x | 1.57x |
| prefill | 256 x 16 | 4096 | 128 | all_max | 128 | 1 | 113.8 | 286.7 | 141.8 | 148.9 | 0.80x | 0.40x | 0.95x |

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
