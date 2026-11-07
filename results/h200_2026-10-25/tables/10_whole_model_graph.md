### Whole Qwen2.5-0.5B forward, CUDA graphs (ms per forward)

GPU time only: the whole forward captured in a CUDA graph and replayed (median of 7 x 20). Qwen2.5-0.5B-Instruct (24 layers, hidden 896) in PyTorch, fp16, all 168 projections (q,k,v,o,gate,up,down) carrying one shared rank-64 Matryoshka adapter; each request has one rank. Decode = one new token per request on a 128-token KV cache; prefill = whole prompts. Mixes: all64, uniform over the 7 tiers, mostlylow (skew_low), mostlylow+64 (one request at 64). 'LoRA %' = (time - none) / time. Every backend matches an fp32 reference (table 13). Source: model_forward_repo.json.

| phase | requests x tokens | rank mix | no LoRA ms | B1 ms | B3 ms | RF1 ms | RF2 segment ms | RF2 bucketed ms | RF2 auto ms | LoRA % B1 | LoRA % B3 | LoRA % RF1 | RF1 speedup vs B1 | RF1 speedup vs B3 | RF1 / RF2 bucketed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decode | 1 x 1 | all64 | 2.53 | 4.92 | 4.18 | 3.56 | 3.64 | 3.52 | 3.56 | 48.70 | 39.50 | 29.00 | 1.38 | 1.17 | 1.01 |
| decode | 1 x 1 | uniform | 2.53 | 4.90 | 4.18 | 3.44 | 3.55 | 3.43 | 3.44 | 48.40 | 39.50 | 26.50 | 1.43 | 1.21 | 1.00 |
| decode | 1 x 1 | mostlylow | 2.53 | 4.88 | 4.18 | 3.39 | 3.47 | 3.38 | 3.39 | 48.30 | 39.50 | 25.40 | 1.44 | 1.23 | 1.00 |
| decode | 1 x 1 | mostlylow+64 | 2.53 | 4.92 | 4.18 | 3.56 | 3.65 | 3.52 | 3.56 | 48.60 | 39.50 | 28.90 | 1.38 | 1.17 | 1.01 |
| decode | 8 x 1 | all64 | 2.69 | 5.49 | 4.20 | 3.73 | 3.81 | 3.69 | 3.73 | 51.00 | 36.00 | 27.90 | 1.47 | 1.13 | 1.01 |
| decode | 8 x 1 | uniform | 2.69 | 5.21 | 4.61 | 3.62 | 3.73 | 3.71 | 3.63 | 48.30 | 41.70 | 25.70 | 1.44 | 1.27 | 0.98 |
| decode | 8 x 1 | mostlylow | 2.69 | 5.06 | 4.59 | 3.56 | 3.65 | 3.56 | 3.56 | 46.80 | 41.30 | 24.30 | 1.42 | 1.29 | 1.00 |
| decode | 8 x 1 | mostlylow+64 | 2.69 | 5.49 | 4.59 | 3.73 | 3.82 | 3.75 | 3.73 | 51.00 | 41.40 | 28.00 | 1.47 | 1.23 | 1.00 |
| decode | 32 x 1 | all64 | 2.84 | 5.70 | 4.46 | 3.94 | 4.03 | 3.91 | 3.94 | 50.10 | 36.30 | 27.90 | 1.44 | 1.13 | 1.01 |
| decode | 32 x 1 | uniform | 2.86 | 5.70 | 5.19 | 3.94 | 4.04 | 4.03 | 3.94 | 49.80 | 44.90 | 27.50 | 1.44 | 1.32 | 0.98 |
| decode | 32 x 1 | mostlylow | 2.86 | 5.70 | 5.18 | 3.94 | 4.04 | 4.00 | 3.94 | 49.80 | 44.90 | 27.60 | 1.44 | 1.31 | 0.99 |
| decode | 32 x 1 | mostlylow+64 | 2.86 | 5.70 | 5.17 | 3.94 | 4.04 | 4.01 | 3.94 | 49.90 | 44.70 | 27.40 | 1.45 | 1.31 | 0.98 |
| decode | 128 x 1 | all64 | 3.20 | 6.15 | 5.15 | 4.36 | 4.45 | 4.32 | 4.36 | 48.00 | 37.90 | 26.60 | 1.41 | 1.18 | 1.01 |
| decode | 128 x 1 | uniform | 3.20 | 6.15 | 5.34 | 4.36 | 4.43 | 4.44 | 4.35 | 47.90 | 40.00 | 26.70 | 1.41 | 1.22 | 0.98 |
| decode | 128 x 1 | mostlylow | 3.19 | 6.14 | 5.34 | 4.36 | 4.42 | 4.40 | 4.35 | 48.00 | 40.20 | 26.70 | 1.41 | 1.23 | 0.99 |
| decode | 128 x 1 | mostlylow+64 | 3.20 | 6.15 | 5.33 | 4.36 | 4.43 | 4.41 | 4.35 | 48.00 | 40.00 | 26.60 | 1.41 | 1.22 | 0.99 |
| prefill | 4 x 512 | all64 | 8.73 | 13.00 | 13.32 | 11.01 | 11.17 | 10.98 | 11.01 | 32.90 | 34.50 | 20.70 | 1.18 | 1.21 | 1.00 |
| prefill | 4 x 512 | uniform | 8.74 | 13.01 | 13.24 | 10.53 | 10.66 | 11.09 | 10.54 | 32.80 | 34.00 | 17.10 | 1.23 | 1.26 | 0.95 |
| prefill | 4 x 512 | mostlylow | 8.74 | 12.98 | 13.29 | 10.36 | 10.48 | 10.57 | 10.36 | 32.70 | 34.20 | 15.70 | 1.25 | 1.28 | 0.98 |
| prefill | 4 x 512 | mostlylow+64 | 8.74 | 13.01 | 13.29 | 11.01 | 10.92 | 10.99 | 11.00 | 32.80 | 34.30 | 20.60 | 1.18 | 1.21 | 1.00 |
| prefill | 16 x 128 | all64 | 7.85 | 12.12 | 12.44 | 10.13 | 10.28 | 10.11 | 10.13 | 35.30 | 36.90 | 22.50 | 1.20 | 1.23 | 1.00 |
| prefill | 16 x 128 | uniform | 7.84 | 12.11 | 12.45 | 10.13 | 10.03 | 10.35 | 10.35 | 35.20 | 37.00 | 22.60 | 1.20 | 1.23 | 0.98 |
| prefill | 16 x 128 | mostlylow | 7.85 | 12.14 | 12.49 | 9.63 | 9.69 | 10.18 | 9.64 | 35.40 | 37.20 | 18.50 | 1.26 | 1.30 | 0.95 |
| prefill | 16 x 128 | mostlylow+64 | 7.85 | 12.12 | 12.50 | 10.12 | 9.99 | 10.34 | 10.34 | 35.20 | 37.20 | 22.50 | 1.20 | 1.23 | 0.98 |
