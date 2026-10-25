### How much of a stock vLLM step is LoRA (Qwen2.5-0.5B, vLLM 0.32)

Stock vLLM 0.32.0 on one H200, prefix caching off, one engine per run. Decode step time at batch B (prompts of 128 tokens; (t_144 - t_16) / 128 removes the prefill); prefill = one request of L tokens. lora16 / lora64 = engine with max_lora_rank 16 / 64 serving one adapter of that rank; 'unused' = LoRA enabled but the request carries no adapter. 'LoRA %' = (time - base) / time. Source: p22_lora_share.jsonl.

| CUDA graphs | phase | batch or prompt tokens | base ms | rank 16 ms | LoRA % r16 | rank 64 ms | LoRA % r64 | LoRA enabled, unused ms | overhead % unused |
|---|---|---|---|---|---|---|---|---|---|
| graphs | decode step | 1 | 1.19 | 2.23 | 46.40 | 2.51 | 52.40 | 1.23 | 2.70 |
| graphs | decode step | 8 | 1.32 | 2.10 | 37.20 | 2.36 | 44.20 | 1.38 | 4.60 |
| graphs | decode step | 32 | 1.71 | 2.34 | 26.90 | 2.51 | 32.00 | 1.69 | -0.90 |
| graphs | decode step | 128 | 2.88 | 3.13 | 7.90 | 3.13 | 8.00 | 2.99 | 3.70 |
| graphs | prefill | 128 | 5.54 | 5.77 | 4.00 | 5.70 | 2.80 | 5.71 | 3.00 |
| graphs | prefill | 512 | 5.59 | 6.04 | 7.40 | 6.31 | 11.40 | 5.80 | 3.60 |
| graphs | prefill | 2048 | 9.18 | 24.92 | 63.20 | 24.71 | 62.90 | 12.72 | 27.90 |
| graphs | prefill | 8192 | 23.89 | 37.41 | 36.20 | 41.22 | 42.00 | 24.36 | 2.00 |
| eager | decode step | 1 | 8.77 | 27.84 | 68.50 | 27.69 | 68.30 | 14.44 | 39.30 |
| eager | decode step | 8 | 9.05 | 27.92 | 67.60 | 27.64 | 67.20 | 14.71 | 38.50 |
| eager | decode step | 32 | 9.36 | 28.56 | 67.20 | 28.18 | 66.80 | 15.16 | 38.30 |
| eager | decode step | 128 | 10.42 | 29.75 | 65.00 | 29.41 | 64.60 | 16.43 | 36.60 |
| eager | prefill | 128 | 10.35 | 28.96 | 64.30 | 29.14 | 64.50 | 16.15 | 35.90 |
| eager | prefill | 512 | 10.34 | 29.16 | 64.60 | 29.09 | 64.50 | 16.17 | 36.10 |
| eager | prefill | 2048 | 10.73 | 29.78 | 64.00 | 29.78 | 64.00 | 16.70 | 35.80 |
| eager | prefill | 8192 | 24.41 | 37.65 | 35.20 | 41.52 | 41.20 | 24.82 | 1.60 |
