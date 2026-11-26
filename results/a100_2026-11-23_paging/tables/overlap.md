### RF1-MA decode step with and without a concurrent H2D copy (NVIDIA A100 80GB PCIe) — rerun job 7284990 (source/rerun_overlap_ready)

One CUDA graph = RF1-MA on every (layer, q|k|v / o / gate|up / down) of the model (2 launches each), ranks 16/32/64/128 uniform over the step's tokens, every adapter resident at r 128 in the whole-model pool. 'with copy': a copy stream streams whole-model pages for the whole timed window.

| model | tokens | adapters | step alone ms (p95) | step with copy ms (p95) | slowdown | copy GB/s alone | copy GB/s during steps | copy covered window |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0.5B | 16 | 32 | 1.437 (1.441) | 1.440 (1.443) | x1.002 | 26.2 | 26.2 | False |
| 0.5B | 64 | 32 | 2.874 (2.882) | 2.967 (2.973) | x1.032 | 26.3 | 26.3 | True |
| 0.5B | 256 | 32 | 4.192 (4.204) | 4.291 (4.301) | x1.024 | 26.3 | 26.3 | True |
| 7B | 16 | 16 | 3.604 (3.613) | 3.663 (3.669) | x1.016 | 26.3 | 26.3 | True |
| 7B | 64 | 16 | 7.376 (7.395) | 7.590 (7.609) | x1.029 | 26.3 | 26.3 | True |
| 7B | 256 | 16 | 10.627 (10.642) | 10.848 (10.941) | x1.021 | 26.3 | 26.3 | True |
