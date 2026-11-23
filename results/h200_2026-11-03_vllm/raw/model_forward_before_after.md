Eager wall time per forward (ms) of Qwen2.5-0.5B in bench/model_harness.py, before (rankfuse_fused.py at git HEAD) and
after the launch-path change; none/B1/B3 from the after run (the before run's agree within 2%). Last two columns: graph replay, GPU ms.

| workload | none | B1 | B3 | RF1 before | RF1 after | B1/RF1 before | B1/RF1 after | graph RF1 before | graph RF1 after |
|---|---|---|---|---|---|---|---|---|---|
| decode 1x1 all64 | 12.66 | 24.73 | 36.22 | 30.81 | 20.44 | 0.80x | 1.21x | 3.594 | 3.596 |
| decode 1x1 uniform | 12.70 | 24.68 | 36.33 | 31.24 | 20.85 | 0.80x | 1.18x | 3.481 | 3.481 |
| decode 8x1 all64 | 13.10 | 25.28 | 36.65 | 31.28 | 20.76 | 0.80x | 1.22x | 3.765 | 3.764 |
| decode 8x1 uniform | 13.21 | 25.10 | 36.51 | 31.86 | 21.28 | 0.79x | 1.18x | 3.654 | 3.654 |
| decode 32x1 all64 | 12.62 | 25.45 | 36.71 | 31.38 | 20.88 | 0.81x | 1.22x | 3.971 | 3.971 |
| decode 32x1 uniform | 12.72 | 25.70 | 36.52 | 31.14 | 21.52 | 0.82x | 1.19x | 3.972 | 3.972 |
| decode 128x1 all64 | 12.67 | 25.64 | 36.38 | 30.97 | 21.00 | 0.82x | 1.22x | 4.398 | 4.397 |
| decode 128x1 uniform | 13.22 | 25.71 | 37.10 | 31.48 | 21.07 | 0.82x | 1.22x | 4.397 | 4.396 |
| prefill 4x512 all64 | 12.95 | 25.78 | 36.83 | 30.74 | 21.29 | 0.84x | 1.21x | 11.076 | 11.077 |
| prefill 4x512 uniform | 13.07 | 25.82 | 36.88 | 30.60 | 21.01 | 0.84x | 1.23x | 10.590 | 10.589 |
| prefill 16x128 all64 | 13.32 | 26.01 | 36.70 | 31.16 | 21.55 | 0.83x | 1.21x | 10.156 | 10.153 |
| prefill 16x128 uniform | 13.03 | 25.77 | 36.89 | 30.96 | 21.18 | 0.84x | 1.22x | 10.156 | 10.151 |

B1/RF1 eager, all 24 workloads (4 rank mixes): before 0.78-0.85x, after 1.17-1.26x
graph-replay RF1, after/before: 0.999-1.001 (unchanged)
