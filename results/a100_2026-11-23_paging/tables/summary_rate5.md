### rf_prefix vs each baseline over the whole grid

Per config (model, r_max, budget, adapters, alpha, mix) where both are feasible. Miss ratio = baseline misses / rf_prefix misses (>1: prefix misses less); bytes ratio = baseline bytes / rf_prefix bytes; wait ratio likewise (configs with prefix wait > 0).

| baseline | configs | prefix hit rate >= baseline | median hit-rate gain (pts) | max gain (pts) | median miss ratio | median bytes ratio | median wait ratio | only prefix feasible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vllm | 1620 | 1620 | 31.45 | 74.5 | 1.54 | 1.18 | 1.55 | 0 |
| slora | 1620 | 1620 | 15.83 | 37.4 | 1.34 | 1.05 | 1.10 | 0 |
| rf_whole | 1620 | 1588 | 6.95 | 37.7 | 1.16 | 2.99 | 3.06 | 0 |

### Smallest budget (GB) reaching a 50% hit rate

1000 adapters, uniform tiers; >20 = not reached at 20 GB.

| model | r_max | alpha | vllm | slora | rf_whole | rf_prefix |
|---|---:|---:|---:|---:|---:|---:|
| 0.5B | 64 | 0.8 | >20 | >20 | 16 | 8 |
| 0.5B | 64 | 1.0 | >20 | 12 | 8 | 4 |
| 0.5B | 64 | 1.2 | 12 | 4 | 2 | 1 |
| 0.5B | 128 | 0.8 | >20 | >20 | >20 | 16 |
| 0.5B | 128 | 1.0 | >20 | 20 | 12 | 8 |
| 0.5B | 128 | 1.2 | 20 | 8 | 4 | 2 |
| 0.5B | 256 | 0.8 | >20 | >20 | >20 | >20 |
| 0.5B | 256 | 1.0 | >20 | >20 | 20 | 12 |
| 0.5B | 256 | 1.2 | >20 | 12 | 8 | 4 |
| 7B | 64 | 0.8 | >20 | >20 | >20 | >20 |
| 7B | 64 | 1.0 | >20 | >20 | >20 | 16 |
| 7B | 64 | 1.2 | >20 | 12 | 8 | 4 |
| 7B | 128 | 0.8 | >20 | >20 | >20 | >20 |
| 7B | 128 | 1.0 | >20 | >20 | >20 | >20 |
| 7B | 128 | 1.2 | >20 | >20 | 12 | 8 |
| 7B | 256 | 0.8 | >20 | >20 | >20 | >20 |
| 7B | 256 | 1.0 | >20 | >20 | >20 | >20 |
| 7B | 256 | 1.2 | >20 | >20 | >20 | 12 |

### Smallest budget (GB) reaching a 80% hit rate

1000 adapters, uniform tiers; >20 = not reached at 20 GB.

| model | r_max | alpha | vllm | slora | rf_whole | rf_prefix |
|---|---:|---:|---:|---:|---:|---:|
| 0.5B | 64 | 0.8 | >20 | >20 | >20 | >20 |
| 0.5B | 64 | 1.0 | >20 | >20 | >20 | 20 |
| 0.5B | 64 | 1.2 | >20 | 20 | 12 | 8 |
| 0.5B | 128 | 0.8 | >20 | >20 | >20 | >20 |
| 0.5B | 128 | 1.0 | >20 | >20 | >20 | >20 |
| 0.5B | 128 | 1.2 | >20 | >20 | 20 | 12 |
| 0.5B | 256 | 0.8 | >20 | >20 | >20 | >20 |
| 0.5B | 256 | 1.0 | >20 | >20 | >20 | >20 |
| 0.5B | 256 | 1.2 | >20 | >20 | >20 | >20 |
| 7B | 64 | 0.8 | >20 | >20 | >20 | >20 |
| 7B | 64 | 1.0 | >20 | >20 | >20 | >20 |
| 7B | 64 | 1.2 | >20 | >20 | >20 | >20 |
| 7B | 128 | 0.8 | >20 | >20 | >20 | >20 |
| 7B | 128 | 1.0 | >20 | >20 | >20 | >20 |
| 7B | 128 | 1.2 | >20 | >20 | >20 | >20 |
| 7B | 256 | 0.8 | >20 | >20 | >20 | >20 |
| 7B | 256 | 1.0 | >20 | >20 | >20 | >20 |
| 7B | 256 | 1.2 | >20 | >20 | >20 | >20 |
