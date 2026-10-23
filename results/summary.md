# Laptop baseline metrics

GPU: NVIDIA GeForce RTX 4060 Laptop GPU. All figures state that this is a laptop GPU, not Phoenix H100.

## Decode summary (uniform, h=4096, r_max=64)

| Batch | B1 padded | B1g padded | B2 grouped | B2g grouped | B2/B1 | B2g/B1g |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 246.272 | 16.384 | 305.472 | 16.384 | 1.240 | 1.000 |
| 8 | 243.712 | 35.840 | 612.048 | 70.656 | 2.511 | 1.971 |
| 64 | 244.832 | 33.792 | 813.696 | 93.184 | 3.323 | 2.758 |
| 256 | 278.080 | 50.176 | 831.168 | 87.040 | 2.989 | 1.735 |

## Figure takeaways

- **latency_vs_batch:** Uniform decode latency was measured for all requested batch sizes, with repeat ranges shown as error bars; B2 and B2g share one grouped roofline line.
- **rank_sensitivity:** For B1, the measured all_min/all_max medians were batch 1: all_min 269.952 us, all_max 233.024 us; batch 32: all_min 252.960 us, all_max 233.472 us; batch 256: all_min 242.656 us, all_max 260.096 us, so these data do not show a large rank-distribution reduction in padded latency.
- **speedup_vs_b1:** 68 of 90 like-for-like grouped/B1 cells were slower than their matching B1 variant.
- **launch_overhead:** Profiler measurements show wall time substantially exceeds summed GPU kernel time for eager B1/B2, while graphed variants narrow that gap; grouping and max-rank work are outside the timed graphed calls.
- **roofline_gap:** The two panels aggregate rank distributions at each workload x position with median lines and min-max bands; values below 1 remain unresolved because the analytic model was not changed.
- **graphed variants:** B1g captures the batch max rank, and B2g captures rank grouping, outside the timed replay; this is a best-case launch-overhead comparison.

## Profiling and unresolved roofline observations

- The machine uses the Windows WDDM driver model.
- Below-roofline B2g rows imply 261-349 GB/s versus the 256 GB/s model entry and 226.99 GB/s measured copy bandwidth; this may reflect optimistic byte accounting and/or cache effects despite L2 flush, and is not treated as proof of a physical bandwidth result.

### Profiler rows

| Implementation | Batch | Wall us | Kernel us | Kernel launches |
|---|---:|---:|---:|---:|
| b1_padded | 1 | 288.768 | 13.376 | 7 |
| b1_padded_graphed | 1 | 35.104 | 9.054 | 6 |
| b2_grouped | 1 | 424.640 | 30.945 | 12 |
| b2_grouped_graphed | 1 | 38.656 | 9.760 | 4 |
| b1_padded | 32 | 364.672 | 17.411 | 8 |
| b1_padded_graphed | 32 | 75.232 | 13.632 | 7 |
| b2_grouped | 32 | 1693.312 | 130.111 | 36 |
| b2_grouped_graphed | 32 | 123.296 | 92.864 | 16 |
| b1_padded | 256 | 756.576 | 29.824 | 8 |
| b1_padded_graphed | 256 | 67.456 | 26.241 | 7 |
| b2_grouped | 256 | 2081.568 | 107.936 | 40 |
| b2_grouped_graphed | 256 | 105.664 | 67.840 | 20 |

### Below-roofline rows

| Phase | Implementation | Requests | Tokens | Distribution | Median us | Roofline us | Implied GB/s |
|---|---|---:|---:|---|---:|---:|---:|
| decode | b2_grouped_graphed | 256 | 256 | all_min | 108.544 | 116.800 | 275.5 |
| decode | b2_grouped_graphed | 256 | 256 | all_max | 130.048 | 131.584 | 259.0 |
| decode | b2_grouped_graphed | 256 | 256 | uniform | 142.336 | 145.644 | 261.9 |
| prefill | b2_grouped_graphed | 1 | 512 | all_min | 105.472 | 115.264 | 279.8 |
| prefill | b2_grouped_graphed | 1 | 512 | all_max | 111.616 | 119.296 | 273.6 |
| prefill | b2_grouped_graphed | 1 | 512 | uniform | 106.496 | 115.264 | 277.1 |
| prefill | b2_grouped_graphed | 1 | 512 | skew_low | 105.472 | 115.264 | 279.8 |
| prefill | b2_grouped_graphed | 1 | 512 | skew_high | 105.472 | 119.296 | 289.6 |
| prefill | b2_grouped_graphed | 4 | 512 | all_min | 106.496 | 115.264 | 277.1 |
| prefill | b2_grouped_graphed | 4 | 512 | all_max | 109.568 | 119.296 | 278.7 |
| prefill | b2_grouped_graphed | 4 | 512 | uniform | 101.376 | 115.264 | 291.1 |
| prefill | b2_grouped_graphed | 4 | 512 | skew_low | 106.496 | 115.264 | 277.1 |
| prefill | b2_grouped_graphed | 4 | 512 | skew_high | 111.616 | 116.336 | 266.8 |
| prefill | b2_grouped_graphed | 4 | 2048 | all_min | 388.096 | 459.520 | 303.1 |
| prefill | b2_grouped_graphed | 4 | 2048 | all_max | 406.528 | 464.896 | 292.8 |
| prefill | b2_grouped_graphed | 4 | 2048 | uniform | 392.192 | 459.520 | 299.9 |
| prefill | b2_grouped_graphed | 4 | 2048 | skew_low | 390.144 | 459.520 | 301.5 |
| prefill | b2_grouped_graphed | 4 | 2048 | skew_high | 342.016 | 460.736 | 344.9 |

## Noisy points (max/min > 1.3)
- decode b1_padded requests=1 tokens=1 dist=all_min: 1.39x
- decode b1_padded requests=1 tokens=1 dist=skew_high: 1.39x
- decode b2_grouped requests=2 tokens=2 dist=all_min: 1.33x
- decode b1_padded requests=2 tokens=2 dist=skew_low: 1.69x
- decode b2_grouped requests=2 tokens=2 dist=skew_low: 1.41x
- decode b2_grouped requests=2 tokens=2 dist=skew_high: 1.31x
- decode b1_padded requests=4 tokens=4 dist=all_min: 1.51x
- decode b1_padded requests=4 tokens=4 dist=uniform: 1.45x
- decode b1_padded requests=8 tokens=8 dist=uniform: 1.33x
- decode b1_padded requests=32 tokens=32 dist=skew_low: 1.64x
- decode b1_padded requests=64 tokens=64 dist=uniform: 1.52x
- decode b1_padded requests=128 tokens=128 dist=all_min: 1.35x
- decode b1_padded requests=256 tokens=256 dist=skew_low: 1.34x
- decode b1_padded requests=1 tokens=1 dist=all_max: 1.58x
- decode b1_padded requests=64 tokens=64 dist=all_max: 1.35x
- decode b1_padded requests=64 tokens=64 dist=uniform: 1.86x
- decode b2_grouped requests=64 tokens=64 dist=uniform: 1.30x
- decode b1_padded requests=256 tokens=256 dist=all_min: 1.58x
- decode b1_padded requests=256 tokens=256 dist=all_max: 1.44x
- decode b1_padded requests=256 tokens=256 dist=uniform: 1.44x
- prefill b1_padded requests=1 tokens=128 dist=all_min: 1.70x
- prefill b1_padded requests=1 tokens=128 dist=all_max: 1.74x
- prefill b2_grouped requests=1 tokens=128 dist=all_max: 1.30x
- prefill b1_padded requests=1 tokens=128 dist=uniform: 1.37x
- prefill b2_grouped requests=1 tokens=128 dist=skew_low: 1.39x
- prefill b1_padded requests=1 tokens=512 dist=all_max: 1.63x
- prefill b1_padded requests=1 tokens=512 dist=skew_low: 1.97x
- prefill b2_grouped requests=1 tokens=512 dist=skew_low: 1.40x
- prefill b1_padded requests=1 tokens=512 dist=skew_high: 1.50x
- prefill b1_padded requests=4 tokens=512 dist=all_min: 1.61x
- prefill b1_padded requests=4 tokens=2048 dist=skew_high: 1.36x

## Dropped or skipped workloads

- None recorded by the figure-generation step.

## Regeneration commands

```powershell
python scripts\measure_gpu.py
python bench\run_sweep.py bench/configs/laptop_decode.yaml --out results/raw/laptop_decode_1.csv
python bench\run_sweep.py bench/configs/laptop_decode.yaml --out results/raw/laptop_decode_2.csv
python bench\run_sweep.py bench/configs/laptop_decode.yaml --out results/raw/laptop_decode_3.csv
python bench\run_sweep.py bench/configs/laptop_prefill.yaml --out results/raw/laptop_prefill_1.csv
python bench\run_sweep.py bench/configs/laptop_prefill.yaml --out results/raw/laptop_prefill_2.csv
python bench\run_sweep.py bench/configs/laptop_prefill.yaml --out results/raw/laptop_prefill_3.csv
python analysis\make_slide_figs.py --decode results\raw\laptop_decode_1.csv results\raw\laptop_decode_2.csv results\raw\laptop_decode_3.csv --prefill results\raw\laptop_prefill_1.csv results\raw\laptop_prefill_2.csv results\raw\laptop_prefill_3.csv
```
