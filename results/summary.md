# Laptop baseline metrics

GPU: NVIDIA GeForce RTX 4060 Laptop GPU. All figures state that this is a laptop GPU, not Phoenix H100.

## Decode summary (uniform, h=4096, r_max=64)

| Batch | B1 padded (us) | B2 grouped (us) | B2g graphed (us) | B2/B1 | B2g/B1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 238.272 | 302.080 | 16.384 | 1.268 | 0.069 |
| 8 | 245.360 | 642.752 | 71.680 | 2.620 | 0.292 |
| 64 | 252.912 | 895.472 | 93.184 | 3.541 | 0.368 |
| 256 | 229.392 | 774.144 | 87.040 | 3.375 | 0.379 |

## Figure takeaways

- **latency_vs_batch:** Uniform decode latency was measured for all requested batch sizes, with repeat ranges shown as error bars.
- **rank_sensitivity:** For B1, the measured all_min/all_max medians were batch 1: all_min 226.304 us, all_max 225.920 us; batch 32: all_min 240.640 us, all_max 297.680 us; batch 256: all_min 240.320 us, all_max 228.480 us, so these data do not show a large rank-distribution reduction in padded latency.
- **speedup_vs_b1:** 35 of 70 plotted grouped/B1 cells were slower than B1.
- **launch_overhead:** The graphed grouped curve is lower than the eager grouped curve for the plotted uniform decode batches, while B1 remains a separate measured baseline.
- **roofline_gap:** The two panels show measured latency divided by the analytic lower bound for the requested decode and prefill workloads.

## Noisy points (max/min > 1.3)
- decode b1_padded requests=1 tokens=1 dist=all_min: 1.81x
- decode b2_grouped requests=1 tokens=1 dist=all_max: 1.32x
- decode b2_grouped requests=1 tokens=1 dist=uniform: 1.34x
- decode b1_padded requests=1 tokens=1 dist=skew_low: 1.30x
- decode b2_grouped requests=1 tokens=1 dist=skew_low: 1.44x
- decode b1_padded requests=1 tokens=1 dist=skew_high: 1.36x
- decode b2_grouped requests=1 tokens=1 dist=skew_high: 1.44x
- decode b1_padded requests=2 tokens=2 dist=all_min: 1.95x
- decode b2_grouped requests=2 tokens=2 dist=all_min: 1.51x
- decode b1_padded requests=2 tokens=2 dist=all_max: 1.98x
- decode b2_grouped requests=2 tokens=2 dist=all_max: 1.66x
- decode b1_padded requests=2 tokens=2 dist=uniform: 2.01x
- decode b1_padded requests=2 tokens=2 dist=skew_low: 1.32x
- decode b2_grouped requests=2 tokens=2 dist=skew_low: 1.54x
- decode b1_padded requests=4 tokens=4 dist=all_min: 1.55x
- decode b2_grouped requests=8 tokens=8 dist=all_min: 1.46x
- decode b1_padded requests=8 tokens=8 dist=all_max: 1.94x
- decode b1_padded requests=8 tokens=8 dist=uniform: 1.39x
- decode b1_padded requests=8 tokens=8 dist=skew_low: 1.34x
- decode b1_padded requests=16 tokens=16 dist=all_min: 1.44x
- decode b1_padded requests=16 tokens=16 dist=all_max: 1.45x
- decode b1_padded requests=16 tokens=16 dist=uniform: 1.85x
- decode b2_grouped requests=16 tokens=16 dist=uniform: 1.36x
- decode b1_padded requests=16 tokens=16 dist=skew_low: 1.31x
- decode b1_padded requests=16 tokens=16 dist=skew_high: 1.43x
- decode b1_padded requests=32 tokens=32 dist=all_min: 1.44x
- decode b2_grouped requests=32 tokens=32 dist=all_min: 1.51x
- decode b1_padded requests=32 tokens=32 dist=all_max: 1.38x
- decode b2_grouped requests=32 tokens=32 dist=all_max: 1.50x
- decode b1_padded requests=32 tokens=32 dist=uniform: 1.61x
- decode b2_grouped requests=32 tokens=32 dist=uniform: 1.32x
- decode b2_grouped requests=32 tokens=32 dist=skew_low: 1.35x
- decode b1_padded requests=32 tokens=32 dist=skew_high: 1.37x
- decode b2_grouped requests=32 tokens=32 dist=skew_high: 1.42x
- decode b1_padded requests=64 tokens=64 dist=all_min: 1.60x
- decode b2_grouped requests=64 tokens=64 dist=all_min: 1.52x
- decode b1_padded requests=64 tokens=64 dist=all_max: 1.38x
- decode b2_grouped requests=64 tokens=64 dist=all_max: 1.58x
- decode b1_padded requests=64 tokens=64 dist=uniform: 1.48x
- decode b1_padded requests=64 tokens=64 dist=skew_low: 1.87x
- decode b2_grouped requests=64 tokens=64 dist=skew_low: 1.32x
- decode b1_padded requests=64 tokens=64 dist=skew_high: 1.38x
- decode b1_padded requests=128 tokens=128 dist=all_min: 1.32x
- decode b2_grouped requests=128 tokens=128 dist=all_min: 1.47x
- decode b1_padded requests=128 tokens=128 dist=all_max: 1.34x
- decode b2_grouped requests=128 tokens=128 dist=all_max: 1.43x
- decode b1_padded requests=128 tokens=128 dist=uniform: 1.33x
- decode b1_padded requests=128 tokens=128 dist=skew_low: 1.51x
- decode b2_grouped requests=128 tokens=128 dist=skew_low: 1.32x
- decode b1_padded requests=128 tokens=128 dist=skew_high: 1.35x
- decode b2_grouped requests=128 tokens=128 dist=skew_high: 1.42x
- decode b1_padded requests=256 tokens=256 dist=all_min: 1.73x
- decode b2_grouped requests=256 tokens=256 dist=all_min: 1.51x
- decode b1_padded requests=256 tokens=256 dist=all_max: 1.48x
- decode b1_padded requests=256 tokens=256 dist=uniform: 1.73x
- decode b2_grouped requests=256 tokens=256 dist=uniform: 1.31x
- decode b1_padded requests=256 tokens=256 dist=skew_low: 1.72x
- decode b2_grouped requests=256 tokens=256 dist=skew_low: 1.46x
- decode b1_padded requests=256 tokens=256 dist=skew_high: 1.32x
- decode b1_padded requests=1 tokens=1 dist=all_min: 1.56x
- decode b2_grouped requests=1 tokens=1 dist=all_min: 1.33x
- decode b1_padded requests=1 tokens=1 dist=all_max: 1.61x
- decode b2_grouped requests=1 tokens=1 dist=all_max: 1.39x
- decode b2_grouped requests=1 tokens=1 dist=uniform: 1.41x
- decode b1_padded requests=16 tokens=16 dist=all_min: 1.49x
- decode b2_grouped requests=16 tokens=16 dist=all_min: 1.35x
- decode b2_grouped requests=16 tokens=16 dist=all_max: 1.33x
- decode b1_padded requests=64 tokens=64 dist=all_min: 1.31x
- decode b2_grouped requests=64 tokens=64 dist=all_min: 1.35x
- decode b1_padded requests=64 tokens=64 dist=uniform: 1.34x
- decode b2_grouped requests=64 tokens=64 dist=uniform: 1.44x
- decode b1_padded requests=256 tokens=256 dist=all_min: 1.68x
- decode b2_grouped requests=256 tokens=256 dist=all_min: 1.35x
- decode b1_padded requests=256 tokens=256 dist=all_max: 1.80x
- decode b2_grouped requests=256 tokens=256 dist=all_max: 1.38x
- decode b1_padded requests=256 tokens=256 dist=uniform: 1.51x
- decode b2_grouped requests=256 tokens=256 dist=uniform: 1.46x
- prefill b1_padded requests=1 tokens=128 dist=all_min: 1.74x
- prefill b2_grouped requests=1 tokens=128 dist=skew_low: 1.35x
- prefill b1_padded requests=1 tokens=128 dist=skew_high: 1.79x
- prefill b1_padded requests=1 tokens=512 dist=all_min: 1.58x
- prefill b2_grouped requests=1 tokens=512 dist=all_min: 1.31x
- prefill b1_padded requests=1 tokens=512 dist=all_max: 1.43x
- prefill b2_grouped requests=1 tokens=512 dist=all_max: 1.31x
- prefill b1_padded requests=1 tokens=512 dist=uniform: 1.44x
- prefill b1_padded requests=1 tokens=512 dist=skew_low: 1.43x
- prefill b1_padded requests=1 tokens=512 dist=skew_high: 1.88x
- prefill b1_padded requests=4 tokens=512 dist=all_min: 1.89x
- prefill b1_padded requests=4 tokens=512 dist=all_max: 1.93x
- prefill b2_grouped requests=4 tokens=512 dist=uniform: 1.36x
- prefill b1_padded requests=4 tokens=512 dist=skew_low: 1.70x
- prefill b1_padded requests=4 tokens=512 dist=skew_high: 1.57x
- prefill b2_grouped requests=4 tokens=512 dist=skew_high: 1.31x
- prefill b1_padded requests=4 tokens=2048 dist=all_min: 1.53x
- prefill b1_padded requests=4 tokens=2048 dist=skew_low: 1.38x

## Dropped or skipped workloads

- None recorded by the figure-generation step.
- ShareGPT-length trace was skipped: the attempted Hugging Face dataset load failed with `DataFilesNotFoundError`; no fabricated token-length distribution was used.
- The synthetic rank mixes in `rankfuse/workloads.py` remain the analogue of the adapter-spread workloads used in the compared serving papers.

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
