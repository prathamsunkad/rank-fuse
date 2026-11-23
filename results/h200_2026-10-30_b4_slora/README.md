# B4: S-LoRA's bgmv kernel vs RankFuse (H200, 2026-10-30)

Reviewer question: "S-LoRA's MBGMV already batches mixed ranks; is RankFuse just MBGMV with every
request pointing at the same weights?" B4 runs S-LoRA's own kernel (vendored from S-LoRA c1ddf48,
`rankfuse/baselines/slora_ext/`, wrapper `rankfuse/baselines/slora_bgmv.py`) two ways:

- **B4 copies** (`b4_slora_copies`): how S-LoRA serves a Matryoshka adapter today. Each rank tier is
  its own adapter at its true rank; ranks 1/2/4 are stored padded to 8 (the expand loads 8 ranks per
  16-byte vector). That's 144 pool rows per matrix, against RankFuse's 64.
- **B4 shared** (`b4_slora_shared`): the reviewer's hypothetical. A_t and B^T are stored once
  (64 rows each) and every rank points at them. The shrink runs at each token's true rank; the expand
  runs over all 64 columns, because B^T's layout is fixed by the stored rank.

Both were measured against RF1 (`rf1_fused`), B3 (vLLM 0.32's kernels, 7 padded copies) and B1 (PyTorch, padded).
One H200, fp16, r_max 64, graph-replay GPU time (`bench/run_sweep.py --timing graph`, configs
`bench/configs/b4_slora_{decode,prefill}.yaml`), the 4096 x 4096 layer plus Qwen2.5-0.5B's projections.

| file | contents |
|---|---|
| tables/01_b4_decode.md | decode, T = 1, 8, 32, 128, 256 x {uniform, skew_low, all_max} x 5 shapes |
| tables/02_b4_prefill_LABELLED.md | 2048 / 8192-token batches. **Not how S-LoRA runs prompts** (it uses a Triton kernel for >= 200 tokens) |
| tables/03_b4_decode_summary.md | RF1 speedups per shape, geometric mean |
| tables/04_b4_ncu_dram_bytes.md | Nsight Compute DRAM / L2 bytes at T = 32, uniform, 4096 x 4096 |
| source/*.csv, *.json | raw run_sweep rows; ncu CSVs; `build_tables.py` rebuilds the tables |

## Summary

RF1 speedup over each baseline. The figure is the geometric mean over the 5 shapes x 3 rank mixes; the range is min-max.

| tokens | vs B4 copies | vs B4 shared | vs B3 (vLLM) | vs B1 |
|---|---|---|---|---|
| decode 1 | 1.54 (1.20-1.80) | 1.56 (1.18-1.84) | 2.00 | 3.03 |
| decode 8 | 1.54 (1.28-1.77) | 1.53 (1.26-1.74) | 1.98 | 2.87 |
| decode 32 | 1.64 (1.31-2.18) | 1.64 (1.28-2.23) | 1.93 | 2.80 |
| decode 128 | 3.02 (1.80-5.11) | 3.00 (1.77-5.12) | 2.15 | 2.77 |
| decode 256 | 3.73 (2.06-7.52) | 3.69 (2.02-7.50) | 1.71 | 2.15 |
| prompts 2048 (labelled) | 19.2 (11.3-36.0) | 19.2 (11.4-36.1) | 2.83 | 2.44 |
| prompts 8192 (labelled) | 34.7 (18.6-54.8) | 34.8 (18.7-55.0) | 4.18 | 1.52 |

HBM traffic at T = 32 (4096 x 4096, uniform ranks): B3 7.92 MB, B4 copies 2.91 MB, B4 shared 1.60 MB,
RF1 1.46 MB. L2 traffic: B4 shared 22.1 MB against RF1's 5.0 MB.

## Verdict: is RankFuse just MBGMV with shared weights?

No. The B4 shared run gives bgmv exactly the weights RankFuse has, and it does not close the gap:

- **Sharing the weights does not speed bgmv up.** B4 shared is within ±8% of B4 copies everywhere
  (copies/shared time ratio 0.92-1.07). Sharing cuts bgmv's HBM reads almost to RF1's level
  (1.60 vs 2.91 MB; RF1 1.46 MB), yet B4 shared is still 1.2-7.5x slower than RF1 at decode.
  The gap therefore isn't the weight bytes.
- **The bottleneck is how the kernel works.** bgmv runs one matrix-vector product per token: thread
  blocks per call = T x (64 + h_out / 16) (10,240 at T = 32, against RF1's 96). Every token re-reads
  its weight rows from L2: 22 MB of L2 traffic against RF1's 5 MB at T = 32, mostly B4 shared's
  expand, where each token reads all of B^T. Its time grows about linearly with T (4096 x 4096,
  all_max: 10.5 us at T = 1, 79 us at T = 256), while RF1 loads each prefix weight tile once per
  block of tokens and stays at 6-13 us. The gap opens with batch size: 1.5x at T <= 32, 3-3.7x at
  T = 128-256, and 20-35x on prompt-sized batches.
- **Not tensor cores, at decode.** At T = 32 neither kernel is compute-bound: tensor-pipe activity is
  5.7% (RF1 shrink) and 0.9% (RF1 expand), against 0.1-0.9% for bgmv. Prompt-sized batches were not
  profiled, so whether tensor cores explain the prefill gap is not measured.
- **Caveat at small T:** every call returns the delta, so B4 also zeroes the output (about 2 us per call
  under ncu). S-LoRA itself adds into the projection output and would skip that zeroing, so the
  speedups at T <= 32 are somewhat overstated. RF1 needs no zeroing kernels.
- **Where bgmv is competitive:** at T <= 32 it beats vLLM's kernels (B3) on most shapes, because it
  doesn't read 7 padded copies. RF1 is still 1.2-2.2x faster there.

So, to answer the reviewer: MBGMV with shared weights is what B4 shared measures, and RankFuse is
1.5-3.7x faster than it at decode (geometric mean per batch size). The difference is in how the
kernel reuses each prefix tile across tokens, not in where the weights are stored.

## Build (what worked)

- `source scripts/ice/env.sh` (now also sets `TORCH_EXTENSIONS_DIR=$RF_ROOT/cache/torch_extensions`),
  `TORCH_CUDA_ARCH_LIST=9.0`, then any first call (or `slora_bgmv._ext()`) JIT-builds with
  `torch.utils.cpp_extension.load` on an H200 node in about 1 minute.
- On the ice-gpu node, torch found `/usr/local/cuda` (CUDA 13), matching torch's cu130. The `cuda`
  modules (12.x) don't match. If no matching nvcc is found, the wrapper falls back to the pip
  toolkit already in the env (`site-packages/nvidia/cu13`, nvcc 13.4) and symlinks its
  `libcudart.so.13` for `-lcudart`.
- The kernel needs PyTorch's `-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
  -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__` defaults removed, as S-LoRA's
  setup.py does. The wrapper passes `-U` for each (this extension only).
