# Stage 2 multi-adapter sweeps on A100 80GB PCIe, r_max 64 / 128 / 256 (2026-11-23 to 2026-11-24)

These sweeps measure the current multi-adapter code on the A100 80GB at r_max 64, 128 and 256. They are the main-GPU numbers for the class presentation and the "before" reference for the port to Tony's kernels. All tables are in `ALL_TABLES.md` and `tables/`; rebuild them with `python build_tables.py`.

## Exact conditions

| | |
|---|---|
| GPU | `NVIDIA A100 80GB PCIe` (sm 8.0, 108 SMs, 79.3 GiB, Triton max_shared_mem 166912 B), driver 615.71.09, max clocks 1410 MHz SM / 1512 MHz memory |
| Nodes | PACE ICE atl1-1-01-005-15-0 for r64, r128 and r256; Nsight on atl1-1-01-005-13-0. Partition coc-gpu, account cse, QOS coc-ice, one GPU per job |
| Software | torch 2.13.0+cu130 (CUDA 13.0), Triton 3.7.1, vLLM 0.32.0 (used by B3) |
| Code | branch `pratham/rf1-splitk` at `31adb60`, plus an uncommitted tracked diff with sha256 prefix `b9becdf71d134753` (saved as `source/logs/diff-*.patch`). The untracked files and their hashes are listed in `source/logs/env-*.txt` |
| S-LoRA extension | built for sm_80 and sm_90 (`TORCH_CUDA_ARCH_LIST="8.0;9.0"`) in `cache/torch_extensions_a100_b60417aa` |
| Timing | CUDA-graph replay of one prepared call; metadata built outside timing; L2 flushed before each rep; median of 7 reps × 20 replays, with p20/p80; warmup 10 |
| Seeds | 0, 1, 2. The seed changes the weights, adapter draws and ranks |
| Bandwidth reference | 1.935 TB/s (A100 80GB PCIe spec) for the fraction-of-bandwidth tables |

## Jobs

| Job | What | Run | Result |
|---|---|---|---|
| 7267662 | smoke: 4 test files and a mini-sweep | H100 80GB HBM3 (ice-gpu), 2026-11-24 09:50, 7 min 55 s | passed (see below) |
| 7260646 | sweep, r_max 64 | 10:33, 1h17m | 46,494 rows, 0 errors |
| 7260647 | sweep, r_max 128 | 13:18, 1h40m | 45,234 rows, 0 errors |
| 7260648 | sweep, r_max 256 | 18:27, 2h36m | 45,234 rows, 0 errors |
| 7260649 | Nsight Compute byte counts | 21:03, 16 min | 36 CSVs in `source/ncu/` |

**Smoke results.**
- `test_multi_correctness.py`: 6742 passed, 24 skipped (B4 with merged slices).
- `test_multi_graphs.py`: 56 passed.
- `test_slora_bgmv.py`: 511 passed.
- New `test_multi_rmax.py`: 4166 passed, 192 skipped (B4 at 896→128).

Smoke ran on an H100 to save A100-80GB time, so the A100's 164 KB shared memory and 108 SMs were first exercised by the r64 sweep. All three sweeps completed with no errors.

## Grid (per r_max: 5,670 batches × each implementation)

- **Implementations:**
  - RF1-MA (`rf1_ma`);
  - RF2-MA `rf2_ma_splitk`, `rf2_ma_segment` and `rf2_ma_bucketed`;
  - B1-multi, B3-multi (vLLM 0.32 ops with padded pair copies), B4 copies and B4 shared (S-LoRA bgmv);
  - Stage 1 `stage1_rf1_fused` (1 adapter only).
- **Shapes:**
  - 4096→4096;
  - Qwen2.5-0.5B: q_o 896→896, k_v 896→128, gate_up 896→4864, down 4864→896;
  - Qwen2.5-7B: q7_o 3584→3584, kv7 3584→512, gate_up7 3584→18944, down7 18944→3584.
- **Adapters:** 1, 4, 16, 64, 256. **Popularity:** uniform or Zipf α 1.0.
- **Rank mix:** uniform, skew_low or all_max, with tiers 1, 2, 4, …, r_max.
- **Tokens:** decode at 1, 8, 32, 128 and 256 tokens; prefill 2048 (16 × 128) and 8192 (32 × 256). Pages shuffled, fp16.
- Configs: `bench/configs/multi_a100_r{64,128,256}.yaml` (`order: weights`, so the weights are reused across tokens, popularity and rank mix).

**Caps and skips.**
- No out-of-memory caps were needed for B1 or B3 at 256 adapters × r_max 256.
- B4 can't express a delta width at or above min(h_in, h_out). So the 0.5B k_v shape (896→128) is unsupported for B4 at r_max 128 and 256 (630 rows each), and is recorded as unsupported.
- For B4, the S-LoRA bgmv instantiations gained narrow sizes 128 and 256 (`rankfuse/baselines/slora_ext/bgmv/bgmv_config.h`, marked `RankFuse A100 r_max 128/256`). Stage 1's `slora_bgmv.py` is unchanged.

## Headline (table S4): RF1-MA speedup, geometric mean (min–max); batches where RF1-MA is more than 5% slower

| r_max | phase | vs B3 | vs B4 copies | vs B4 shared | vs B1 |
|---|---|---|---|---|---|
| 64 | decode | 2.52x (0.91–15.9); 6/4050 | 1.83x; 40/4050 | 1.87x; 40/4050 | 9.14x; 0 |
| 64 | prefill | 2.29x (1.23–8.30); 0/1620 | 17.6x; 0 | 17.8x; 0 | 7.07x; 0 |
| 128 | decode | 2.78x (0.49–16.8); 44/4050 | 2.05x; 356/3600 | 2.16x; 332/3600 | 7.07x; 0 |
| 128 | prefill | 2.36x (0.72–14.7); 59/1620 | 21.7x; 0 | 22.1x; 0 | 4.59x; 0 |
| 256 | decode | 2.95x (0.28–21.1); 189/4050 | 2.12x; 446/3600 | 2.34x; 389/3600 | 4.81x; 128/4050 |
| 256 | prefill | 1.85x (0.40–26.2); 307/1620 | 18.4x; 0 | 19.0x; 0 | 2.07x; 96/1620 |

## Where RF1-MA loses (from `raw/`, rf1_ma > 1.05 × baseline)

- **vs B3: wide outputs with few adapters at full rank.**
  - At r_max 256: 499 batches.
  - By shape: gate_up7 211, gate_up 94, 4096² 81, q7_o 55, down7 53.
  - Mostly prefill and decode-128/256, with 1–4 adapters and all_max ranks.
  - Worst case: gate_up7, decode-256, 1 adapter, all_max. RF1-MA takes 205 µs and B3 57 µs. RF2 split-K and bucketed are no better (~205 µs). The best RF2 variant fixes 172 of the 499.
  - At r_max 128 the same pattern covers 105 batches (worst: 90 µs vs 44 µs); at r_max 64 it is negligible (6).
  - Likely cause, not yet checked: the R = 256 tiles shrunk to fit the A100's 164 KB of shared memory, at wide h_out.
- **vs B4: decode-1 and decode-8 at all_max ranks, across every shape.**
  - At r_max 256: 448 batches. At r_max 128: 357 batches, typically RF1-MA ~13 µs vs B4 ~9.5 µs.
  - Worst at r256 is down7 decode-1 (40 µs vs 19 µs).
  - This is a per-call fixed or latency cost, and RF2 fixes only 59.
- **vs B1 at r_max 256:** 228 batches, mostly down7 with 1 adapter. RF2 bucketed fixes 123. For example, gate_up7 decode-256 with skew_low ranks: RF2 bucketed 51 µs vs B1 137 µs vs RF1-MA 204 µs.

## Nsight Compute (table 10): 4096², 16 adapters, uniform ranks, one prepared call

DRAM read MB against ideal MB (the bytes the batch must read):

| r_max | tokens | ideal | RF1-MA | RF2 split-K | RF2 bucketed | B3 | B4 copies | B4 shared |
|---|---|---|---|---|---|---|---|---|
| 64 | 32 | 8.5 | 8.4 | 8.4 | 10.8 | 29.9 | 12.0 | 10.2 |
| 64 | 128 | 16.8 | 16.0 | 16.0 | 21.7 | 79.8 | 26.8 | 17.4 |
| 128 | 32 | 15.2 | 15.1 | 15.1 | 17.0 | 57.2 | 18.1 | 19.3 |
| 128 | 128 | 30.4 | 29.9 | 29.9 | 41.2 | 163.7 | 45.5 | 32.6 |
| 256 | 32 | 29.6 | 29.7 | 29.6 | 33.9 | 118.1 | 34.5 | 38.9 |
| 256 | 128 | 55.6 | 55.8 | 55.5 | 78.6 | 338.0 | 85.4 | 109.1 |

- RF1-MA and RF2 split-K read exactly the ideal bytes, about 1.00×, at every r_max.
- B3 reads 3.5–6.1× the ideal, from its padded pair copies; B4 1.1–2.0×; RF2 bucketed 1.1–1.4×.
- The fraction of 1.935 TB/s reached at 16 adapters on 4096² (table 08b):
  - RF1-MA reaches 34–39% at decode-128/256 for every r_max.
  - In prefill it reaches 53–65% at r64 but only 14–18% at r256. RF2 bucketed reaches 30–31% at r256.

## H200 vs A100 at r_max 64 (tables 11 and 11b; seed 0, the 5 shapes of the H200 run)

- **Decode:** RF1-MA is 1.18× slower on the A100, against a 4.8 / 1.935 = 2.48× bandwidth ratio, so most decode batches are latency-bound. Speedup vs B3 is 2.36× on the A100 vs 2.04× on the H200.
- **Prefill:** RF1-MA is 1.83× slower on the A100. Speedup vs B3 is 2.45× vs 2.84×.

## Files

- `raw/multi-a100-r{64,128,256}.jsonl`: one row per (batch, implementation).
- `source/logs/`:
  - Slurm logs, env dumps with versions and hashes, diff patches;
  - test outputs and junit XML;
  - sweep and Nsight logs.
- `source/smoke-smoke-7267662.jsonl`: the smoke mini-sweep (H100).
- `source/ncu/`: Nsight CSV and `.ncu-rep` files.
- `scripts/`: the sbatch file (`MODE=smoke|sweep|ncu`), `ncu_all.sh` (the full Nsight loop) and `ncu_multi.sh` (one Nsight run).
- `tables/` and `ALL_TABLES.md`, built by `build_tables.py`. Table 12 gives the seed spread: speedups vary by under 1% across seeds.

## Left out of git (size and log rules, 2026-11-26)

These stay on PACE scratch, under `~/scratch/rankfuse/rank-fuse/results/a100_2026-11-23_multi_v1/`:

- `raw/multi-a100-r{64,128,256}.jsonl` (35-37 MB each). The tables here were built from them.
- `source/logs/` (Slurm and pytest logs).
