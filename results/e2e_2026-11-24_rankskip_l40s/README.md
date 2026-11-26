# Rank skipping under CUDA graphs (rf2_index): checks on an L40S, 24 Nov 2026

**GPU:** 1x NVIDIA L40S (46 GB), PACE `coc-gpu`, Slurm job 7284703. These are labelled L40S results: they
check correctness and locate the break-even point. They are not headline numbers.

**Software:** vLLM 0.32.0, torch 2.13.0+cu130, Triton 3.7.1.

**Code:**
* worktree `wt/e2e` at 23499b0, plus new uncommitted files: `rankfuse/integration/rank_skip.py`,
  `frontend.py`, `rank_policy.py`, and `bench/e2e/*`;
* the plugin mode is `RANKFUSE_VLLM_KERNEL=rf2_index`, with `--worker-extension-cls
  rankfuse.integration.rank_skip.RankSkipWorkerExtension`;
* to reproduce: `bash bench/e2e/jobs/rank_skip_checks.sh <out>` inside an allocation.

## 1. Correctness: passed

* **Tests:** `pytest.txt`: 30/30 pass. The run covers `tests/test_rank_skip_launcher.py` (eager, and graph
  replay with ranks and order refilled after capture, compared with RF1) and `tests/test_e2e_harness.py`.
* **Gate:** `correctness/{rf2_index,rf2_segment}/compare.txt`. This is the Stage 1 P4.6 gate (`bench/vllm_correctness.py`,
  Qwen2.5-0.5B, r_max 64 adapters) run in the rank-skip modes against stock vLLM, with CUDA graphs on and
  off. It passed in both modes:
  * decisive top-1 agreement is 100% everywhere;
  * mean |dlogprob| is 0.032–0.041, against a stock control of 0.046;
  * the same holds at every rank 1–64, for the mixed-rank batch, and with no adapter (bitwise 0).

## 2. Graph-replay microbenchmark (`microbench_{q7,q05}.{jsonl,log}`)

**Setup:**
* one CUDA graph of 20 calls per kernel, median of 50 replays;
* bf16, r_max 256, add into the output;
* times are per transformer layer, summed over the 4 LoRA calls (q|k|v, o, gate|up, down);
* the baseline is RF1 at r_limit = r_max, which is what the Stage 1 plugin runs under graphs.

**rf2_index speedup over RF1@r_max, Qwen2.5-7B shapes:**

| T (tokens per step) | 1 | 8 | 32 | 64 | 128 | 256 | 512 | 2048 |
|---|---|---|---|---|---|---|---|---|
| all ranks 256 | 0.95 | 0.94 | 0.94 | 0.96 | 0.97 | 0.96 | 0.98 | 0.96 |
| uniform {8..256} | 1.04 | 0.94 | 1.05 | 1.24 | 1.50 | 1.78 | 1.79 | 1.75 |
| skewed (alpha 1) | 2.30 | 1.47 | 1.88 | 1.32 | 1.75 | 2.35 | 2.20 | 2.01 |
| all rank 8 | 2.36 | 2.50 | 2.96 | 3.41 | 3.80 | 3.92 | 3.54 | 2.41 |
| uniform, 25% no adapter | 1.47 | 1.22 | 1.05 | 1.24 | 1.61 | 2.02 | 1.98 | 2.09 |

**rf2_index speedup over RF1@r_max, Qwen2.5-0.5B shapes:**

| T (tokens per step) | 1 | 8 | 32 | 64 | 128 | 256 | 512 | 2048 |
|---|---|---|---|---|---|---|---|---|
| all ranks 256 | 0.92 | 0.92 | 0.91 | 0.93 | 0.92 | 0.92 | 0.95 | 0.97 |
| uniform | 1.03 | 0.92 | 1.00 | 1.13 | 1.18 | 1.43 | 1.61 | 1.96 |
| skewed | 1.86 | 1.36 | 1.59 | 1.19 | 1.27 | 1.80 | 1.88 | 2.78 |
| all rank 8 | 1.88 | 1.98 | 2.43 | 2.65 | 2.65 | 3.08 | 3.07 | 3.73 |

**When rank skipping wins.** T up to 256 is decode-sized; 512 and 2048 are prefill-sized.

| rank mix | rf2_index wins from |
|---|---|
| uniform | about 32–64 tokens per step, on both models and in both regimes |
| skewed toward low ranks, or a homogeneous low rank | T = 1 |
| every token at r_max | never: it costs 3–8% (the rank read, the branch, and the extra compiled arms) |

**Where RankFuse does not win:**
* **rf2_segment** (no sort) is about equal to RF1 on uniform mixes: a 16-token block almost always holds a
  rank-256 token. The sort is what makes skipping work.
* **Homogeneous low rank.** RF1 compiled for the batch maximum beats rf2_index (7B, T = 1, all rank 8: 16.9
  vs 24.5 µs). That RF1 timing is what eager mode or a per-bucket graph would give, but not what the plugin
  runs under graphs. The cause is that the skip path runs the r_max-width plan (block size, pipeline depth)
  plus a branch. Capturing one graph per rank bucket would recover it; that is not built.

**Host cost per step** (not per layer) of the rank sort and the host-to-device copy:

| T | ≤ 512 | 2048 | 8192 |
|---|---|---|---|
| µs | 17–25 | 64 | 292 |

That is small next to 24–28 layers × 50–2000 µs, and the end-to-end runs include it.

## Left out of git (size and log rules, 2026-11-26)

These stay on PACE scratch, under `~/scratch/rankfuse/wt/e2e/results/e2e_2026-11-24_rankskip_l40s/`:

- per-request records `*__*_rate*.jsonl` and workload traces `traces/*.jsonl`;
- raw correctness output dumps `*_eager.json` / `*_graphs.json` (about 4 MB each);
- server, driver and Slurm logs (`*.log`, `slurm-*.out`).

The summaries (`summary.md`, `correctness_summary.json`, `compare.txt`), memory, commit, GPU and version files are kept.
