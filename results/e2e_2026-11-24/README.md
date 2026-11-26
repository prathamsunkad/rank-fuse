# End-to-end single-adapter sweeps, 24 Nov 2026 (first numbers)

| | 0.5B (`q05/`) | 7B (`q7/`) |
|---|---|---|
| GPU | 1× H100 80GB HBM3, job 7288947, node atl1-1-03-012-23-0 | 1× H200 141 GB, job 7288948, node atl1-1-03-017-9-0 |
| model | Qwen2.5-0.5B-Instruct | Qwen2.5-7B-Instruct |

* Every system of a sweep ran in the same job, one server at a time.
* Software:
  * vLLM 0.32.0, torch 2.13.0+cu130, Triton 3.7.1;
  * the worktree `wt/e2e` at 23499b0, plus uncommitted new files: `bench/e2e/*`,
    `rankfuse/integration/{frontend,rank_policy,rank_skip}.py`, and the tests.
* Adapter: synthetic Matryoshka, r_max 256, all 7 projections, fp16, lora_alpha = r, seed 0, in
  `$RF_ADAPTERS/synthetic_r256/<model>`. The tier copies are r8 … r256.
* Engine flags (every system): `--max-model-len 4096 --max-num-seqs 256 --gpu-memory-utilization 0.85 --seed 0`.
  CUDA graphs and prefix caching are on (the defaults). Every prompt is random tokens, so no prefix
  hits occur.
* Traces (`<model>/traces/`, seed 0, 3000 requests): Poisson arrivals at unit rate; prompt length
  lognormal (median 256, σ 0.8, clipped to [16, 2048]); output length lognormal (median 128, σ 0.7,
  clipped to [8, 512]). The `uniform`, `skewed` (α 1), `fixed8`, `fixed64` and `fixed256` traces differ
  only in the rank.
* Client: open loop, streaming, on the same node.
  * Warm-up: 128 requests, 32 at a time, discarded.
  * Per rate: the first requests that arrive within 40 s (0.5B) or 60 s (7B), with 300 at least and
    1500 at most.
  * `inf` sends 1500 requests at once.
  * `--stop-when-saturated` is set.

## Systems

| name | what runs |
|---|---|
| `base` | the model with no adapter |
| `stock-pad` (a) | stock vLLM, `max_lora_rank 256`, one adapter per rank tier, `max_loras 6` |
| `stock-full` (a′) | stock vLLM, one r256 adapter, every request at 256 |
| `stock-r8`, `stock-r64` (b) | stock vLLM at that rank, on the fixed-rank trace |
| `rankfuse` (c) | the Stage 1 plugin with RF1 |
| `rankfuse-skip` (c′) | `RANKFUSE_VLLM_KERNEL=rf2_index` |

## Files

| file | contents |
|---|---|
| `<model>/summary.md` | the tables |
| `summary.jsonl` | the raw summaries |
| `<system>__<trace>_rate<R>.jsonl` | per-request records |
| `*.server.log` | the server logs |
| `*.memory.json` | memory figures parsed from the server logs |
| `driver.log`, `job.log` | the driver output |

**Reproduce:**

```
srun --jobid=<id> --overlap -N1 -n1 --export=ALL,RATES=...,CLIENT_ARGS=... \
    bash bench/e2e/jobs/single_adapter.sh <q05|q7> <out> <systems>
```

The exact `RATES` and `CLIENT_ARGS` are recorded in `job.log`.

## Validity caveats

* **7B RankFuse runs are missing.**
  * What happened: both 7B RankFuse servers failed the pre-run HTTP rank check. Greedy text at rank 8
    was identical to rank 256, so the runner stopped them by design.
  * Unresolved cause, two candidates:
    * the check is too weak for this adapter: a random synthetic adapter on 7B may not flip any greedy
      token in 24 steps;
    * a real problem at 7B: the kernel tests at 7B q|k|v shapes and the 0.5B gate pass, so this is
      less likely.
  * Next step: a check that compares logprobs, not text, and a stock-full vs base control on 7B.
* **The `inf` rows are flagged for client lag.** 1500 requests sent at once lag the client. They still
  measure maximum throughput, but their latency percentiles are queueing-dominated.
* **Most open-loop grids stop below saturation.** On 0.5B only `stock-pad` saturated (at 80 req/s); the
  others were still keeping up at 80. On 7B every system kept up at 16 req/s. Maximum throughput
  therefore comes from the `inf` rows. A finer grid near capacity is needed.
* **Isolated multi-second TTFT p99 outliers in `rankfuse-skip`.** Examples: `fixed64` at 10 req/s, and
  `fixed8` at 80 req/s, where TPOT p99 was 38 ms. These look like Triton compiles of new shapes during
  the run: the skip mode has more compiled variants than the 128-request warm-up covered. RF1 and stock
  rarely show them.
* **Sample sizes.** 300–1500 requests per point and one run each, so p99 is noisy.
* **Results validity.** No request errors occurred, and every request got exactly its output length.

## Left out of git (size and log rules, 2026-11-26)

These stay on PACE scratch, under `~/scratch/rankfuse/wt/e2e/results/e2e_2026-11-24/`:

- per-request records `*__*_rate*.jsonl` and workload traces `traces/*.jsonl`;
- raw correctness output dumps `*_eager.json` / `*_graphs.json` (about 4 MB each);
- server, driver and Slurm logs (`*.log`, `slurm-*.out`).

The summaries (`summary.md`, `correctness_summary.json`, `compare.txt`), memory, commit, GPU and version files are kept.
