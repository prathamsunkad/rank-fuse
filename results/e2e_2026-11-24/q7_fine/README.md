# 7B rerun on an H200: log-prob rank check, warm-up grid, finer grid (job 7303413, 25 Nov 2026)

## Setup

* **GPU:** 1× H200, node atl1-1-03-017-16-0. The earlier 7B run (`../q7/`) was on node atl1-1-03-017-9-0.
* **Traces:** the same as `../q7/` (symlinked).
* **Client:**
  * `--warmup-grid`: prompt length {16, 120, 400, 900, 2000} × burst {1, 4, 16, 64} × every rank, plus a
    mixed-rank burst;
  * 45 s windows;
  * rates `inf 8 12 … 48`;
  * `--stop-when-saturated`.
* **Job file:** `bench/e2e/jobs/h200_7b_rerun.sbatch`.
* The 0.5B rank-check confirmation is in `../q05_check/`.

## Log-prob rank check

Mean |d logprob| in nats, from 4 prompts of 96 random tokens scored teacher-forced. RankFuse must
have r8 vs r256 above 0.003 nats, and r256 vs full at or below it.

| model | system | noise (same request twice) | adapter vs base | r8 vs r256 | r256 vs full | pass |
|---|---|---|---|---|---|---|
| 0.5B | RankFuse | 0.000 | 0.244 | 0.237 | 0.000 | yes |
| 0.5B | stock-full (control) | 0.023 | 0.242 | | | |
| 7B | RankFuse RF1 and skip | 0.000 | 0.072 | 0.072 | 0.000 | yes |
| 7B | stock-full (control) | 0.018 | 0.071 | | | |

* **Why the old check failed on 7B.** The synthetic adapter is 3.4× weaker on 7B than on 0.5B (0.07 vs
  0.24 nats). That is too weak to flip any greedy token in 24 steps, so the old greedy-text check was too
  strict. The rank does reach the kernel on 7B.
* **Noise.** RankFuse is deterministic run to run (noise 0). Stock vLLM is not (0.013–0.023 nats), because
  of its split-K atomics.

## Validity: do not compare these absolute numbers with `../q7/`

* **This node was about 3× slower.** The engine config and logs match the earlier run, yet stock-r8 time
  per output token at 8 req/s was 17.6 ms here vs 6.3 ms there, and its max throughput 12.3 vs 42.4 req/s.
  The likely cause is a throttled or degraded GPU on atl1-1-03-017-16-0; it was not verified, because GPU
  clocks were not logged. Only ratios within this run are usable, and only if the slowdown stayed
  constant over the job.
* **The fine grid stopped after one point.** With 45 s windows, req/s is n / (45 s + drain tail), which
  falls below 0.8× the offered rate. `--stop-when-saturated` therefore fired at 8 req/s for every system
  except rank-skip `fixed8`, which reached 12 req/s. That is a harness artifact. The fix is to compare
  the offered rate with n / (last arrival − first arrival), or to use longer windows.
* **Prefix-cache reuse.** Every rate point replays the start of the same trace, with prefix caching on.
  Later rate points therefore hit the prefix cache (19–36% hit rate in the logs). This also applies to
  `../q05/` and `../q7/`, the same way for every system. Fix: `--no-enable-prefix-caching`, or a different
  trace offset per rate.
* **Base** was not rerun on this node.
