# End-to-end serving benchmark (bench/e2e)

One trace format, one HTTP client and one set of metrics for every serving system: vLLM 0.32
(stock and with RankFuse) now, and SGLang and S-LoRA later.

| file | what it does |
|---|---|
| `gen_trace.py` | writes a unit-rate JSONL trace. Each request has an arrival time, prompt token ids, an output length, an adapter and a rank |
| `client.py` | open-loop streaming replay over OpenAI `/v1/completions` at a sweep of rates. Samples the server's `/metrics` |
| `metrics.py` | throughput, TTFT / TPOT / ITL / E2E percentiles, SLO attainment, goodput |
| `memory.py` | reads adapter and KV memory from vLLM logs, plus an analytic "adapters under a budget" table |
| `run_system.py` | starts one system (`base`, `stock-pad`, `stock-full`, `stock-r<r>`, `rankfuse`), replays traces, stops it |
| `serve_vllm.py` | `vllm serve` with the RankFuse front end installed (stock when `RANKFUSE_VLLM=0`) |
| `summarize.py` | markdown tables from a results folder |
| `jobs/single_adapter.sh` | the Step 3 sweep (single adapter, one GPU) |
| `../../rankfuse/integration/frontend.py`, `rank_policy.py` | the per-request rank over HTTP, and the rank-controller hook |

## Traces

* **Unit rate.** `t` is in seconds at a mean rate of 1 req/s. The client sends request i at
  `t_i / R`. Every rate point therefore replays the same requests with the same burstiness.
* **Same requests in every trace.** The RNG draw order is fixed, so traces built with the same
  seed and length options share arrivals, prompts and lengths, whatever the rank option. Only the
  ranks differ.
* **Arrivals:** `poisson`, `gamma:<cv>`, `onoff:<on>:<off>`, `burst`.
* **Lengths:** `fixed`, `uniform`, `lognormal:<median>:<sigma>:<lo>:<hi>`, or
  `sharegpt:<local json>`. No ShareGPT copy exists on PACE, and none was downloaded.
* **Adapters:** Zipf(a) over N.
* **Ranks:** `fixed:<r>`, `uniform:<list>`, `skewed:<alpha>:<list>` (P ∝ (r/r_min)^-alpha),
  `weights:...`, or `none`. With `none` the server decides.
* **Prompts:** sent as token ids, so prompt lengths are exact on every system.
* **Outputs:** forced with `ignore_eos` + `max_tokens`.

## How a request asks for a rank (vLLM + RankFuse)

```json
POST /v1/completions
{"model": "matryoshka-0", "prompt": [...], "max_tokens": 128,
 "vllm_xargs": {"matryoshka_rank": 32}}
```

The value can be an int, `"auto"` (the rank policy decides) or `"full"`. If the field is
absent, the policy decides; with no policy configured, the request gets the full adapter. With
the OpenAI SDK, pass `extra_body={"vllm_xargs": {"matryoshka_rank": 32}}`.

**Why a request field and not adapter aliases (`name@r32`):**
* `vllm_xargs` already exists on vLLM's completion and chat requests, so no protocol or
  validation patch is needed.
* The model name stays the adapter's identity, which matches the plugin: one `lora_int_id`, one
  CPU cache entry and one GPU slot for every rank.
* Aliases would multiply the model namespace by R. `/v1/models`, routers and per-model metrics
  would all see N×R models, and `_check_model` plus the listing would need patching as well.
* An absent field is the natural hand-off point to the rank controller.

**Mechanism.** `frontend.install()` patches `BaseServing._maybe_get_adapters`. This is the one
call every OpenAI endpoint makes to map `request.model` to a `LoRARequest`. It runs once per
request, before the engine sees the request, and returns a `MatryoshkaLoRARequest` with the
same name, id and path plus `matryoshka_rank`. The existing plugin then carries the rank into
the engine core (msgspec decode), the prefix-cache key and the kernel.

* **The rank is fixed at admission** and never changes afterwards. LoRA on q/k/v (and on any
  earlier layer) makes every K/V entry depend on the rank.
* **Prefix cache.** The key needs the rank, and the Stage 1 plugin already includes it:
  `_gen_lora_extra_hash_keys` appends `("matryoshka_rank", r)`, and every block-hash path
  (`block_pool`, the KV-offload manager) goes through it.
* **Cost of that key.** Requests at different ranks never share prefix blocks, so a system
  prompt is cached once per rank in use. `gen_trace.py --shared-prefix` exists to measure this.
  The Step 3 traces use random prompts, so they have no prefix hits on any system.

## Rank-controller hook (part C)

`RANKFUSE_RANK_POLICY` selects the policy:
* `fixed:<r>`;
* `module:factory` (kwargs from `RANKFUSE_RANK_POLICY_ARGS`);
* unset: no policy.

The policy's `choose_rank(state) -> int | None` gets a `RankState` with these fields:
* the adapter, `r_max` and tiers;
* prompt and output budget;
* engine-core scheduler stats: running, waiting, KV-cache usage and its age;
* the front end's in-flight count;
* EWMAs of TTFT and TPOT;
* the request's other `vllm_xargs`, e.g. a difficulty hint.

None of these needs a GPU sync or an RPC. The scheduler stats are captured from
`StatLoggerManager.record`. The timing EWMAs come from an `AsyncLLM.generate` wrapper, which is
installed only when a policy is set, so the explicit-rank benchmarks run an unmodified front-end
hot path. Explicit ranks always win over the policy.

## Metrics

All metrics are measured at the client, from the HTTP send time.

| metric | definition |
|---|---|
| TTFT | time to the first chunk that carries a token |
| TPOT | (end − first) / (n − 1) |
| ITL | gaps between chunks |
| E2E | send to end of stream |
| req/s, out tok/s, total tok/s | over the span from the first send to the last completion |
| SLO attainment | (TTFT ≤ a AND TPOT ≤ b) over all requests; failed or unfinished requests count as misses |
| goodput | SLO-meeting requests / s |
| client lag | actual minus scheduled send time; p99 > 50 ms flags the run |
| server gauges (`/metrics`, every 1 s) | KV-cache usage, running, waiting, preemptions, prefix hits |
| adapter memory | model-loading memory of a LoRA engine minus the base engine's (from the server log) |
| KV memory, tokens | from the server log |

## Systems (single adapter, Step 3)

| name | engine | requests |
|---|---|---|
| `base` | no LoRA | – |
| `stock-pad` (a) | stock, `max_lora_rank 256`, one adapter per tier (`-r8` … `-r256`), `max_loras 6` | the trace's rank via the tier copy |
| `stock-full` (a′) | stock, `max_lora_rank 256`, one r256 adapter | everyone at 256 |
| `stock-r<r>` (b) | stock, `max_lora_rank r`, the rank-r copy only | everyone at r (fixed-r trace) |
| `rankfuse` (c) | `RANKFUSE_VLLM=1`, one r256 adapter, `max_loras 1` | the trace's rank via `vllm_xargs` |

Engine flags are the same for every system:
`--max-model-len 4096 --max-num-seqs 256 --gpu-memory-utilization 0.85 --seed 0`, with CUDA
graphs on and prefix caching on (the vLLM defaults).

## Later steps (planned, not built)

### Multi-adapter with prefix paging (`wt/paging/docs/PAGING_VLLM_DESIGN.md`)

* **Traces.** The trace format needs no change: `--adapters N --zipf {0.8, 1.0, 1.2}`.
* **Client.** `--adapter-name 'matryoshka-{a}'` already maps adapter ids to names.
* **Adapters.** Generate N synthetic adapters with `make_adapters.py --num-adapters N`. Stock
  baselines use `--no-slices` only where no tier copies are needed.
* **New systems in `run_system.py`:**
  * `rankfuse-paged`: page pool of G GB, `max_loras` large;
  * `stock-pad-multi`: N×T tier slots, LRU over `max_loras` slots, `max_cpu_loras` ≥ N×T;
  * `stock-full-multi`.
* **Extra metrics:**
  * adapter load stalls: scrape the paging manager's counters once it exports them, or log them
    through the plugin;
  * hit rate;
  * the fit table measured, not only analytic.
* **Admission.** The paging design waits for missing pages by default. That interacts with TTFT,
  so report TTFT split into queueing and loading where possible.

### SGLang and S-LoRA baselines (nothing installed)

The details are in the session report. In short:
* **SGLang:** OpenAI `/v1/completions`. `client.py` works as is with `--rank-mode tier-copies`
  and `--lora-paths name=path`.
* **S-LoRA:** its own `/generate_stream` API. This needs a small `SLoRABackend` in `client.py`.
