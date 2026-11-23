# RankFuse inside vLLM 0.32: Stage 1 (one Matryoshka adapter, mixed ranks), 3 Nov 2026

What was built, how it was checked, and what it measures:

1. a fast launch path for `rankfuse_lora` / `rf1_fused` (eager host cost per call ~70 -> ~24 us; ~14 us
   for the serving launcher; kernels and results unchanged),
2. a `vllm.general_plugins` plugin that runs RankFuse inside vLLM 0.32 when `RANKFUSE_VLLM=1`
   (`rankfuse/integration/`),
3. the P4.6 correctness gate against stock vLLM (passed, CUDA graphs on and off),
4. Level-2 step times (decode and prefill) for base / stock LoRA / RankFuse.

Nothing here is committed.

## Conditions

| | |
|---|---|
| GPU | 1x NVIDIA H200 (132 SMs), PACE ICE `ice-gpu`, node atl1-1-03-017-2-0, Slurm job 6787847 (interactive allocation, commands run with `srun --overlap`) |
| Software | PyTorch 2.13.0+cu130, Triton 3.7.1, vLLM 0.32.0 (model runner V2, default), Python 3.11 |
| Model | Qwen/Qwen2.5-0.5B-Instruct, bf16 in vLLM (its config dtype); the PyTorch harness runs fp16 |
| Adapter | `scripts/make_adapters.py` output under `$RF_ADAPTERS/qwen2.5-0.5b`: `matryoshka-0` (r_max 64, all 7 projections, lora_alpha = r) and the per-tier copies `matryoshka-0-r{1,2,4,...,64}` (each lora_alpha = r, so every rank has scale 1 in both engines) |
| vLLM LoRA config | `enable_lora`, `max_lora_rank=64`; `max_loras=1` (stock rank-64 and RankFuse) or 7 (stock per-tier mix) |
| Code | branch `pratham/rf1-splitk` at e9e10e8 plus uncommitted work (this session's files listed at the end; other sessions' uncommitted files left as they were) |

## 1. Launch path of `rankfuse_lora` / `rf1_fused`

**Where the ~108 us went (before).** `raw/launch_before.txt`, `scripts/launch_before.py`. Per eager call
(qkv layer, T = 8): Triton's Python launcher for the two kernels was ~41% of the profiled time (argument
binding, per-argument specialization, cache-key computation, twice per call); `rankfuse_lora`'s checks ~9%;
the split-K launch configuration ~9%; `torch.empty` for the workspace ~2%; and with `r_limit=None`, the
`aminmax().tolist()` host sync added ~35 us (rf1 ~70 us with `r_limit`, ~106 us without).

**What changed** (`rankfuse/kernels/rankfuse_fused.py`, host code only; no kernel was touched):

* `_SplitKPlan` / `_splitk_plan`: the launch configuration (block sizes, split-K count, grids, constexprs,
  workspace size) is computed once per call shape and cached.
* `_launch`: the first call of a shape goes through Triton's launcher and keeps the compiled kernels; later
  calls launch them directly (`CompiledKernel.run`) with exactly the arguments Triton would bind. The cache
  key is everything Triton specializes on that the plan key does not fix: r_limit's `== 1` / `% 16` class,
  the dtypes of token_ranks and the scale table, and the 16-byte alignment bit of every pointer. If a Triton
  launch hook (a profiler) is installed, the normal launcher is used.
* `_r_pad` replaces `triton.next_power_of_2` on the host (a `constexpr_function` in Triton 3.7, ~2 us).
* `RF1Launcher` (serving form): checks once at construction; `r_limit` required (no host sync); per-shape plans;
  one shared fp32 split-K workspace per device (`reserve_workspace`, sized for `max_tokens` at setup and never
  reallocated during a call, so CUDA graphs stay valid); `token_ranks` may be a longer static buffer (only
  the first T entries are read). The vLLM op uses it.

**Result** (`raw/vllm_rf1_launch.json`, `bench/vllm_rf1_launch.py`; host wall time per call, 300 back-to-back
calls, median of 5; Qwen2.5-0.5B shapes, fp16, r_max 64, add into the base output):

| layer (T = 32) | B1 (all slices) | B1 (one slice) | RF1 before | RF1 now | RF1 now, r_limit=None | RF1Launcher |
|---|---|---|---|---|---|---|
| q\|k\|v (896 -> 896+128+128) | 198.7 | 65.9 | 71.8 | 24.0 | 55.0 | 14.0 |
| o (896 -> 896) | 64.9 | 63.7 | 69.9 | 23.2 | 54.5 | 13.9 |
| gate\|up (896 -> 2x4864) | 133.7 | 65.7 | 70.7 | 24.3 | 56.5 | 14.3 |
| down (4864 -> 896) | 67.5 | 66.2 | 70.5 | 23.9 | 56.4 | 14.4 |

The same holds from T = 1 to 512; at T = 2048 on gate|up the GPU time (40 us) dominates. Graph-replayed GPU
time is identical before and after (e.g. q|k|v T = 32: 5.3 us), and the fast path is bitwise identical to
the original launch for every case in the table (`bitwise_equal_fast_vs_rf1` in the JSON) and in the new
tests. Target met: per-call overhead (24 us, or 14 us for the launcher) is below B1's ~64-66 us per slice.

**Whole model, eager** (`raw/model_forward_before_after.md`, `raw/model_forward_{before,after}.{json,log}`;
`bench/model_forward.py` restricted to none/b1/b3/rf1 by `scripts/model_forward_rf1_launch.py`, the "before"
run loads `rankfuse_fused.py` from git HEAD):

| | decode 1x1 | decode 32x1 | decode 128x1 | prefill 4x512 | prefill 16x128 |
|---|---|---|---|---|---|
| B1, ms | 24.73 | 25.45 | 25.64 | 25.78 | 26.01 |
| RF1 before, ms | 30.81 | 31.38 | 30.97 | 30.74 | 31.16 |
| RF1 now, ms | 20.44 | 20.88 | 21.00 | 21.29 | 21.55 |

(all64 mix.) B1/RF1 over all 24 workloads: **0.78-0.85x before, 1.17-1.26x now**. Graph replay of the same
forward: RF1 now / before = 0.999-1.001 (unchanged). The harness (owned by another session) was not modified;
its `rf.rankfuse_lora` calls simply got cheaper.

## 2. The vLLM plugin

`pyproject.toml` registers `rankfuse = "rankfuse.integration.vllm_plugin:register"` under
`[project.entry-points."vllm.general_plugins"]`. vLLM calls it in every process (front end, engine core,
worker) before building the model. Without `RANKFUSE_VLLM=1` it returns before importing vLLM, and a test
checks that every patch target below is untouched afterwards (`test_flag_off_is_stock_vllm`).

Every vLLM 0.32.0 symbol patched with the flag on (monkeypatching only; site-packages is not edited):

| vLLM symbol | change |
|---|---|
| `config.lora.LoRAConfig.compute_hash` | also hashes `"rankfuse-stage1"` and the source of `rankfuse/integration/*.py` + `rankfuse_fused.py`. It feeds vLLM's torch.compile / AOT cache key, so cached graphs never cross between stock and RankFuse engines (see "bugs found" below) |
| `v1.engine.EngineCoreRequest.__annotations__["lora_request"]` | `MatryoshkaLoRARequest \| None`, so the engine core's msgpack decoder keeps the trailing rank (checked by a round-trip test through `MsgpackEncoder` / `MsgpackDecoder`) |
| `lora.request.LoRARequest` (not replaced) | subclassed as `rankfuse.integration.MatryoshkaLoRARequest` with a trailing `matryoshka_rank: int \| None = None` (msgspec appends subclass fields, so the existing array_like wire format is unchanged). Equality and hash stay on `lora_name` (also across the two classes): every rank of one adapter is one `lora_int_id`, one CPU cache entry and one GPU slot |
| `v1.core.kv_cache_utils._gen_lora_extra_hash_keys` | appends `("matryoshka_rank", r)` to the block-hash key when a rank is set; keys without a rank are the stock ones |
| `v1.worker.gpu.lora_utils.LoraState.__init__` / `add_request` / `make_lora_inputs` | a `lora_ranks` array next to `lora_ids` (FULL_RANK for `matryoshka_rank=None`, 0 without an adapter); `make_lora_inputs` returns the token-id tuple as a `TokenRankMapping` (a tuple subclass carrying the token-rank tuple built with the same `repeat`), so the runner's `_set_active_loras(*lora_inputs)` call is unchanged |
| `v1.worker.lora_model_runner_mixin.LoRAModelRunnerMixin._set_active_loras` | builds a `RankedLoRAMapping` (dataclass subclass of `lora.layers.LoRAMapping` with `index_rank_mapping`); dataclass equality includes the ranks, so `LoRAModelManager.set_adapter_mapping`'s skip-if-unchanged check stays correct. Mappings that arrive without ranks (warm-up, and the CUDA-graph capture hook `maybe_select_dummy_loras`, which goes through this same method) get cyclic rank tiers for adapter tokens |
| `v1.worker.gpu_input_batch.InputBatch.make_lora_inputs` (V1 runner) | raises if a request carries a Matryoshka rank (the V1 runner is not supported) |
| `current_platform`'s class `.get_punica_wrapper` (CUDA) | returns `rankfuse.integration.punica.RankFusePunicaWrapper` (a `PunicaWrapperGPU` subclass) |
| `lora.model_manager.LoRAModelManager._create_lora_modules` | after the stock body: every LoRA linear layer of a supported type (Column/MergedColumn/QKV/MergedQKV/RowParallel/Replicated, TP 1) gets rank-major copies `A [slots, n_slices, r_max, h_in]`, `B [slots, n_slices, r_max, n_max]` (vLLM stores B as `[slot, 1, out, r]`); raises for `max_loras != 1` and fully sharded LoRA; tells the wrapper whether every LoRA module is RankFuse-handled |
| `lora.model_manager.LoRAModelManager.activate_adapter` | after the stock `set_lora` / `reset_lora` loop, refreshes the copies of the filled slot (A copied, B transposed) |
| `lora.layers.base_linear.BaseLinearLayerWithLoRA._apply_lora_to_output` | passes the layer's copies to `punica_wrapper.add_lora_linear(..., rankfuse_weights=...)`; layers without copies call the stock method |

`RankFusePunicaWrapper` (`rankfuse/integration/punica.py`):
* `__init__`: static `_token_lora_ranks` int32 `[max_num_batched_tokens]` on the GPU, `_r_limit_cpu` int32 `[1]`
  on the CPU, optional per-rank `scale_table` (`RANKFUSE_RANK_SCALE=alpha_over_r` gives `r_max / r` on top of
  vLLM's folded `alpha / r_max`; default none, correct for our alpha = r adapters).
* `update_metadata`: stock base metadata, then the ranks: clamp to `max_lora_rank`, 0 for tokens without an
  adapter, one pinned host-to-device `copy_` into the static buffer (zeroing what the previous step wrote past
  this step's tokens, so CUDA-graph padding reads rank 0), and r_limit = max rank computed on the host. When every
  LoRA module is RankFuse-handled the stock `LoRAKernelMeta.prepare_tensors` (torch.unique/sort plus a host sync,
  run inside `gpu_sync_allowed()` by stock vLLM) is skipped, since nothing reads it.
* `add_lora_linear`: with `rankfuse_weights`, one `torch.ops.rankfuse.lora_linear(y, x, A[slot 0], B[slot 0],
  _token_lora_ranks, _r_limit_cpu, output_slices, scale_table, add_inputs, max_num_batched_tokens)`; otherwise
  the stock method. This covers all four Qwen2 LoRA layer types: merged q|k|v (3 slices), o, merged gate|up
  (2 slices), down. `_rankfuse_apply` is the only call site of the kernel: the Stage 2 multi-adapter kernel
  would take `_token_lora_indices` (already filled by the base class) and all slots of the copies instead of
  slot 0.

`torch.ops.rankfuse.lora_linear` (`rankfuse/integration/ops.py`) is registered like vLLM's own ops (Library
define + impl + fake) so Dynamo records one opaque call. It reads r_limit from the CPU tensor at run time (no
GPU sync, nothing baked into the traced graph). Under CUDA-graph capture it launches with r_limit = r_max,
because a graph is replayed for later steps whatever their ranks; ranks are data read from the static buffer at
replay, so replays stay exact. Eager steps and batches above the largest captured size use the step's own
maximum rank.

**No GPU sync** (`raw/sync_check/`, `scripts/sync_check.py`): with `VLLM_GPU_SYNC_CHECK=error` (vLLM's checker,
which arms PyTorch's sync-debug mode after warm-up; PyTorch calls that detector a prototype that may miss some
syncs) both engines, graphs and eager, generate a mixed-rank batch without error.

## 3. Correctness gate (P4.6): passed

`raw/correctness/compare.txt` (full table), `correctness_summary.json`, per-engine JSON and logs;
`bench/vllm_correctness.py`. Four engines: stock and RankFuse, each with CUDA graphs (default) and with
`enforce_eager`. 24 prompts.

**Method, and why not bit-exact or `VLLM_BATCH_INVARIANT=1`.** Stock vLLM's shrink accumulates split-K partials
with atomics and RankFuse sums in a different order, so the two agree to bf16 rounding, not bit for bit. The
base model is not batch-invariant either: the same stock engine gives different logprobs for a prompt alone vs
in a batch (mean |d logprob| 0.033-0.036 nats here), which flips greedy choices at near-ties (logits come out of
the bf16 lm_head in 0.125 steps). Token-for-token greedy equality therefore fails even for stock against itself
(stock control below). `VLLM_BATCH_INVARIANT=1` was not used: it swaps stock vLLM's kernels (the reference would
no longer be the default engine), and RankFuse's split-K configuration depends on the token count. Instead:

* **gate (teacher-forced):** every engine scores the same fixed sequences (prompt + a fixed ~65-token answer) with
  `prompt_logprobs=2`. Pass = (1) the same top-1 token at every decisive position (reference top-1 ahead of its
  top-2 by >= 0.5 nats) and (2) mean and p99 |d logprob| no more than 2x the stock control (same engine,
  batched vs alone).
* **reported:** greedy generation (48 tokens), as exact / near-tie (first differing tokens within 0.25 nats in
  both runs) / diverged sequence counts, next to the same counts for the stock control.

Graphs on (eager is in `compare.txt`; it gives the same picture):

| check | mean \|dlp\| | p99 | decisive top-1 agree | greedy exact/tie/diverged |
|---|---|---|---|---|
| control: stock, adapter batched vs alone | 0.0329 | 0.159 | 1889/1889 | 7/16/1 |
| signal: stock matryoshka-0 vs base / rank 1 vs rank 64 | 0.222 / 0.218 | | | |
| RankFuse engine, no adapter, vs stock base | 0.0000 | 0.000 | 1933/1933 | 24/0/0 |
| every request at rank 64 vs stock `matryoshka-0` | 0.0335 | 0.162 | 1893/1893 | 8/15/1 |
| `matryoshka_rank=None` vs stock `matryoshka-0` | 0.0347 | 0.151 | 1893/1893 | 8/15/1 |
| rank r vs stock `matryoshka-0-r{r}`, r = 1 ... 64 | 0.0336-0.0358 | 0.161-0.188 | 100% each | 0-1 diverged each |
| mixed-rank batch vs the same requests run alone | 0.0325 | 0.157 | 1924/1924 | 12/12/0 |
| mixed-rank batch vs stock per-tier adapters | 0.0293 | 0.155 | 1924/1924 | 13/11/0 |

Eager: the same checks give mean 0.032-0.042 vs a stock control of 0.036, 100% decisive agreement everywhere.
Graphs vs eager inside one engine differ by the same amount for stock (0.043-0.057) and RankFuse (0.043-0.056).
The mixed batch includes requests without an adapter (rank 0) and ran with prefix caching on, after the same
prompts had been run at other ranks, so it also exercises the rank in the block hash. The tier-1 vs tier-64
signal (0.22 nats) is about 7x the noise, so a rank routed to the wrong value would fail the gate.

**Bugs found on the way (both fixed, both worth knowing):**
* *Compile-cache cross-over.* The first gate run passed in eager mode but RankFuse-with-graphs produced the
  same output at every rank (`raw/correctness/attempt1_stale_compile_cache/`). Cause: vLLM's AOT compile cache is
  keyed on the config and model code, not on monkeypatches, so the RankFuse engine reloaded the graph the stock
  engine had compiled minutes earlier ("Directly load AOT compilation"). That graph calls vLLM's own LoRA kernels
  with metadata the RankFuse wrapper no longer builds. Fix: the plugin folds itself into `LoRAConfig.compute_hash`.
  Any vLLM plugin that monkeypatches traced code has this problem.
* *Stale `rankfuse.egg-info`.* A git-ignored `rankfuse.egg-info` from 22 Oct in the repo root has no entry points
  and shadows the installed metadata for any process with the repo root on `sys.path`, so vLLM would silently not
  load the plugin. It was regenerated (`python -c "from setuptools import setup; setup()" egg_info`), and the bench
  scripts now assert the plugin state matches `RANKFUSE_VLLM`.

## 4. Step time (Level 2)

`raw/step_time/summary.md` (tables below), `raw/step_time/step_time.jsonl` (raw), per-engine logs;
`bench/vllm_step_time_rankfuse.py`. Same method as `bench/vllm_step_time.py`: one engine per process, prefix
caching off, wall time of `llm.generate`; decode step = (t(144 tokens) - t(16 tokens)) / 128 with 128-token
prompts, median of 3; prefill = `max_tokens=1`, median of 5 after a warm-up. The share in parentheses is LoRA's
share of the step, (t - base) / t.

Engines: **base** (no LoRA); **stock64** (stock vLLM, one rank-64 adapter, max_loras 1); **stockmix** (stock vLLM,
request i on the per-tier copy `matryoshka-0-r{MIX[i % 8]}`, max_loras 7 = tiers present); **rfmix** (RankFuse, one
adapter, `matryoshka_rank = MIX[i % 8]`); **rf64** (RankFuse, every request at 64). MIX = 64, 1, 2, 4, 8, 16, 32,
64. Decode at batch 1 and the one-request prefill use the first request's rank (64) in every engine; "8 requests"
is the same L prompt tokens split over 8 requests with all tiers present.

#### CUDA graphs on (default)

Decode step time, ms (LoRA share of the step)

| batch | base | stock64 | stockmix | rfmix | rf64 | stock64/rf64 | stockmix/rfmix |
|---|---|---|---|---|---|---|---|
| 1 | 1.264 | 2.516 (50%) | 2.642 (52%) | 1.802 (30%) | 1.807 (30%) | 1.39x | 1.47x |
| 8 | 1.359 | 2.359 (42%) | 4.496 (70%) | 1.897 (28%) | 1.912 (29%) | 1.23x | 2.37x |
| 32 | 1.784 | 2.475 (28%) | 4.364 (59%) | 1.898 (6%) | 1.924 (7%) | 1.29x | 2.30x |
| 128 | 2.988 | 3.207 (7%) | 3.617 (17%) | 3.266 (9%) | 3.095 (3%) | 1.04x | 1.11x |

Prefill, one request of L tokens, ms (LoRA share)

| tokens | base | stock64 | stockmix | rfmix | rf64 | stock64/rf64 | stockmix/rfmix |
|---|---|---|---|---|---|---|---|
| 128 | 5.619 | 5.967 (6%) | 5.855 (4%) | 5.802 (3%) | 5.726 (2%) | 1.04x | 1.01x |
| 512 | 5.699 | 6.310 (10%) | 6.906 (17%) | 5.879 (3%) | 5.796 (2%) | 1.09x | 1.17x |
| 2048 | 9.295 | 25.166 (63%) | 25.409 (63%) | 15.777 (41%) | 13.599 (32%) | 1.85x | 1.61x |
| 8192 | 24.012 | 41.526 (42%) | 49.690 (52%) | 30.823 (22%) | 30.477 (21%) | 1.36x | 1.61x |

Prefill, 8 requests of L/8 tokens (mixed ranks), ms (LoRA share)

| tokens | base | stock64 | stockmix | rfmix | rf64 | stock64/rf64 | stockmix/rfmix |
|---|---|---|---|---|---|---|---|
| 128 | 12.147 | 12.030 (-1%) | 13.589 (11%) | 14.244 (15%) | 11.259 (-8%) | 1.07x | 0.95x |
| 512 | 10.622 | 14.908 (29%) | 13.278 (20%) | 11.797 (10%) | 13.588 (22%) | 1.10x | 1.13x |
| 2048 | 16.271 | 33.908 (52%) | 33.137 (51%) | 20.448 (20%) | 22.094 (26%) | 1.53x | 1.62x |
| 8192 | 24.894 | 61.956 (60%) | 67.865 (63%) | 39.082 (36%) | 38.438 (35%) | 1.61x | 1.74x |


#### CUDA graphs off (enforce_eager)

Decode step time, ms (LoRA share of the step)

| batch | base | stock64 | stockmix | rfmix | rf64 | stock64/rf64 | stockmix/rfmix |
|---|---|---|---|---|---|---|---|
| 1 | 9.076 | 28.543 (68%) | 28.739 (68%) | 15.089 (40%) | 15.158 (40%) | 1.88x | 1.90x |
| 8 | 9.291 | 28.912 (68%) | 28.955 (68%) | 15.463 (40%) | 15.429 (40%) | 1.87x | 1.87x |
| 32 | 9.595 | 29.170 (67%) | 29.402 (67%) | 15.876 (40%) | 16.034 (40%) | 1.82x | 1.85x |
| 128 | 10.932 | 30.642 (64%) | 30.549 (64%) | 16.943 (35%) | 17.118 (36%) | 1.79x | 1.80x |

Prefill, one request of L tokens, ms (LoRA share)

| tokens | base | stock64 | stockmix | rfmix | rf64 | stock64/rf64 | stockmix/rfmix |
|---|---|---|---|---|---|---|---|
| 128 | 10.370 | 30.046 (65%) | 29.629 (65%) | 16.787 (38%) | 17.418 (40%) | 1.72x | 1.77x |
| 512 | 10.330 | 29.752 (65%) | 30.439 (66%) | 16.327 (37%) | 16.596 (38%) | 1.79x | 1.86x |
| 2048 | 11.581 | 31.600 (63%) | 30.484 (62%) | 16.688 (31%) | 16.837 (31%) | 1.88x | 1.83x |
| 8192 | 24.589 | 41.955 (41%) | 49.693 (51%) | 30.811 (20%) | 30.760 (20%) | 1.36x | 1.61x |

Prefill, 8 requests of L/8 tokens (mixed ranks), ms (LoRA share)

| tokens | base | stock64 | stockmix | rfmix | rf64 | stock64/rf64 | stockmix/rfmix |
|---|---|---|---|---|---|---|---|
| 128 | 21.946 | 61.994 (65%) | 63.669 (66%) | 32.763 (33%) | 32.625 (33%) | 1.90x | 1.94x |
| 512 | 20.084 | 63.600 (68%) | 63.455 (68%) | 35.083 (43%) | 35.415 (43%) | 1.80x | 1.81x |
| 2048 | 20.646 | 62.265 (67%) | 62.592 (67%) | 33.671 (39%) | 35.093 (41%) | 1.77x | 1.86x |
| 8192 | 29.899 | 67.898 (56%) | 73.860 (60%) | 41.962 (29%) | 42.269 (29%) | 1.61x | 1.76x |

Engine startup, s: base/eager 11, base/graphs 30, rf64/eager 12, rf64/graphs 26, rfmix/eager 12, rfmix/graphs 42, stock64/eager 11, stock64/graphs 49, stockmix/eager 11, stockmix/graphs 49

**Reading it.**
* Graphs on, decode: LoRA costs stock vLLM 50/42/28/7% of a step at batch 1/8/32/128 with one rank-64 adapter;
  RankFuse 30/29/7/3% (rf64), i.e. a 1.23-1.39x faster step at batch 1-32. Serving the same rank mix as per-tier
  adapters costs stock much more (52-70% at batch 1-32: 7 adapter slots, the grid covers max_loras + 1 slots);
  RankFuse serves it from one slot at 1.47-2.37x lower step time.
* Graphs on, prefill: up to 512 tokens (vLLM's largest captured graph size here, `max_cudagraph_capture_size`
  512) the step is dominated by fixed per-call cost (stock within ~1.2 ms of base, RankFuse within ~0.2 ms).
  At 2048-8192 tokens (above 512, so these run compiled but not graphed)
  LoRA is 42-63% of the step for stock and 20-41% for RankFuse (1.36-1.85x).
* Graphs off: stock LoRA adds ~19.5 ms to every decode step (64-68% of it): two custom ops per layer, the
  per-step metadata build with its host sync, and per-call Python. RankFuse adds ~6 ms (35-40%), 1.8-1.9x lower.

**Caveats.**
* rfmix is not faster than rf64, by construction. RF1 pads every token to the batch maximum rank (B1's math), and
  every mixed batch of 2+ requests here contains rank 64; under CUDA graphs the captured kernels always run at
  r_limit = r_max (ranks are data at replay). RankFuse's win on mixed ranks comes from one shared slot and one
  launch per layer, not from skipping rank work (that is RF2, not wired into vLLM yet).
* RankFuse also skips stock's `LoRAKernelMeta.prepare_tensors` (torch.unique/sort plus a host sync per step),
  since none of its layers read it; this is part of its advantage, especially eager.
* Wall-clock medians of 3 (decode) or 5 (prefill) runs; the spread was not recorded. Differences below ~0.2 ms
  (e.g. rfmix vs rf64 at decode 128, or the one-request prefill at 2048 where both RankFuse engines run rank 64:
  15.8 vs 13.6 ms) are within run-to-run noise. Small prefills (128 tokens, 8 requests of 16) are dominated by
  per-request overhead and even show negative LoRA shares.
* A 0.5B model has small projections, so per-call cost dominates and LoRA's share is larger than on bigger models.
* Engine startup (s): stock 49 vs RankFuse 26-42 with graphs; RankFuse compiles its own graph (no shared cache
  with stock, by design).

## Tests

`raw/pytest_full.{xml,log}`: 7631 passed, 54 skipped, 0 failed. Per file:

| file | passed | skipped | note |
|---|---|---|---|
| test_correctness.py | 935 | 29 | pre-existing |
| test_kernel_features.py | 69 | 0 | pre-existing |
| test_make_adapters.py | 4 | 0 | pre-existing |
| test_roofline.py | 7 | 0 | pre-existing |
| test_vllm_ops.py | 5 | 0 | pre-existing |
| test_slora_bgmv.py | 511 | 0 | pre-existing (uncommitted, another session) |
| test_multi_correctness.py | 5989 | 24 | new, Stage 2 session |
| test_multi_graphs.py | 51 | 0 | new, Stage 2 session |
| test_vllm_integration_launch.py | 52 | 0 | new, this session |
| test_vllm_integration.py | 8 | 1 | new, this session (the skipped one is the opt-in end-to-end engine test, `RANKFUSE_VLLM_E2E=1`; it passed when run) |

The pre-existing files sum to 1531 passed / 29 skipped, the same as before this work.

## Files

Added (this session):
* `rankfuse/integration/__init__.py`, `ops.py`, `punica.py`, `request.py`, `vllm_plugin.py`
* `bench/vllm_rf1_launch.py`, `bench/vllm_correctness.py`, `bench/vllm_step_time_rankfuse.py`
* `scripts/ice/vllm_stage1.sbatch` (reruns everything in this folder)
* `tests/test_vllm_integration.py`, `tests/test_vllm_integration_launch.py`
* `results/h200_2026-11-03_vllm/` (this README, `raw/`, `scripts/`)

Changed:
* `rankfuse/kernels/rankfuse_fused.py`: host launch path only (`_SplitKPlan`, `_launch`, `_r_pad`,
  `RF1Launcher`, `reserve_workspace`, docstring); kernels and numerics unchanged
* `pyproject.toml`: the entry point
* build artifacts, not in git: the editable install's metadata in the RF env (reinstalled with
  `pip install --no-deps --no-build-isolation -e .` to register the entry point) and the git-ignored
  `rankfuse.egg-info`

## Reproduce

```bash
sbatch scripts/ice/vllm_stage1.sbatch
```
