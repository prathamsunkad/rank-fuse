# Multi-adapter RankFuse with prefix paging inside vLLM 0.32

Status: working end to end on Qwen2.5-0.5B (worktree `wt/vllm-multi`, detached at `226368e`,
nothing committed). Results, what does not work and the gaps are at the end.

```
RANKFUSE_VLLM=1 RANKFUSE_MULTI=1                     many Matryoshka adapters, vLLM's slots (dense)
RANKFUSE_VLLM=1 RANKFUSE_MULTI=1 RANKFUSE_PAGING=1   many adapters, weights from the prefix-page pool
```

Kernel: `RANKFUSE_MULTI_KERNEL` = `rf1_ma` (default) | `rf1_ma_table` | `rf2_ma_splitk` |
`rf2_ma_splitk_table` | `rf2_ma_widths`, all on kernels-v2's `multi_lora_splitk` (`226368e`).
The kernel is chosen per step and carried in the metadata (`meta_cpu[5]`), so one process can mix
them (the rank-time bench does). The variants differ as follows:
* rf1_ma ("group" tiles): masks rank rows at its tokens' ranks, so it saves bytes.
* rf2_* (RANK_SKIP): also runs each tile at its own width bucket, so it saves MMA work as well.
* *_table: takes the width from the tile table instead of the tokens' ranks.

All of them read ranks and limits from device memory, so lower ranks save time under CUDA-graph
replay too. A new kernel is one entry in `multi_ops._KERNELS`: (metadata mode, rank_skip,
table_width).

Every request carries its own adapter (vLLM's `lora_int_id`) and its own Matryoshka rank
(`MatryoshkaLoRARequest.matryoshka_rank`, None = the adapter's full rank). Every dense LoRA layer
runs RF1-MA (`MULTI_REGISTRY['rf1_ma']`, `rankfuse.kernels.rankfuse_multi.multi_lora`, mode
"group") instead of vLLM's punica shrink + expand. Adapters may have different `r_max` (up to
`max_lora_rank`, at most 256 tested).

## Where things are

| file | what |
|---|---|
| `rankfuse/integration/vllm_multi.py` | the patches (`patch_multi`, called from `vllm_plugin.register()` when `RANKFUSE_MULTI=1`) |
| `rankfuse/integration/multi_punica.py` | `MultiPunicaWrapper`: per-step `MultiMeta` from vLLM's token -> LoRA mapping + token -> rank mapping, static buffers, `step_meta` |
| `rankfuse/integration/multi_ops.py` | `torch.ops.rankfuse.multi_lora_linear` (the custom op every layer calls) |
| `rankfuse/integration/paging.py` | `PluginPageManager` (ModelPageManager + zero row + unregister), `ModuleLayout`, `pack_lora_model`, scheduler page arithmetic |
| `rankfuse/integration/vllm_plugin.py` | 3-line change: `register()` dispatches to `patch_multi()` when `RANKFUSE_MULTI=1`; Stage 1 is unchanged otherwise |
| `rankfuse/paging/{bridge,model_pages}.py`, `docs/PAGING_VLLM_DESIGN.md`, `tests/test_{model_pages,paging_bridge}.py` | **copies** from `wt/paging` (uncommitted there), taken 2026-11-23 21:29 EST, unmodified; sha256 in the table at the end |
| `rankfuse/integration/{frontend,rank_policy}.py` | **copies** from `wt/e2e` (e2e-harness session, uncommitted there), taken 2026-11-23 22:05 EST, unmodified: the OpenAI-server rank field (`vllm_xargs.matryoshka_rank`) and the rank-policy hook |
| `bench/vllm_multi_rank_time.py` | GPU time of one whole-model graph replay as ranks drop (both kernels) |
| `tests/test_vllm_multi.py` | op / graph / pager unit tests (no engine) |
| `bench/vllm_multi_check.py` | end-to-end gate: stock vLLM vs RankFuse dense / paged, per (prompt, adapter, rank) triple |
| `bench/vllm_multi_layer_check.py` | in-engine layer check: every checked LoRA call vs fp32 B0 from the adapters' safetensors |
| `bench/vllm_multi_prefix.py` | is the rank needed in the prefix-cache key? |
| `scripts/ice/vllm_multi.sh` | runs all of the above on one GPU |

## How a step runs

```
engine core (same process as the worker with UniProcExecutor)
  Scheduler._handle_blocked_waiting_request  [paged: page budget, deferred cold requests]
worker
  GPUModelRunner.prepare_inputs              [stash num_tokens_after_padding]
  _set_active_loras                          [patched]
     paged: _apply_adapters (host pages only) -> begin_step -> ensure -> set_adapter_mapping
            -> wait_for_copies -> prefetch hints
     dense: set_active_adapters (vLLM slots, activate_adapter refreshes rank-major copies)
     set_adapter_mapping -> MultiPunicaWrapper.update_metadata: rows, ranks -> MultiMeta -> buffers
  forward (eager / piecewise / full graph): every LoRA layer -> torch.ops.rankfuse.multi_lora_linear
  GPUModelRunner.execute_model returns       [paged: end_step()]
```

## Per-step metadata and CUDA graphs

`MultiPunicaWrapper.update_metadata` gets vLLM's per-token LoRA ids and the plugin's per-token
ranks (Stage 1's `TokenRankMapping` / `LoraState.lora_ranks`, unchanged). It does all of this on
the host:

1. Map each id to a **row**. In dense mode that is vLLM's slot (`lora_index_to_id`). In paged mode
   it is the pager slot; warm-up and capture dummies map to the zero row.
2. Clamp each rank to the adapter's own rank and to `max_lora_rank`. A rank-0 token has no adapter.
3. Pad to the step's **padded** token count T. The stash in `prepare_inputs` provides T; padded
   tokens get rank 0.
4. Run `build_meta(..., mode="group")`.
5. Copy `order`, `tiles` and the token ranks into static device buffers with pinned
   `non_blocking` copies on the compute stream, so they are ordered after the previous step's
   kernels. `(T, cap, bm, r_pad, active)` goes into a CPU tensor that the op reads (not a GPU
   sync).

vLLM records one graph per padded T. A replay re-runs the recorded grid on the recorded
pointers, so the plugin keeps both fixed:

| captured | how it stays valid |
|---|---|
| buffer pointers | `order`, `tiles` and `token_ranks` are allocated once at `max_num_batched_tokens` / `tile_capacity(max_tokens, 16, rows)`; each step's views are prefixes of the same storage |
| grid `(cap, split_k, slices)` | for every T <= the largest capture size, `step_meta` uses `r_pad = max_lora_rank`, `bm = graph_block_m(T)` and `cap = tile_capacity(T, bm, rows)`: functions of T only, so a replay always gets metadata built the way the capture was. Larger T always runs eagerly and gets the step's own `r_pad`, `bm` and tile count |
| weights | dense: the rank-major slot copies never move. Paged: one pool tensor and one page table; only their contents change |
| no-LoRA steps | `active = 0`: the op launches nothing. vLLM captures and replays a separate no-LoRA graph (case 0) for those steps |

`cudagraph_specialize_lora=False` is recommended (the tests set it): RF1-MA does not specialize
on the number of active adapters, so the extra per-count graphs only cost capture time and memory.

## Paging (`RANKFUSE_PAGING=1`)

* **Pool before KV sizing.** The patched `_create_lora_modules` runs inside `load_model`. It
  wraps the layers with `[1, 1, 16, *]` placeholder slot buffers (a config proxy with
  `max_loras = 1` and `max_lora_rank = 16` while vLLM creates them). Then it builds a
  `ModuleLayout` from the wrapped modules and allocates `PluginPageManager(frames + 1 zero frame,
  rows = max_cpu_loras + 1 dummy row)`. `determine_available_memory` runs later, so the KV cache
  gets the rest. Frames: `RANKFUSE_PAGE_FRAMES`, else `RANKFUSE_PAGE_POOL_GB` (default 2) /
  page bytes. For Qwen2.5-0.5B one page is 17.6 MB in bf16.
* **Registration.** `_apply_adapters` loads new adapters into vLLM's CPU LRU (`max_cpu_loras`) but
  never into slots. The patched `LRUCacheLoRAModelManager.add_adapter` packs pinned host pages
  (B made rank-major; `pack_lora_model`), registers them, and drops the LoRAModel's tensors.
  Removal from the CPU LRU (`deactivate_adapter`, `remove_all_adapters`) unregisters. The row goes
  back to -1 through the fenced table copy, and its slot is reused.
* **Per step** (`_set_active_loras`): `begin_step(adapters)`, then `ensure({adapter: max rank in
  step})` (copies on the copy stream), then metadata, then `wait_for_copies()` on the compute
  stream. `end_step()` runs in the wrapped `execute_model`, after the forward (or graph replay) is
  enqueued and before `sample_tokens`.
* **Scheduler.** `Scheduler._handle_blocked_waiting_request` is wrapped. A waiting request is
  admitted only if the sum over the running set plus the request of ceil(max rank per adapter / 16)
  is <= frames - `RANKFUSE_PAGE_RESERVE`. Because of this budget, `ensure` never runs out of
  evictable pages. The adapter's r comes from its `adapter_config.json` (cached), so full-rank
  requests on small adapters are not over-counted. `max_loras` now only bounds the number of
  distinct adapters per step (metadata size, capture dummies).
* **Cold requests wait** (`RANKFUSE_DEFER_COLD=1`, the default): a request is deferred if its
  adapter is registered on the host but not resident to its rank, and other requests are running.
  It never runs at a lower rank. A prefetch hint is served right after the current step's
  `wait_for_copies` (free frames and idle adapters' pages only), so the copy overlaps the running
  step. The request is admitted anyway after `RANKFUSE_DEFER_STEPS` (8) deferrals, or when nothing
  else runs; that step then waits for its copies. A first-ever request for an adapter is not
  deferred: its host pages do not exist yet.

## Rank passing

The e2e-harness session (`wt/e2e`, `rankfuse/integration/frontend.py`, uncommitted) chose a
request field: `vllm_xargs: {"matryoshka_rank": r}` (or `"auto"` / omitted for its rank policy).
The OpenAI front end fixes the rank **at admission**, when it builds the `MatryoshkaLoRARequest`.
This engine side consumes exactly that object, so the two compose with no change. Offline
`LLM.generate` passes `MatryoshkaLoRARequest(..., matryoshka_rank=r)` directly. The rank is read
once per request (`LoraState.add_request`) and never changes for the request's lifetime.

## Prefix cache and the rank

The rank **must** be in the prefix-cache key. LoRA on q/k/v makes every K/V entry depend on the
rank, so a block computed at rank 8 is wrong for a rank-128 request on the same adapter. The
Stage 1 plugin already appends `("matryoshka_rank", r)` to `_gen_lora_extra_hash_keys`, and
multi-adapter mode keeps that. `bench/vllm_multi_prefix.py` shows both sides (results below):
with the rank in the key, a rank-128 request after a rank-8 one gets no cached tokens and matches
a cold run exactly. With the rank removed, it reuses the rank-8 blocks and its first token's
logprob moves by more than 0.1 nats.

Two refinements are not done:
* `rank=None` (full) and `rank=r_max` hash differently although they compute the same thing (a
  missed share only).
* Under degraded serving, which is not implemented, the key would have to be the rank each block
  was computed at.

## Assumptions

* One GPU, tensor parallel 1, model runner V2, UniProcExecutor. The scheduler reads the worker's
  pager in-process (page budget, residency for deferral). With a multiprocess executor the budget
  falls back to `RANKFUSE_PAGE_FRAMES`, and deferral is off.
* Every LoRA module is a dense linear layer of a supported type (true for Qwen2/Llama with the
  usual target modules). Paged mode refuses anything else. Dense mode leaves other modules on
  vLLM's kernels.
* `max_lora_rank` is a multiple of 16 and at most 256 (the kernels' widest bucket).
* Scale: vLLM folds alpha/r into B at load time. Our adapters use alpha = r at every tier, so the
  scale is 1 at every rank (`RANKFUSE_RANK_SCALE=alpha_over_r` covers fixed-alpha adapters as in
  Stage 1).
* `max_loras` must be >= the number of distinct adapters in a step, and `max_cpu_loras` >=
  `max_loras` (vLLM's own check). In paged mode `max_cpu_loras` is also the number of page-table
  rows (registered adapters).
* The adapter's r for the page budget comes from its `adapter_config.json` (the scheduler never
  loads weights). If that file can't be read, the budget counts `max_lora_rank`.
* The test adapters are random, generated at `--init-scale 0.1`. At 0.5 the model became chaotic:
  stock vLLM and RankFuse both sat 0.2-1.2 nats from an HF reference, and no gate means anything
  there.

## Differences from docs/PAGING_VLLM_DESIGN.md

* **Layout from the wrapped modules, not the HF config** (`ModuleLayout`). Same frame format
  (`FrameLayout` blocks, layer-major), but it follows whatever vLLM wrapped.
* **Dummy adapters** share one dummy page-table row whose entries all point at the zero frame, as
  the doc says. Dummy ids are tracked from `add_dummy_lora` and never registered.
* **Pages above the step's need are evictable** (`PluginPageManager._evict_one`). The doc's
  invariant "everything else in the pool belongs to adapters not in the step" is false when an
  in-step adapter still holds pages from an earlier higher-rank request. `ensure` then failed under
  a budget that allowed the step (found in the 20-frame engine run). Trimming those top pages
  restores the guarantee.
* **Compile-cache key** includes the pool settings (`RANKFUSE_PAGE_FRAMES` / `_POOL_GB`,
  `max_cpu_loras`, `max_loras`), as the doc asked. Without that, a 20-frame engine reloaded a
  200-frame engine's AOT graph and failed its shape guard (found the same way).
* **Deferred admission without a worker -> scheduler message.** The doc proposes a
  `rankfuse_ready` field in `ModelRunnerOutput`. Here the scheduler reads residency directly
  (same process), defers at most `RANKFUSE_DEFER_STEPS` times, and posts prefetch hints that the
  worker serves right after the step's `wait_for_copies`.
* **No degraded serving** (`min_rank`, `allow_partial`): cold requests always wait.
* **The LoRAModel's CPU tensors are dropped after packing**, so vLLM's CPU cache holds only
  bookkeeping and the pinned host pages are the one host copy. The doc leaves the host tier
  unspecified.

## Results (2026-11-24, `226368e`, one NVIDIA L40S: correctness only, timings indicative)

Raw files: `results/vllm_multi_2026-11-24/` (`compare.txt`, `ranktime_*.json`, `layer_*.out`,
`prefix_*.json`, per-run logs; `initscale0.5/` and `failed_before_fixes/` hold the runs discussed
below). Setup: Qwen2.5-0.5B-Instruct bf16, 12 Matryoshka adapters with
`r_max = 16, 32, 64 x 6, 128 x 2, 256 x 2` (init-scale 0.1), 72 (prompt, adapter, rank) triples
covering full rank, every tier below r_max and no adapter. `max_lora_rank 256`, CUDA graphs at
1..64 tokens.

### End-to-end gate vs stock vLLM (teacher-forced, `bench/vllm_multi_check.py`)

Stock runs each (adapter, rank) as its per-tier copy (the B0 of an engine). The "all" runs send
the 72 triples in one call; the "waves" runs send shuffled waves of 6.

| engine | mean \|Δlp\| | p99 | decisive top-1 | greedy exact / tie / diverged | no-adapter Δ |
|---|---|---|---|---|---|
| stock control, graphs (waves vs all) | 0.041 | 0.238 | 5607/5607 | 34 / 37 / 1 | n/a |
| stock control, eager | 0.046 | 0.246 | 5607/5607 | 33 / 37 / 2 | n/a |
| dense, graphs, all | 0.040 | 0.229 | 5604/5607 | 41 / 31 / 0 | 0 |
| dense, graphs, waves | 0.043 | 0.248 | 5607/5607 | 38 / 33 / 1 | 0.032 (batching) |
| dense, eager, all | 0.042 | 0.231 | 5606/5607 | 36 / 31 / 5 | 0 |
| dense `rf2_ma_splitk` / `rf2_ma_widths`, graphs, all | 0.040 | 0.229 | 5604/5607 | 40 / 32 / 0, 34 / 37 / 1 | 0 |
| paged, 200 frames (no eviction), graphs, all / waves | 0.040 / 0.043 | 0.226 / 0.236 | 5604 / 5607 | 44 / 28 / 0, 41 / 30 / 1 | 0 |
| paged, 200 frames, eager, all / waves | 0.042 / 0.046 | 0.236 / 0.246 | 5606 / 5607 | 36 / 32 / 4, 37 / 31 / 4 | 0 |
| **paged, 20 frames**, graphs, all | 0.040 | 0.231 | 5607/5607 | 40 / 32 / 0 | 0 |
| **paged, 20 frames**, graphs, waves | 0.043 | 0.238 | 5606/5607 | 39 / 32 / 1 | 0.035 (batching) |
| **paged, 20 frames**, eager, all / waves | 0.043 / 0.047 | 0.241 / 0.252 | 5607 / 5607 | 39 / 27 / 6, 38 / 29 / 5 | 0 |

Paging activity in the 20-frame runs (12 adapters whose full pages need 1 to 16 frames each):
* all: 211 page loads, 191 evictions, 2740 budget skips, 215 cold deferrals, 86 prefetches;
* waves: 339 loads, 319 evictions, 277 budget skips, 64 deferrals, 62 prefetches.

In the graphs runs, every decode step replayed a captured graph while pages moved underneath.

**Reading the gate.** Every RankFuse engine is at or below stock's own batch-composition noise
(mean, p99). Tokens with no adapter are bit-identical whenever the batching matches. The strict
Stage 1 rule fails on 1 to 3 "decisive" positions in some runs. Each is a position where the
reference's top-1 leads by 0.5-0.85 nats. With these adapters, stock's own control moves
logprobs by up to 1.15-1.26 nats, so a fixed 0.5-nat threshold is inside the noise. Example:
triple 58, position 34, margin 0.81, RankFuse 0.40 off, control 0.28 off. I did not change the
gate; this is the reading.

**First attempt, kept for the record.** With adapters at init-scale 0.5, stock and RankFuse
disagreed by about 0.65 nats. An independent HF-transformers reference with the deltas taken from
the safetensors showed both engines equally far off it (0.2-1.2 nats, growing with adapter
strength). The model is chaotic at that strength, so the gate measures nothing there.

### Layer check in the engine vs fp32 B0 from the safetensors (`bench/vllm_multi_layer_check.py`)

The check recomputes the kernel's delta from the engine's own metadata and weights for the first
8 LoRA calls of every step, eager mode. B0 is built from the adapter files on disk, not from
slots or pages.

| engine | calls | max rel err (all module types, full and low ranks) | rank-0 rows |
|---|---|---|---|
| dense | 216 | 0.56% | exactly 0 |
| paged, 200 frames | 192 | 0.53% | exactly 0 |
| paged, 20 frames (306 loads, 286 evictions during the check) | 472 | 0.51% | exactly 0 |

### GPU time vs rank under CUDA-graph replay (`bench/vllm_multi_rank_time.py`, T = 64)

One graph is captured over every LoRA layer of the model, reading the page pool. Each step's
tile table and token ranks are reloaded in place, then the graph is replayed (µs per
whole-model replay). Every kernel matches B0 at every rank (max relative error 0.37%).

| shape | kernel | r16 | r32 | r64 | r128 | r256 | r16 / r_max |
|---|---|---|---|---|---|---|---|
| 0.5B (96 layers), r_max 256 | rf1_ma | 1398 | 1559 | 1945 | 2809 | 4645 | 0.30 |
| | rf1_ma_table | 1356 | 1520 | 1909 | 2771 | 4607 | 0.29 |
| | rf2_ma_splitk | 906 | 1120 | 1604 | 2609 | 4636 | **0.20** |
| | rf2_ma_splitk_table | 867 | 1084 | 1565 | 2570 | 4590 | **0.19** |
| | rf2_ma_widths | 1212 | 1377 | 1799 | 2766 | 4675 | 0.26 |
| 7B shapes (112 layers), r_max 128 | rf1_ma | 2396 | 2979 | 4304 | 7229 | n/a | 0.33 |
| | rf2_ma_splitk | 1983 | 2627 | 4108 | 7341 | n/a | **0.27** |
| | rf2_ma_splitk_table | 1938 | 2588 | 4065 | 7297 | n/a | 0.27 |
| | rf2_ma_widths | 2521 | 3045 | 4438 | 7503 | n/a | 0.34 |

With kernels-v2's `047f671`, every kernel reads per-token ranks at replay, so rf1_ma also drops
(bytes). The RF2-MA kernels also skip MMA width. A mixed batch costs about the same as r_max at
T = 64, because every adapter group then contains at least one high-rank token. That is a
property of per-group limits, not of graphs. These timings are from an L40S; the headline timing
belongs on the A100-80GB / H200.

### Prefix cache (`bench/vllm_multi_prefix.py`, dense and paged: both PASS)

| | cached tokens of r128 after r8 | first-token Δlp vs cold r128 | common greedy prefix (of 16) |
|---|---|---|---|
| rank in the key (the plugin) | 0 | 0.0 | 16 |
| rank removed from the key | 96 | first token differs | 0 |
| same-rank repeat (cache hit, rank in key) | 96 | 0.09 | 7 (bf16 tie later) |

## What works / what does not / gaps

Works:
* Many adapters per batch, each request with its own adapter and rank, on any of five RF1-MA /
  RF2-MA variants (`RANKFUSE_MULTI_KERNEL`), with mixed r_max up to 256.
* Eager mode and CUDA graphs (piecewise and full).
* Prefix paging behind vLLM:
  * the pool is allocated before KV sizing;
  * a page budget replaces `max_loras` in the scheduler;
  * cold requests wait (deferral plus prefetch);
  * eviction and reload happen under load, with graph replay staying correct while pages move.
* The rank in the prefix-cache key, shown to be necessary.

Bugs found and fixed during this work, each with a regression test or a run that shows it:
1. **Compile-cache key without the pool size.** A 20-frame engine reloaded a 200-frame graph.
2. **`ensure` failing under a valid budget.** An in-step adapter held pages above its need.
   Fix: trim those pages.
3. **Infinite loop.** The fix for 2 made a prefetch of an in-step adapter above its step rank
   loop. Fix: trim only for the step's own `ensure`.

Not done / gaps:
* **No commits.** Everything is in `wt/vllm-multi` (detached at `226368e`), untracked apart from
  the 3-line `vllm_plugin.py` change.
* **Timing.**
  * Engine step-time numbers with and without paging, and the overlap of page copies with
    compute, were not measured. The only timings are op-level, on an L40S.
  * Copy overlap is designed for (copy stream, per-adapter fences, prefetch) but not measured
    in the engine.
* **Host side.**
  * The first request for an adapter loads it from disk inside the step (vLLM's `_load_adapter`).
    There is no host-tier prefetch.
  * The pinned host pages are capped only by `max_cpu_loras`.
* **Larger models.** Only Qwen2.5-0.5B was run end to end. 7B shapes were run only in the op
  bench.
* **Degraded serving** (serve at the resident rank, `min_rank`) is not implemented. The
  prefix-cache key would then need the effective rank.
* **Kernel coverage.**
  * `rf2_ma_bucketed` is not wired: its launch count varies per step, so one capture can't
    replay it.
  * Mixed-rank batches cost about as much as their highest rank under rf1-style per-group limits.
* **Prefix-cache key.** `rank=None` and `rank=r_max` hash differently (a missed share).
* **Frontend not wired in.** The copied `frontend.py` is not installed automatically; e2e's
  `serve_vllm.py` installs it. The offline `LLM` path uses `MatryoshkaLoRARequest` directly.
* **Gate threshold.** The Stage 1 gate's fixed 0.5-nat threshold is under the noise for strong
  adapters. A noise-scaled threshold would be fairer but was not adopted unilaterally.

## Copied files (sha256 at copy time)

| file | from | sha256 |
|---|---|---|
| rankfuse/paging/bridge.py | wt/paging, 2026-11-23 21:29 | 554ac08b784903941b2284262ed45147feecd7c83a6bce273d9bc99a21ebf0ad |
| rankfuse/paging/model_pages.py | wt/paging | 288bd183b7418975b2404132aaf9140ed03ef143c467d207a9cb7a92828457af |
| docs/PAGING_VLLM_DESIGN.md | wt/paging | b1e85cb309b237262298e75d0870d11321e21beebb955fe0ff9fdeceed132b9c |
| tests/test_model_pages.py | wt/paging | 814ebbcc75efcdf0402e2d907b99e822a50355adbbd08bb227847d2d8fdf1b9e |
| tests/test_paging_bridge.py | wt/paging | e6fe5d2b25cdd9b6820edb5c9b5b56214377ea0158fe9d819f878df227fe7943 |
| rankfuse/integration/frontend.py | wt/e2e, 2026-11-23 22:05 | 379261b1748ab68f95081b85a182df468c359b46b31dc07a972bf73a50993eb8 |
| rankfuse/integration/rank_policy.py | wt/e2e | 7d5f2204ab20a449a8a7104f9ffdc5271c49b9c7d09b5abb6d49c7f9e3cf0e4a |

Run everything (one GPU, H100 / H200 / A100-40GB / L40S, about 1 h):
`bash scripts/ice/vllm_multi.sh <out> unit make step1 step1rf2 layer ranktime paged pagedsmall prefix compare`.
