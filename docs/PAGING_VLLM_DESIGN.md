# Prefix paging in vLLM 0.32: integration design (next milestone)

Status: design only. The paging pieces it builds on exist and are tested outside vLLM
(`rankfuse/paging/model_pages.py`, `rankfuse/paging/bridge.py`, `tests/test_model_pages.py`,
`tests/test_paging_bridge.py`). Measurements: `results/a100_2026-11-23_paging/`. vLLM line numbers
are for the installed 0.32.0 (`env/lib/python3.11/site-packages/vllm/`), as in
`docs/VLLM_LORA_NOTES.md` (cited as NOTES §n).

## 0. What exists and what this milestone adds

* **Stage 1 plugin** (`rankfuse/integration/vllm_plugin.py`). One Matryoshka adapter is loaded at
  `r_max` into one vLLM slot (`max_loras = 1`), and every request carries a rank. Tokens get their
  rank through `LoraState.lora_ranks` and `RankFusePunicaWrapper`. The RF1 kernel replaces vLLM's
  shrink + expand inside `add_lora_linear`.
* **Paging.** `ModelPageManager` owns one fixed GPU pool of whole-model frames
  (`[F, frame_elems]`). A frame holds 16 rank rows of one adapter for every layer and projection.
  * The manager also keeps one int32 `page_table[slots, r_max/16]`, pinned host pages per
    adapter, a copy stream, and the events that order copies against compute.
  * `manager.weights(layer, group)` returns the `PagedWeights` that RF1-MA reads for one
    (layer, q|k|v / o / gate|up / down) block: the same pool tensor, the same table, and different
    offsets.
* **This milestone.** Many adapters (`max_loras` becomes irrelevant), each request at its own rank,
  and weights coming from the paged pool instead of vLLM's per-layer slot buffers. vLLM's
  adapter-slot machinery (`LRUCacheLoRAModelManager.activate_adapter`, per-layer `set_lora`) is
  bypassed for dense layers.

```
engine core                                   worker (GPU process)
-----------                                   --------------------
Scheduler.schedule()                          execute_model()
  admission: page budget instead of max_loras   _set_active_loras()  [patched]
  (sum over adapters in the step of               manager.begin_step(adapters)
   ceil(max rank / 16) <= frames - reserve)       manager.ensure({adapter: max rank in step})
  prefetch hints (waiting queue head)             -> page copies on the copy stream
        |                                         update_metadata(): token ranks, build_meta
        v                                         manager.wait_for_copies()  (compute stream)
SchedulerOutput (+ rankfuse_prefetch)         forward (eager or CUDA graph): every LoRA layer
        |                                         multi_lora(x, manager.weights(l, g), ranks, meta)
        +------------------------------------->  manager.end_step()  (event on compute stream)
                                              manager.prefetch(hinted adapters)  (free/idle frames)
```

## 1. Where the pool is allocated relative to KV-cache sizing

vLLM sizes the KV cache from whatever memory is left after model load and a profiling forward:

* `Worker.determine_available_memory` (`v1/worker/gpu_worker.py:634`) runs
  `model_runner.profile_run()` (`gpu/model_runner.py:991`) and gives the KV cache
  `gpu_memory_utilization * total - (non-KV memory measured after load and profiling)`, unless
  `kv_cache_memory_bytes` fixes it.
* LoRA buffers are already "non-KV memory": `GPUModelRunner.load_model`
  (`gpu/model_runner.py:378`) calls `load_lora_model` (`lora_model_runner_mixin.py:41`), which builds
  the `LoRAModelManager`. `_create_lora_modules` allocates every layer's
  `lora_a_stacked`/`lora_b_stacked` of `max_loras x max_lora_rank` (NOTES §2, §5).

**Design.** The plugin's patched `_create_lora_modules`, which already wraps every dense LoRA layer
(Stage 1), allocates the page pool once for the whole model, right after the layers are wrapped:

* frames = `RANKFUSE_PAGE_POOL_GB * 1e9 // layout.page_bytes()`, or `RANKFUSE_PAGE_FRAMES`;
* the layout comes from the model config: `ModelSpec.transformer(...)` with the model's hidden size,
  intermediate size, heads, KV heads and head dim, merged groups q|k|v and gate|up;
* the pool is allocated before `determine_available_memory` runs, so the KV cache automatically
  gets the rest. No change to the KV allocator, the block manager or `gpu_memory_utilization`
  semantics is needed.

The pool is **separate from the KV cache**, unlike S-LoRA's unified pool. Trading memory between
the two is a later item (§7).

vLLM's own slot buffers must not double the memory. With the plugin active, `create_lora_weights`
is patched to allocate `[1, 1, 16, *]` placeholders (vLLM code still indexes them). `max_lora_rank`
stays at the adapters' `r_max`, because `PEFTHelper.validate_legal` checks it at load time
(NOTES §2). `max_loras` is set to a large value (e.g. 256). It still sizes `LoRAKernelMeta` and the
per-step `lora_index_to_id` list, but no longer limits GPU memory.

**Sizing guidance** comes from the simulator (`results/.../tables/hit_rate.md`).
* At 1000 adapters with Zipf 1.0 and uniform tiers, the hit rate grows roughly logarithmically
  with the budget.
* Prefix paging reaches a given hit rate with less memory than whole-adapter residency (see
  `summary.md`, smallest budget reaching 50% / 80%).
* A sensible default is a few GB: for Qwen2.5-7B, 4 GB holds 49 pages, i.e. 49 rank-16 prefixes or
  12 rank-64 adapters.

## 2. Hook points: `ensure_prefix`, `begin_step`, `end_step`

### 2.1 Registration (CPU, per adapter, once)

* `LRUCacheWorkerLoRAManager.add_adapter` (`lora/worker_manager.py:289`) loads the PEFT
  checkpoint on the CPU (`LoRAModel.from_local_checkpoint`).
* Patched: after `_create_merged_loras_inplace` packs q|k|v and gate|up, the plugin calls
  `ModelPageLayout.pack(weights, rank)` to build the adapter's pinned host pages
  `[ceil(r/16), frame_elems]`, then `manager.register(lora_int_id, host_pages)`.
* Each page is contiguous on the host, so loading it later is one `cudaMemcpyAsync`.
* B must be transposed into rank-major form here: vLLM's `lora_b` is `[out, r]`, while a page
  stores B rows `[16, n_s]`.
* `optimize()` folds the scaling into B. That is fine for `alpha == r`. Fixed-alpha Matryoshka
  adapters keep B unscaled and pass the per-rank `scale_table` to `manager.weights(..., scale_table)`
  (NOTES §2).
* `max_cpu_loras` becomes the host-tier capacity in adapters. Pinned host memory is the real limit:
  1000 x Qwen2.5-7B at r 64 is 323 GB. The host tier is therefore itself an LRU over pinned buffers,
  with checkpoints on disk behind it. That is outside the GPU pager, and the scheduler's prefetch
  hints (§3) should reach it first.

### 2.2 Per step (worker)

The worker learns the step's adapters and ranks in
`LoRAModelRunnerMixin._set_active_loras(prompt_mapping, token_mapping, lora_requests)`
(`lora_model_runner_mixin.py:58`). It is called from `execute_model` at `gpu/model_runner.py:1834`,
before the forward and before CUDA-graph dispatch. The plugin already patches this function to carry
token ranks. The paged version:

```python
def _set_active_loras(self, prompt_mapping, token_mapping, lora_requests):
    mgr = self.rankfuse_pager
    need = {}                                   # adapter -> max rank in this step (host data)
    for lora_id, rank in zip(token_mapping.ids, token_mapping.ranks):
        if lora_id > 0:
            need[lora_id] = max(need.get(lora_id, 0), rank)
    mgr.begin_step(need)                        # pins these adapters: never evicted this step
    got = mgr.ensure(need, allow_partial=ALLOW_DEGRADED)    # copies on mgr.copy_stream
    ranks = clamp_token_ranks(token_mapping, got)            # only if degraded serving (section 5)
    slots = [mgr.slot(i) if i > 0 else -1 for i in token_mapping.ids]   # page-table rows
    self.lora_manager.punica_wrapper.update_rankfuse(slots, ranks)       # build_meta + load_
    mgr.wait_for_copies()                       # compute stream waits for copy_done
```

* `update_rankfuse` is the paged replacement of `RankFusePunicaWrapper.update_metadata`. It runs
  `build_meta(slots, ranks, num_adapters=slots_total, r_pad=r_max, capacity=tile_capacity(...))` on
  the host and copies the result into fixed buffers with `MultiMeta.load_`. There is no GPU sync.
* `end_step()` must run after the step's kernels are **enqueued** (not finished). It records the
  event that later evictions of these adapters wait on. The place is the end of `execute_model`,
  after the forward is launched (and after the graph replay), before `sample_tokens`
  (`gpu/model_runner.py:2088`). With async scheduling the next step's `_set_active_loras` may run
  before this step's kernels finish. That is safe, because evictions wait on the events, not on the
  CPU.
* `LRUCacheWorkerLoRAManager.set_active_adapters` / `_apply_adapters`: the check
  "more distinct `lora_int_id` than `lora_slots`" (L274) is removed (patched). `activate_adapter`
  is never called for paged adapters.

### 2.3 Per step (scheduler, engine core process)

* `Scheduler.schedule` (`v1/core/sched/scheduler.py:591`) admits requests subject to
  `len(scheduled_loras) <= max_loras` (L890–951, L1361).
* Paged: replace that test with a **page budget**. Track the step's `need` exactly as the worker
  does. Admit a request only if `sum(ceil(need[a] / 16)) <= frames - reserve`. This is pure host
  arithmetic on request ranks, with no residency state.
* This guarantees that `ensure` never fails for lack of evictable pages: everything else in the pool
  belongs to adapters not in the step. Use `reserve = 0`, or a few frames kept free for prefetch.
* The scheduler does **not** need to know residency for correctness. For overlap it should know
  roughly what is resident (§3).

## 3. Overlapping page loads with steps

The measurements below are on the A100 80GB PCIe (`results/a100_2026-11-23_paging/`).
* Pinned H2D runs at 26.3 GB/s, with t = 14 us + n / 26.28 GB/s. One Qwen2.5-7B page (80.7 MB)
  takes 3.09 ms; one 0.5B page (17.6 MB) takes 0.68 ms. A whole 7B adapter takes 12.3 ms at r64 and
  49.2 ms at r256.
* A copy on its own stream slows a full-model RF1-MA decode step (one CUDA graph, 192–224
  launches) by **1.6–3.2%**, and the copy keeps its full 26.3 GB/s.
* A rank-r request needs `ceil(r/16)` pages, not the whole adapter. For 7B at r_max 256, a rank-16
  request is ready in 3.1 ms against 49.4 ms for the whole adapter (15.8x); rank 64 in 12.4 ms
  (4.0x).
* `ensure()` costs 12 us of host time per step when every page is already resident.

Where the copy time can hide:

1. **Free frames need no fence.** `ensure` into a free frame, or a frame evicted from an adapter
   whose last step has finished, is enqueued with no wait on the compute stream. Only the kernels
   that need the new pages wait (`wait_for_copies`). A request whose adapter is cold therefore
   delays only the step it joins, by the copy time of its missing pages.
2. **Deferred admission (recommended default).**
   * The scheduler admits a request with a cold prefix but keeps it out of the batch for one step,
     a "loading" state.
   * The worker starts its pages (`prefetch`, no pin) during the current step, using free frames
     and idle adapters' pages.
   * The worker reports readiness back in `ModelRunnerOutput` (a new field `rankfuse_ready:
     list[int]`).
   * Other requests keep decoding, so the load is hidden behind one or more decode steps. On the
     A100, RF1-MA alone takes 3.6–10.6 ms of a 7B decode step (16–256 tokens), and the whole step
     is longer; one 7B page takes 3.1 ms, so each step hides at least one page.
   * The cost is at most one step of added latency for the cold request. The win is that warm
     requests never stall behind a cold load.
3. **Scheduler prefetch hints.**
   * `SchedulerOutput` gets `rankfuse_prefetch: list[(lora_int_id, rank)]` for the head of the
     waiting queue.
   * The worker calls `manager.prefetch` for these after `end_step`. It never evicts an adapter of
     the current step, never raises, and goes into free frames first.
   * This is level 3 (progressive loading) of the build plan. `ModelPageManager.prefetch` exists;
     the hints do not.
4. **Progressive rank growth.** Because pages are ordered, a request can start once page 0..k-1
   are in, and the remaining pages of a higher rank can stream behind it (§5, degraded serving).

What must **not** happen: issuing a load on the compute stream (it serializes behind kernels and
blocks the graph), or waiting on the CPU for copies (`copy_done.synchronize()`) in the step path.

## 4. CUDA-graph safety

vLLM captures one graph per token-count bucket. With `cudagraph_specialize_lora` it captures one
per number of active LoRAs (NOTES §3). A captured graph bakes in pointers and launch grids. The
paged design keeps both fixed:

| captured thing | why it stays valid |
|---|---|
| pool pointer (`A_pool = B_pool = frames`), frame stride, per-(layer, group) offsets | allocated once; frames only change contents |
| `page_table` pointer | allocated once; contents rewritten by one H2D copy per `ensure` on the copy stream, ordered before replay by `wait_for_copies` |
| `MultiMeta` tensors (order, groups, tiles) | built with fixed `capacity = tile_capacity(T_bucket, bm, N)` and `r_pad = r_max`; `load_` copies each step's metadata into them |
| launch grid | `(capacity, split_k, n_slices)`: `capacity` fixed per bucket; empty tiles exit at once |
| int32 vs int64 offsets (`OFF64`) | decided from the pool's numel at capture; the pool never changes size |
| token ranks | the wrapper's static int32 buffer (Stage 1) |

Tested: `tests/test_paging_bridge.py::test_cuda_graph_replay_across_residency_changes` (one graph,
4 steps of evictions and reloads) and `tests/test_model_pages.py::test_cuda_graph_whole_model`
(one graph over every (layer, group) of a model).

Rules for the integration:

* **Stream order.**
  * The replay runs on vLLM's compute stream after `wait_for_copies()`, and `end_step()` records
    after the replay.
  * Frames and table entries of an evicted adapter are rewritten only after the copy stream has
    waited for that adapter's last-use event.
  * Without this fence, a load overwrites a frame (and writes -1 into its table entry) that a
    still-running kernel of the previous step reads. A -1 frame index makes the kernel compute a
    negative address. Reproduction: §8.
* **The capture hook** (`maybe_select_dummy_loras`, `lora_model_runner_mixin.py:143`) assigns
  dummy adapters.
  * Under paging they must have valid page-table rows. Reserve one zero frame at pool creation
    and point every dummy adapter's rows at it, as a constant table prefix that is never evicted.
  * Captures then read real memory, and graphs are captured with the full `r_pad = r_max` code
    path.
* **torch.compile.** The kernel is a custom op (`torch.ops.rankfuse.*`, as Stage 1 does). The pool,
  page table and meta tensors are op inputs, never Python ints closed over by traced code. Offsets
  are per-module constants, fixed at load.
  * The plugin's `compute_hash` patch must also hash the page layout and pool size, because they
    change the compiled graph's inputs.
* **No host syncs** inside the step. `ensure` does host bookkeeping, one memcpy of the page table
  into a pinned staging buffer (double-buffered), and async copies. The only host wait is
  `_stage_ev[i].synchronize()` on a staging buffer two `ensure` calls old, which is normally long
  done.

## 5. A request whose pages are not resident: wait, or serve at the resident rank?

Since pages form a prefix, an adapter with k resident pages can serve any rank <= 16k right now.
`ModelPageManager.serveable_rank(a, r)` returns `min(r, resident rank)`, and
`ensure(..., allow_partial=True)` loads as much as fits and returns the rank each adapter can run at.
The simulator counts `partial_hits`: misses where a usable prefix was already resident. These are
the requests that could start immediately at a lower rank.

| policy | behaviour | output quality | latency |
|---|---|---|---|
| **wait** (default) | the request waits (deferred admission, §3) until `ceil(r/16)` pages are resident | exact: identical to an unpaged engine | +copy time of the missing pages, hidden if other requests are running |
| **serve at resident rank** (opt-in per request, `min_rank`) | runs now at `serveable_rank`, rank rises to r as pages arrive | lower-rank Matryoshka model for the first tokens; tokens produced at rank r' < r had K/V computed at r' | no load stall |

Consequences of serving at a degraded rank:
* **KV cache.** K/V of earlier tokens were computed with the rank-r' projections. Later tokens at
  rank r attend to them. That is the same mixed-rank state as a mid-generation rank change, so it
  is acceptable only if the application accepts it.
* **Prefix caching.** Blocks must be hashed with the rank they were *computed* at, not the
  requested rank. The Stage 1 plugin already puts the rank into `_gen_lora_extra_hash_keys`; under
  degraded serving that must be the effective rank per block.
* **Prefill vs decode.** Degrading the prefill changes every prompt K/V. Degrading a few decode
  tokens is milder. Recommendation: allow degraded serving only for decode-time rank *upgrades*
  (the request was admitted at rank r', the client raises it to r), and for requests that set a
  `min_rank` <= resident rank. Otherwise wait.

Default: **wait**. It is exact, the paper's correctness claims hold unchanged, and with deferred
admission the wait overlaps with other requests' decode. Degraded serving is a flag on the request
(`MatryoshkaLoRARequest.min_rank`) plus `allow_partial=True` in `_set_active_loras`.

## 6. Work list for the milestone

1. `rankfuse/integration/paging.py`.
   * Build `ModelSpec` from the HF config.
   * Allocate the `ModelPageManager` in patched `_create_lora_modules`, with a zero frame for dummy
     adapters.
   * Store `(layer, group)` keys on each wrapped layer.
2. Patched `add_adapter`: pack pinned host pages, with B made rank-major, then `register`.
3. Patched `_set_active_loras` / `execute_model`: `begin_step`, `ensure`, `wait_for_copies`,
   `end_step` (§2.2).
4. `RankFusePunicaWrapper.add_lora_linear` calls `multi_lora(x, mgr.weights(l, g), ranks, meta,
   out=y, add=True)`. One meta per step is shared by all layers. It comes from `update_rankfuse`,
   with a fixed capacity per graph bucket.
5. Scheduler: a page-budget admission test instead of `max_loras` (§2.3). Later: deferred
   admission and prefetch hints (§3).
6. Tests:
   * stock-vLLM equality with all tokens at r_max and every adapter resident;
   * equality under forced eviction (pool of a few frames, many adapters);
   * graph-mode replay across residency changes;
   * the eviction-fence test from `tests/test_model_pages.py` run inside the engine.

## 7. Open questions / later

* **Tensor parallelism.** Each TP rank holds its shard of every matrix, so a page becomes 16 rows
  of the local shard, and each rank gets its own pool. Page tables must stay identical across
  ranks: deterministic LRU from the same inputs, or broadcast the decisions from rank 0.
* **Sharing memory with the KV cache** (S-LoRA's unified pool). The pool could grow into free KV
  blocks under low load. That needs vLLM block-manager cooperation, which is not in scope.
* **Pinned host memory budget** and disk-to-host loading for thousands of adapters (§2.1).
* **Eviction policy.** LRU top-page eviction is what the simulator evaluates. Alternatives: popularity-
  or rank-aware eviction (evict the top page of an adapter whose requests rarely use high ranks)
  and protection of page 0 of hot adapters. The simulator can evaluate these first.

## 8. The stream-ordering race in `AdapterPageManager` (for Anagha)

`rankfuse/paging/adapter_page_manager.py` (origin/main) copies pages on `copy_stream` and offers
`wait_for_copies()` so that compute waits for copies. Nothing orders the other direction.
* `evict()` frees a frame (and writes -1 into its page-table entry on the copy stream) immediately.
* The next `ensure_prefix` overwrites that frame on the copy stream, even if a kernel of the
  previous step, still queued or running on the compute stream, is reading it.
* `end_step()` only clears the in-use set. It records nothing on the GPU.

Minimal reproduction (CUDA; reads a frame on the compute stream after a 0.1 s sleep, while the next
step's load overwrites it):

```python
import torch
from rankfuse.paging import AdapterPageManager, PagePool
pool = PagePool(num_frames=4, h_in=896, h_out=896, dtype=torch.float16, device="cuda")
m = AdapterPageManager(pool, 16, num_adapter_slots=2, r_max=64)
for a in range(2):
    m.register(a, torch.randn(64, 896).half(), torch.randn(64, 896).half())
m.begin_step([0]); m.ensure_resident([0], 64); m.wait_for_copies(); torch.cuda.synchronize()
top = int(m.page_table[0, 3]); expect = pool.frames[top].clone()
torch.cuda._sleep(200_000_000)          # step 1's kernel is still running...
seen = pool.frames[top].clone()         # ...and reads adapter 0's top page afterwards
m.end_step()
m.begin_step([1]); m.ensure_resident([1], 16); torch.cuda.synchronize()   # evicts 0's top page
print(torch.equal(seen, expect))        # False on an H100: the read saw adapter 1's data
```

Fix, as done in `rankfuse/paging/bridge.py::BlockPageManager`:
* `end_step(stream)` records an event on the compute stream;
* `evict` makes `copy_stream` wait for it before freeing.

`ModelPageManager` is finer-grained: one event per step, remembered per adapter, so a load waits
only when it reuses a frame of an adapter whose last step may still be running. Loads into free
frames never wait. Both behaviours are tested (`test_eviction_waits_for_running_step`,
`test_eviction_waits_for_last_use`, `test_load_into_free_frame_does_not_wait`).

A second, smaller point: the stock `register`/`put` interleave A and B per row
(`[16, h_in + h_out]`, row j = A_j | B_j). The RankFuse kernels need A rows contiguous with stride
`h_in`. `BlockPageManager` keeps the frame shape but stores all A rows, then all B rows. The
kernels-v2 session may add an optional `a_ld` to `PagedWeights`, which would let the kernels read
the interleaved layout directly.
