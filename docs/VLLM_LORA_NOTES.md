# vLLM 0.32.0 LoRA internals: notes for RankFuse (P0.5)

These notes come from reading the vLLM 0.32.0 package that is installed in our environment:

```
/storage/ice1/4/7/psunkad3/rankfuse/env/lib/python3.11/site-packages/vllm/
```

Every path below is relative to that `vllm/` directory. `file.py:N` is the line where the symbol is
defined in 0.32.0, so line numbers will move in other versions but symbol names should not. The
notes cover the dense (non-MoE) path that Qwen2/Llama use. MoE, fp8, multimodal tower/connector
LoRA and classification heads are mentioned only where they affect that path.

Contents:
1. Request type
2. Adapter managers: slots, limits, padding, LRU
3. Batch metadata: per-token indices and CUDA graphs
4. Triton shrink/expand ops
5. LoRA layer classes and Qwen2/Llama wiring
6. What RankFuse changes: Stage 1 (P4) and baseline B3 (P1.3)
7. Pitfalls found while reading

End-to-end flow, one step of decode:

```
LLM.generate(lora_request=...) / OpenAI server
  -> v1 Request.lora_request -> Scheduler (max_loras check) -> NewRequestData.lora_request
  -> worker: LoraState (V2 runner) or InputBatch (V1 runner) keeps per-request lora_int_id
  -> make_lora_inputs(): per-token and per-sample lora_int_id tuples
  -> LoRAModelRunnerMixin._set_active_loras(): LoRAMapping(...)
  -> LRUCacheWorkerLoRAManager.set_active_adapters(): load/activate adapters into GPU slots
  -> LoRAModelManager.set_adapter_mapping() -> PunicaWrapperGPU.update_metadata()
       convert_mapping(): lora_int_id -> slot index;  LoRAKernelMeta.prepare_tensors()
  -> forward: every *WithLoRA layer -> punica_wrapper.add_lora_linear()
       -> lora_shrink (x @ A^T into an fp32 buffer) -> lora_expand (buffer @ B^T added into y)
```

---

## 1. Request type: `LoRARequest`

**`lora/request.py:8` `class LoRARequest(msgspec.Struct, omit_defaults=True, array_like=True)`**

| field | type / default | meaning |
|---|---|---|
| `lora_name` | `str` | name; also the identity used by `__eq__` (L56) and `__hash__` (L63) |
| `lora_int_id` | `int` | must be >= 1 (`__post_init__` L37). Used as the cache and slot key everywhere |
| `lora_path` | `str = ""` | asserted non-empty in `__post_init__` |
| `base_model_name` | `str \| None` | |
| `load_inplace` | `bool = False` | forces a reload under the same id (`LRUCacheWorkerLoRAManager.add_adapter`) |
| `is_3d_lora_weight` | `bool = False` | MoE layout only |

Properties: `adapter_id` returns `lora_int_id`, plus `name` and `path`.

How it travels:
* Entry points: `entrypoints/llm.py:418` `LLM.generate(..., lora_request=...)`; the OpenAI server builds
  one in `entrypoints/openai/models/serving.py:184`; offline helper `entrypoints/common/offline.py:105`.
* Engine side: `v1/request.py:72/89` `Request.lora_request`.
* Scheduler: `v1/core/sched/scheduler.py` around L890 builds `scheduled_loras: set[int]` of
  `lora_int_id` and asserts `len(...) <= max_loras`. Around L941–951 it skips a waiting request whose
  new `lora_int_id` would exceed `max_loras`, and L1361 adds it once scheduled.
* Scheduler to worker: `v1/core/sched/output.py:40` `NewRequestData.lora_request` (L49).
* Prefix-cache key: `v1/core/kv_cache_utils.py:578` `_gen_lora_extra_hash_keys` returns
  `[("lora", lora_name, lora_path)]`. It is mixed into block hashes at L646–657.

**Where a Matryoshka `rank` field goes.** Add it as the last field, with a default, for example
`matryoshka_rank: int | None = None` (None means the full `r_max`). It must be the last field because
the struct is `array_like=True`: it is serialized positionally between engine core and workers, so the
field order is the wire format. Keep `__eq__`/`__hash__` on `lora_name`. That way every rank of one
adapter collapses to one entry in the `set[LoRARequest]` passed to
`WorkerLoRAManager.set_active_adapters`, which gives one slot (P4.2). The scheduler also counts it
once, because it counts `lora_int_id`. Two other places then have to read the rank:
* the per-request worker state (Section 3), and
* `_gen_lora_extra_hash_keys`. Without this change, two ranks of the same adapter share
  prefix-cache KV blocks, which is wrong because the K/V projections differ by rank.

The alternative is no API change: encode the rank in `lora_name` and keep the same `lora_int_id`. It
is fragile, because `__eq__` would then make the ranks different requests while the slot logic keys
on the id.

---

## 2. Adapter managers: slots, limits, padding, LRU

### Config: `config/lora.py:30` `class LoRAConfig`
* `max_lora_rank: MaxLoRARanks = 16`. `MaxLoRARanks = Literal[1, 8, 16, 32, 64, 128, 256, 320, 512]` (L26).
  This is the padded rank of every GPU buffer.
* `max_loras: int = 1` (>= 1): GPU slots, which is the number of distinct adapters in one batch.
* `max_cpu_loras: int | None`: CPU cache capacity. `_validate_lora_config` (L118) sets it to
  `max_loras` when it is None and raises if it is `< max_loras`.
* `fully_sharded_loras`: selects the S-LoRA `*WithShardedLoRA` layers.
* `lora_dtype` is resolved in `verify_with_model_config` (L137).
* `target_modules` restricts which module suffixes get wrapped.
* `specialize_active_lora`: capture CUDA graphs per number of active LoRAs (Section 3).
* `compute_hash` (L90): any new field that changes the compiled graph must be appended to `factors`.
  A RankFuse switch implemented as a `LoRAConfig` field must be added there.

### Rank check at load time
`lora/peft_helper.py:21` `PEFTHelper`:
* `validate_legal` (L188) raises `ValueError` when `max(r, *rank_pattern.values()) > max_lora_rank`.
* `_scaling` (L117) returns `alpha / r`, or `alpha / sqrt(r)` with rsLoRA.

`lora/lora_weights.py:14` `LoRALayerWeights`:
* `lora_a` has shape `[rank, in]` and `lora_b` has shape `[out, rank]` (see
  `create_dummy_lora_weights`, L81).
* `optimize()` (L37) **folds `scaling` into `lora_b`** and sets `scaling = 1`.

Consequently, every layer calls Punica with `scale=1.0` (`base_linear.py:231`).

Our generated adapters (`rankfuse/adapters/qwen2.5-0.5b/matryoshka-0{,-r1..-r64}`) use
`lora_alpha == r` for every copy, so the scale is 1 at every rank. An adapter trained with a fixed
alpha would need `alpha / r_k` per token, which vLLM cannot express. RankFuse would then have to pass
a per-rank scale table, and B would have to be loaded unscaled.

### Worker-side manager: `lora/worker_manager.py`
* `WorkerLoRAManager` (L27) has `_load_adapter` (L106). It builds `PEFTHelper.from_local_dir` and
  calls `validate_legal`, then `LoRAModel.from_local_checkpoint(..., device="cpu", dtype=lora_dtype)`
  (`lora/lora_model.py:197`).
* `set_active_adapters(requests, mapping)` (L195) runs `_apply_adapters(requests)` and then
  `_adapter_manager.set_adapter_mapping(mapping)`.
* **`LRUCacheWorkerLoRAManager`** (L245) is the class the GPU runner actually uses
  (`v1/worker/lora_model_runner_mixin.py:51`).
  * `_apply_adapters` (L274) raises `RuntimeError` if the batch has more distinct `lora_int_id`
    than `lora_slots` (= `max_loras`).
  * `add_adapter` (L289) loads the adapter if it is new or `load_inplace` is set. When
    `len + 1 > capacity` (`max_cpu_loras`) it calls `remove_oldest_adapter()`. It then adds the
    adapter to the CPU cache and always calls `activate_adapter`.

### Model manager: `lora/model_manager.py`
* `AdapterLRUCache` (L62): an `LRUCache` whose `_on_remove` calls a deactivate callback.
* **`LoRAModelManager`** (L73), `__init__` (L76):
  * `_registered_adapters = AdapterLRUCache(capacity=max_cpu_loras, deactivate_adapter)`: the CPU copies.
  * `_active_adapters = AdapterLRUCache(lora_slots=max_loras, _deactivate_adapter)`: the GPU slots.
  * `lora_index_to_id: list[int | None]` of length `max_loras`: slot index to `lora_int_id`.
  * `max_num_batched_tokens` is rounded up to a multiple of 8.
  * `_init_punica_wrapper` (L185) creates one `get_punica_wrapper(max_num_batched_tokens,
    max_batches=max_num_seqs, device, lora_config)` for the language model, keyed
    `DEFAULT_LANGUAGE_WRAPPER_KEY`.
  * `_create_lora_modules` (L467) walks `model.named_modules()`. It filters with
    `is_supported_lora_module` / `target_modules`, looks up `packed_modules_mapping[suffix]`, then
    calls `from_layer(module, lora_slots, lora_config, packed_list, model.config)` and
    `replace_submodule`. It wraps `lm_head` via `from_layer_logits_processor`, calls
    `register_module` and `_register_packed_modules`, and finally
    **`new_module.set_mapping(punica_wrapper)`**. Every layer holds a reference to the same wrapper
    object.
* `add_adapter` (L1345, non-LRU) raises `RuntimeError("No free adapter slots.")` when the CPU cache
  is full. `_add_adapter` (L412) runs `_create_merged_loras_inplace` (L838), which packs
  `q/k/v` and `gate/up` into `PackedLoRALayerWeights.pack` (`lora_weights.py:135`), calls
  `optimize()` and pins memory.
* **`activate_adapter(lora_id)`** (L346) is the CPU-to-GPU copy into a slot. It takes the first
  `None` in `lora_index_to_id`, or raises `ValueError("No free lora slots")`. Then, for each wrapped
  module, it calls `module.set_lora(index, lora_a, lora_b)`, or `module.reset_lora(index)` if the
  adapter has no weights for that module.
* `_deactivate_adapter` (L405) only clears `lora_index_to_id[index]`. The stale weights stay in the
  buffer until the next `set_lora`, which calls `reset_lora` first.
* **`LRUCacheLoRAModelManager`** (L1378):
  * `add_adapter` (L1381) touches the adapter if it is already present.
  * `activate_adapter` (L1393) evicts `_active_adapters.remove_oldest()` (GPU LRU) when all slots
    are full, then calls the base `activate_adapter` and touches the entry.
  * `pin_adapter` (L1413) pins the adapter in both caches.
* `set_adapter_mapping` (L1354) does nothing unless `(mapping, tuple(lora_index_to_id))` changed. If
  it did, `_set_adapter_mapping` (L424) calls
  `punica_wrapper.update_metadata(mapping, lora_index_to_id, lora_slots + 1, vocab_size)`.
* Warmup: `create_dummy_lora` (L636) and `get_dummy_lora_warmup_rank` (L755).

### Padding to `max_lora_rank`
Buffers are allocated once per layer in `create_lora_weights` (Section 5) with the rank dimension
fixed at `max_lora_rank`. `set_lora` first zeroes the slot (`reset_lora`), then copies into the
leading sub-block: `lora_a_stacked[s][index, 0, :r, :in]` and `lora_b_stacked[s][index, 0, :out, :r]`
(`layers/base_linear.py:183–188`, `layers/column_parallel_linear.py:319`). A rank-`r` adapter
therefore sits in a `max_lora_rank` buffer with zeros beyond `r`. The kernels read the full padded
rank (Section 4).

---

## 3. Batch metadata: per-token indices and CUDA graphs

### Which model runner
`config/vllm.py:703` `VllmConfig.use_v2_model_runner`: **Model Runner V2 is the default** when Triton is
present and `_get_v2_model_runner_unsupported_features` (L2914) lists nothing. LoRA is not on that list.
`VLLM_USE_V2_MODEL_RUNNER=0` (`envs.py:311`) forces V1. Both runners mix in
`v1/worker/lora_model_runner_mixin.py:25` `LoRAModelRunnerMixin`.

### Per-request state (where a parallel per-request rank array goes)
* **V2 (default):** `v1/worker/gpu/lora_utils.py:75` `LoraState` has
  `lora_ids = np.zeros(max_num_reqs, int32)` and `lora_requests: dict[req_id, LoRARequest]`.
  * `add_request` (L82) is called from `v1/worker/gpu/model_runner.py:1220`, and
    `remove_request` (L91) from L1149.
  * `make_lora_inputs(req_ids, idx_mapping, num_scheduled_tokens)` (L94) returns
    `prompt_lora_mapping = tuple(lora_ids[idx_mapping])`,
    `token_lora_mapping = tuple(lora_ids.repeat(num_scheduled_tokens))` and the active request set.
  * It is called at `gpu/model_runner.py:~1826` and passed to `self._set_active_loras(*lora_inputs)`
    (L1834).
  * **Add a `lora_ranks` array next to `lora_ids` here**, and return a token-rank tuple built with
    the same `repeat`.
* **V1:** `v1/worker/gpu_input_batch.py`:
  * `request_lora_mapping` (np.int64 `[max_num_reqs]`, L256) is set in `add_request` (L481) and
    cleared in `remove_request` (L545). It is swapped at L649 and moved during condense at L779.
  * `make_lora_inputs` (L996) does the same `repeat`.
  * It is called from `gpu_model_runner.py:~2234` ("Hot-Swap lora model") via `set_active_loras`
    (mixin L83).
  * A rank array here would need the same four update sites: add, remove, swap and condense.

### `LoRAMapping` and `_set_active_loras`
* `lora/layers/utils.py:34` `@dataclass LoRAMapping(index_mapping, prompt_mapping, is_prefill=False,
  type=LoRAMappingType.LANGUAGE)`. `index_mapping` is the per-token `lora_int_id` and
  `prompt_mapping` is per-sampled-token.
* Mixin `_set_active_loras` (L58) builds `LoRAMapping(token, prompt, is_prefill=True)` and calls
  `lora_manager.set_active_adapters(lora_requests, mapping)`.
* A new `index_rank_mapping: tuple[int, ...]` field fits here. Dataclass equality includes it, so the
  `set_adapter_mapping` change-detection cache stays correct.

### Punica wrapper: id to slot, static buffers
* `lora/punica_wrapper/punica_selector.py:13` `get_punica_wrapper` resolves
  `current_platform.get_punica_wrapper()`, which is
  `"vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"` on CUDA (`platforms/cuda.py:575`), and
  logs "Using PunicaWrapperGPU".
* `lora/punica_wrapper/utils.py:52` `convert_mapping(mapping, lora_index_to_id, max_loras, vocab_size,
  device)` runs a Python loop. It maps `lora_int_id` to a slot index (-1 means no LoRA) and returns
  `base_indices` (token to slot), `sampler_indices`, `sampler_indices_padded` and
  `embeddings_indices`, copied host to device with `async_tensor_h2d`.
* `lora/punica_wrapper/punica_base.py:119` `PunicaWrapperBase.__init__` **preallocates static
  device buffers**: `_token_lora_indices`, `_sampler_indices`, `_sampler_indices_padded` and
  `_embeddings_indices` (long, `[max_num_batched_tokens]`), plus `_seq_start_locs`, `_seq_lengths`
  and `_lora_indices_per_batch` (`[max_batches]`). `_update_base_metadata` (L165) `copy_`s each step
  into the prefix of these buffers, so the addresses never change. That is what makes the metadata
  CUDA-graph safe.
* **`lora/punica_wrapper/punica_gpu.py:33` `PunicaWrapperGPU`** (`@final`):
  * `__init__` (L39) builds `token_mapping_meta` and `prompt_mapping_meta` with
    `LoRAKernelMeta.make(max_loras, max_num_batched_tokens, device,
    captured_lora_counts=get_captured_lora_counts(max_loras, specialize_active_lora))`.
  * `update_metadata` (L74) calls `_update_base_metadata`, then
    `token_mapping_meta.prepare_tensors(self.token_lora_indices)` and
    `prompt_mapping_meta.prepare_tensors(self.sampler_indices)` inside `gpu_sync_allowed()`. The
    code comments "TODO avoid gpu<->cpu sync".
  * **A token-to-rank buffer belongs here**: a preallocated `_token_lora_ranks`
    (`[max_num_batched_tokens]`, int32) filled in `update_metadata` the same way. It could also be a
    new tensor inside `LoRAKernelMeta` if the RankFuse op sorts tokens itself.

### `LoRAKernelMeta` (`lora/ops/triton_ops/lora_kernel_metadata.py:12`)
Static tensors from `make` (L47):
* `token_lora_mapping` and `token_indices_sorted_by_lora_ids`: int32 `[max_num_tokens]`.
* `active_lora_ids` and `num_tokens_per_lora`: `[max_loras + 1]`, because -1 (no LoRA) is a
  possible value.
* `lora_token_start_loc`: `[max_loras + 2]`.
* CPU tensors `no_lora_flag_cpu`, `num_active_loras_cpu` and `default_num_active_loras_cpu`
  (`= max_loras + 1`).

`prepare_tensors(token_lora_mapping)` (L107):
1. `_reset`.
2. Set `no_lora` from `all(mapping == -1)` and return early if true.
3. Copy the mapping.
4. A stable `torch.sort` gives `token_indices_sorted_by_lora_ids`.
5. `torch.unique(sorted=True, return_counts=True)` gives `active_lora_ids` and `num_tokens_per_lora`.
6. If capture counts are set, `num_active_loras` is rounded up to the next captured value with
   `bisect`.
7. `cumsum` gives `lora_token_start_loc[1:]`.

`meta_args(token_nums, specialize_active_lora)` (L167) returns the 7-tuple that is splatted into
both ops:
`(token_lora_mapping[:T], token_indices_sorted_by_lora_ids[:T], num_tokens_per_lora,
lora_token_start_loc, active_lora_ids, no_lora_flag_cpu, num_active_loras)`.

The no-LoRA flag and the active count are **CPU tensors on purpose** (comments at L19–39).
torch.compile would constant-fold Python scalars, but it does not trace inside the custom ops, so the
ops can read `.item()` at runtime.

### CUDA-graph capture
* `lora/utils.py:51` `get_captured_lora_counts(max_loras, specialize)` returns `[max_loras + 1]`
  when `specialize` is off. When it is on, it returns the powers of two up to `max_loras` plus
  `max_loras + 1`.
* V2: `gpu/lora_utils.py:20` `get_lora_capture_cases` returns `[0, max_loras + 1]`, or the
  specialized list when `cudagraph_specialize_lora` is set.
  * `get_num_active_loras_for_dispatch` (L38) feeds `dispatch_cg_and_sync_dp(..., num_active_loras)`
    at `gpu/model_runner.py:~1769`.
  * `create_lora_capture_hook` (L52) calls mixin `maybe_select_dummy_loras` (L143) before each
    capture, assigning dummy ids cyclically.
  * `maybe_setup_dummy_loras` (L104) loads `max_loras` dummy adapters of rank `min(8, max_lora_rank)`
    (`gpu/model_runner.py:1075`).
* With the default (no specialization), the kernel grid's third dimension is always
  `max_loras + 1`, so one graph per token-count bucket serves any adapter mix. Tensor addresses are
  fixed because all metadata lives in the preallocated buffers above.

A RankFuse rank buffer must follow the same rules:
* allocate it once;
* `copy_` into its prefix every step;
* never let a Python int that changes per step reach traced code (use a CPU tensor, or read it only
  inside a custom op);
* fill it during the capture hook too.

The capture hook is easy to miss. If it is skipped, graphs are captured with whatever ranks were
left in the buffer.

---

## 4. Triton shrink/expand ops (`lora/ops/triton_ops/`)

`__init__.py` exports `lora_shrink`, `lora_expand`, `LoRAKernelMeta`, plus fp8 and fused-MoE variants.
Both dense ops are registered with `direct_register_custom_op` as `torch.ops.vllm.lora_shrink` /
`torch.ops.vllm.lora_expand` (`mutates_args=["output_tensor"]`). If registration raises
`AttributeError`, the plain Python function is used instead.

### `lora_shrink`: `lora_shrink_op.py:220` `_lora_shrink`
```python
_lora_shrink(
    inputs,                            # [T, hidden] fp16/bf16, made contiguous
    lora_a_weights: list[Tensor],      # one per slice: [num_loras, rank, hidden] or [num_loras, 1, rank, hidden]
    output_tensor,                     # [num_slices, T, rank], contiguous (fp32 from the wrapper)
    token_lora_mapping,                # [T]           -- meta_args() order from here
    token_indices_sorted_by_lora_ids,  # [T]
    num_tokens_per_lora,               # [max_loras + 1]
    lora_token_start_loc,              # [max_loras + 2]
    lora_ids,                          # [max_loras + 1]  (slot indices, -1 = none)
    no_lora_flag_cpu,                  # CPU [1]
    num_active_loras,                  # CPU [1]
    scaling: float,
) -> None
```
* It returns immediately if `T == 0` or `no_lora_flag_cpu.item()` is set.
* It asserts the dtypes and that `inputs.size(1) == A.size(-1)`, then **runs `output_tensor.zero_()`**
  (L296).
* `N, K = A.shape[-2:]`, so **N = buffer rank = `max_lora_rank`**: padded rows are always read.
* `grid = (SPLIT_K * cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N), NUM_SLICES, num_active_loras.item())`.
* Kernel `_lora_shrink_kernel` (L102):
  * program axis 2 indexes `lora_ids` (an early exit on -1);
  * rows come from `token_indices_sorted_by_lora_ids[lora_token_start_loc[i] : ...]`;
  * the work is done in `kernel_utils.py:237` `do_shrink_kernel`. The A tile address is
    `lora + d0*lora_index + n*d1 + k*d2`, i.e. **layout `[lora, rank, hidden]`**;
  * the scale is applied before the store. With `SPLIT_K == 1` it stores, otherwise it does
    `tl.atomic_add(..., sem="relaxed")` (L356). That is why the output is zeroed and kept in fp32.
* Batch-invariant mode (`VLLM_BATCH_INVARIANT`): fixed `_BI_SPLIT_K = 8` fp32 partials, then
  `_lora_shrink_reduce_kernel` (L27), with no atomics.

### `lora_expand`: `lora_expand_op.py:135` `_lora_expand` (kernel `_lora_expand_kernel` at L23)
```python
_lora_expand(
    inputs,                         # [num_slices, T, rank]  fp16/bf16/fp32 (fp32 -> CAST_TYPE)
    lora_b_weights: list[Tensor],   # one per slice: [num_loras, out_s, rank] or [num_loras, 1, out_s, rank]
    output_tensor,                  # [T, total_out], contiguous; slice s written at its column offset
    token_lora_mapping, token_indices_sorted_by_lora_ids, num_tokens_per_lora,
    lora_token_start_loc, lora_ids, no_lora_flag_cpu, num_active_loras,   # same meta_args()
    offset_start: int = 0,
    add_inputs: bool = False,
) -> None
```
* `K = B.shape[-1]` (rank, padded) and `MAX_N = max(out_s)`.
* `grid = (cdiv(M, BLOCK_M) * cdiv(MAX_N, BLOCK_N), NUM_SLICES, num_active_loras.item())`.
* `kernel_utils.py:111` `do_expand_kernel`:
  * there is no split-K ("K assumed small");
  * B tile at `lora + d0*lora_index + k*d2 + n*d1`, so **layout `[lora, out, rank]`**, with rank
    contiguous;
  * it loops over the full padded rank;
  * the accumulator is cast to the LoRA dtype, then with `ADD_INPUTS` it does load, add and store.
    This is not atomic, because each row is owned by exactly one CTA.
* When the slice widths differ (GQA qkv), `SAME_STRIDE=False` and each CTA reads its width from
  `hidden_sizes_tensor[slice_id]`.

### Pointer/stride packing (`lora/ops/triton_ops/utils.py`)
* `_get_lora_a_ptr` (L25) / `_get_lora_b_ptr` (L74):
  * accept 4-D `(lora_num, 1, size, rank)` (dim 1 is squeezed) or 3-D tensors, which must be
    contiguous;
  * with more than one slice they build a `uint64` pointer tensor;
  * B also builds `slice_start_loc`, starting at `offset_start` and adding up each slice's `out_s`.
    It also returns `hidden_sizes`, `same_stride` and `MAX_N`;
  * all A slices must share strides, or a `ValueError` is raised.
* **The results are memoized in `_LORA_A_PTR_DICT` / `_LORA_B_PTR_DICT`, keyed only by
  `tuple(t.data_ptr())`** ("collected during profile_run, then LUT"). See Section 7.

### Tuning configs ("Using default LoRA kernel configs")
* `load_lora_op_config` (L162) reads JSON only when `VLLM_TUNED_CONFIG_FOLDER` is set. The file name
  is `{gpu_name}_SHRINK.json` or `{gpu_name}_EXPAND_{TRUE|FALSE}.json`, where the suffix is
  `add_inputs` and spaces and dashes become `_`.
  * The JSON is nested as `[max_loras][num_slices][m][k][n]`, and the nearest key is used at each
    level.
* `get_lora_op_configs` (L195, `lru_cache`) logs `warning_once("Using default LoRA kernel configs")`
  at L276 when no file is loaded. That is the log line we saw, because the folder variable is unset.
* `max_loras` is passed as `lora_ids.size(0)`, which is `max_loras + 1`.

Defaults:

| op | block_m | block_n | block_k | split_k | warps | stages | other |
|---|---|---|---|---|---|---|---|
| shrink | 32 | 16 | 256 if M<128 else 32 | 64 if M<128 else 8 (1 if batch-invariant) | 4 | 2 | `group_size_m=8`, `num_ctas=1` |
| expand | 64 | 64 if num_slices>1 else 128 | 32 | – | 4 | 2 | `num_ctas=1` |

`supports_pdl` (L316): SM90+ and not `VLLM_LORA_DISABLE_PDL`. It is used only with
`VLLM_LORA_ENABLE_DUAL_STREAM`.

**Observation for the B3-vs-RF1 comparison.** Take decode (M < 128) on Qwen2.5-0.5B, with hidden
896 for q/gate/up and 4864 for the down projection input:
* shrink uses `BLOCK_K=256, SPLIT_K=64`, so `STEP_K = 16384 > K`;
* `mm_k` (`kernel_utils.py:9`) skips out-of-range K blocks, so for K=896 only 4 of the 64 split-K
  programs do work;
* all 64 programs still issue the `atomic_add` of their (zero) tile.

So B3's launch and atomic traffic is dominated by empty CTAs at small K. The ncu counts in P1.6
should show this.

---

## 5. LoRA layer classes and Qwen2/Llama wiring

### Base classes
* `lora/layers/base.py:17` `BaseLayerWithLoRA(nn.Module)`:
  * `__getattr__` forwards public names to `base_layer`;
  * interface: `slice_lora_a/b`, `create_lora_weights(max_loras, lora_config, model_config)`,
    `reset_lora(index)`, `set_lora(index, lora_a, lora_b)`, `set_mapping(punica_wrapper)` (L94,
    stores `self.punica_wrapper`) and classmethod `can_replace_layer`.
* `lora/layers/base_linear.py:70` `BaseLinearLayerWithLoRA`:
  * `create_lora_weights` (L104) allocates `n_slices` tuples:
    * `lora_a_stacked[s]`: zeros `[max_loras, 1, a_out, input_size]`, where `a_out = max_lora_rank`,
      or `max_lora_rank / tp` for fully-sharded column layers;
    * `lora_b_stacked[s]`: zeros `[max_loras, 1, b_out, max_lora_rank]`, where `b_out = output_size`,
      or `/ tp` for fully-sharded row layers;
    * `output_slices = (b_out,)`.
  * `set_lora` (L162) zeroes the slot, slices for TP, and copies into `[:r, :in]` / `[:out, :r]`.
  * `apply` (L198) goes to `_apply_sync` (L208), which runs `quant_method.apply(base_layer, x, bias)`
    and then `_apply_lora_to_output` (L219).
  * **`_apply_lora_to_output` is the single hook** for every non-sharded linear class:
    `punica_wrapper.add_lora_linear(output, x, lora_a_stacked, lora_b_stacked, 1.0, output_slices)`
    (L231).
  * The dual-stream variant `_apply_async_impl` (L244, custom op `lora_linear_async` at L33) calls the
    same `add_lora_linear` with `add_inputs=False` into a zeroed buffer.
* `lora/utils.py:79` `_all_lora_classes` is ordered; specific classes come first.
  `from_layer` (L107) picks the first class whose `can_replace_layer` returns True, instantiates it
  and calls `create_lora_weights`.

### How `add_lora_linear` runs (`punica_gpu.py:206`)
1. `r = lora_b_stacked[0].size(-1)`, which is the padded `max_lora_rank`.
2. `buffer = torch.empty((n_slices, T, r), float32)`, allocated per call. It is zeroed inside shrink.
3. `add_shrink(buffer, x, lora_a_stacked, scale)` (L91) runs `lora_shrink(x.view(-1, h), A, buffer,
   *token_mapping_meta.meta_args(T, specialize), scale)`.
4. `add_expand(y, buffer, lora_b_stacked, output_slices, add_inputs=True)` (L124) runs
   `lora_expand(buffer, B, y.view(-1, n), *meta_args, offset_start, add_inputs)`.

Other entry points: `add_lora_embedding` (L174, expand only), `add_lora_logits` (L271, uses
`prompt_mapping_meta`) and `apply_lora_full_linear` (L333).

### Slices and offsets for merged layers (`lora/layers/column_parallel_linear.py`)
* `ColumnParallelLinearWithLoRA` (L85) has one slice, with `output_size = output_size_per_partition`.
  * `slice_lora_b` (L106) takes the TP shard; for a merged-column base it splits the two halves
    separately.
  * `forward` (L131) all-gathers when `gather_output` is set.
  * `can_replace_layer` (L162) matches an exact `ColumnParallelLinear`, or a merged layer with one
    packed name.
* **`MergedColumnParallelLinearWithLoRA`** (L184) is gate_up.
  * `output_slices = tuple(output_sizes / tp)` and `n_slices = len(...)`.
  * `create_lora_weights` (L207) makes **one A and one B per slice**: A `[max_loras, 1, max_rank, in]`
    and B `[max_loras, 1, out_s, max_rank]`.
  * `set_lora` (L319) takes lists `[A_gate, A_up]`, `[B_gate, B_up]` (packed by
    `PackedLoRALayerWeights.pack`). It uses `expand_packed_lora` (L265) when the group count
    differs, and `slice_lora_b` (L252).
  * `can_replace_layer` (L364) requires exactly 2 packed names.
* `QKVParallelLinearWithLoRA` (L388) is a single LoRA over the fused qkv (`n_slices=1`); the q/k/v TP
  split happens in `slice_lora_b` (L415).
* **`MergedQKVParallelLinearWithLoRA`** (L453) is q/k/v as 3 slices with
  `output_slices = (q_shard, kv_shard, kv_shard)`. Because the widths differ, expand runs with
  `SAME_STRIDE=False`. `can_replace_layer` (L500) needs 3 packed names and is not fully sharded.
* Offsets: shrink writes `buffer[s]` using `A[s]`. Expand writes
  `y[:, slice_start_loc[s] : slice_start_loc[s] + out_s]`, where
  `slice_start_loc = offset_start + cumsum(out_s)` is computed in `_get_lora_b_ptr`. Slices are
  concatenated in `output_slices` order, which matches the base layer's packed output.
* Fully sharded (S-LoRA) variants:
  * `ColumnParallelLinearWithShardedLoRA` (L519), `MergedColumnParallelLinearWithShardedLoRA` (L559),
    `QKVParallelLinearWithShardedLoRA` (L601) and `MergedQKVParallelLinearWithShardedLoRA` (L636)
    all apply through `_mcp_apply` (L24): shrink, then `tensor_model_parallel_all_gather` of the
    buffer, then expand.
  * `MergedColumnParallelLinearVariableSliceWithLoRA` (L685) handles 3 or more unequal merged
    outputs.
  * **These call `add_shrink`/`add_expand` directly, not `add_lora_linear`.**

### Row-parallel and others
* `lora/layers/row_parallel_linear.py:22` `RowParallelLinearWithLoRA`:
  * `input_size = input_size_per_partition`, one slice;
  * `slice_lora_a` (L32) slices A's input columns by TP rank;
  * `forward` (L42) runs `apply` and then all-reduces;
  * `can_replace_layer` (L87) matches an exact `RowParallelLinear`.
* `RowParallelLinearWithShardedLoRA` (L102), `apply` at L118: shrink, all-reduce the buffer, then
  expand at `offset_start = tp_rank * shard`.
* `lora/layers/replicated_linear.py:16` `ReplicatedLinearWithLoRA`.
* Also present: `VocabParallelEmbeddingWithLoRA` (`vocab_parallel_embedding.py`),
  `LogitsProcessorWithLoRA` (`logits_processor.py`), `FusedMoEWithLoRA` / `FusedMoE3DWithLoRA`
  (`fused_moe.py`) and `ClassificationHeadWithLoRA`.

### What Qwen2 / Llama use (TP = 1, `fully_sharded_loras=False`)
`model_executor/models/qwen2.py:458` `Qwen2ForCausalLM(SupportsLoRA)` has `packed_modules_mapping`
at L462: `{"qkv_proj": ["q_proj","k_proj","v_proj"], "gate_up_proj": ["gate_proj","up_proj"]}`.

| module | base class (qwen2.py) | LoRA class | slices |
|---|---|---|---|
| `self_attn.qkv_proj` | `QKVParallelLinear`, `bias=True` (L157) | `MergedQKVParallelLinearWithLoRA` | 3 (q, k, v; unequal under GQA) |
| `self_attn.o_proj` | `RowParallelLinear` (L166) | `RowParallelLinearWithLoRA` | 1 |
| `mlp.gate_up_proj` | `MergedColumnParallelLinear` (L91) | `MergedColumnParallelLinearWithLoRA` | 2 (equal) |
| `mlp.down_proj` | `RowParallelLinear` (L98) | `RowParallelLinearWithLoRA` | 1 |

* Qwen2 defines no `embedding_modules` (the default `{}` comes from
  `model_executor/models/interfaces.py:681`), so `embed_tokens` and `lm_head` are never wrapped.
* Llama (`llama.py`) uses the same classes at L95/L103/L165/L175 and declares `embedding_modules`
  (L479), so it can also wrap the embedding and the logits processor.
* Our adapters target exactly `q, k, v, o, gate, up, down`, with `lora_alpha == r` and no rsLoRA.

---

## 6. What RankFuse changes

### Stage 1 (P4): one Matryoshka adapter, mixed ranks per token

The design goal: load one adapter at `r_max` into one slot, carry a rank per request, build a
token-to-rank array once per step, and replace the shrink+expand pair with the RankFuse op. A
`vllm.general_plugins` entry point is the least invasive way to install it:
* `plugins/__init__.py:77` `load_general_plugins` is called in `v1/engine/core.py:129` and in the
  worker (`v1/worker/worker_base.py:276`).
* That lets us monkeypatch the classes below from the `rankfuse` package without editing
  site-packages.

The flag (P4.5) is an env var read by the plugin, for example `RANKFUSE_VLLM=1`. If it becomes a
`LoRAConfig` field instead, add it to `compute_hash`.

| # | file | symbol | change |
|---|---|---|---|
| P4.1 | `lora/request.py` | `LoRARequest` | add trailing `matryoshka_rank: int \| None = None`; keep `__eq__`/`__hash__` on name |
| P4.1 | `v1/core/kv_cache_utils.py` | `_gen_lora_extra_hash_keys` (L578) | include the rank in the hash key |
| P4.1 | `entrypoints/openai/models/serving.py` | around L184 | optional: rank from the model name or an extra request field |
| P4.2 | `lora/worker_manager.py` | `LRUCacheWorkerLoRAManager._apply_adapters` / `add_adapter` | nothing, provided every rank uses the same `lora_int_id`; the set dedupes by name, so there is one load and one slot |
| P4.2 | `lora/peft_helper.py` / `lora/lora_weights.py` | `validate_legal`, `LoRALayerWeights.optimize` | run with `max_lora_rank = r_max` (64). If alpha is not equal to r, keep B unscaled and pass a per-rank scale table instead of the folded `alpha / r_max` |
| P4.3 | `v1/worker/gpu/lora_utils.py` | `LoraState.__init__/add_request/make_lora_inputs` | add `lora_ranks[max_num_reqs]` (default `r_max`, 0 for no LoRA); return the token-rank tuple via `repeat(num_scheduled_tokens)` |
| P4.3 | `v1/worker/gpu/model_runner.py` | around L1826–1834 | pass the extra tuple to `_set_active_loras` |
| P4.3 | `v1/worker/gpu_input_batch.py` | `InputBatch` (L256, L481, L545, L649, L779, `make_lora_inputs` L996) | the same for the V1 runner (only if we support `VLLM_USE_V2_MODEL_RUNNER=0`) |
| P4.3 | `v1/worker/lora_model_runner_mixin.py` | `_set_active_loras` (L58), `maybe_select_dummy_loras` (L143) | build the mapping with ranks; give dummy/capture runs a rank pattern (for example cyclic tiers) so graphs are captured with real rank tensors |
| P4.3 | `lora/layers/utils.py` | `LoRAMapping` (L34) | add `index_rank_mapping: tuple[int, ...] = ()` |
| P4.3 | `lora/punica_wrapper/punica_gpu.py` | `PunicaWrapperGPU.__init__/update_metadata` (or a `RankFusePunicaWrapper` installed through `platforms/cuda.py:575 get_punica_wrapper`) | static `_token_lora_ranks` int32 `[max_num_batched_tokens]`, `copy_`'d every step; if RF2 needs it, compute its bucket/sort order here once per step and not per layer |
| P4.4 | `lora/punica_wrapper/punica_gpu.py` | `add_lora_linear` (L206) | the single swap point for all four Qwen2 layer types: call `rankfuse_lora(x, A, B, token_ranks, out=y, add=True, slice_sizes=output_slices)` instead of shrink+expand |
| P4.4 | `lora/layers/base_linear.py` | `create_lora_weights` (L104) / `set_lora` (L162), and the merged overrides in `column_parallel_linear.py` (L207, L319) | match the RankFuse layout: vLLM's B is `[slot, 1, out, r]` (rank contiguous), while `rankfuse.interface.MatryoshkaLoRA.B` / `rankfuse_lora` take `B` as `[S, r_max, n]` (rank-major). Either store a transposed copy at `set_lora` time or make the kernel stride-aware. A (`[slot, 1, r, in]`) already matches `A_t [S, r_max, h_in]` after indexing slot 0 and squeezing dim 1 |
| P4.4 | `lora/layers/column_parallel_linear.py` `_mcp_apply` (L24), `row_parallel_linear.py` `RowParallelLinearWithShardedLoRA.apply` (L118) | only if `fully_sharded_loras` / TP > 1 is in scope; these bypass `add_lora_linear` | |
| P4.6 | – | – | the check: with every token at `r_max`, the outputs must equal stock vLLM on the same adapter |

Notes for Stage 1:
* With a single adapter, `max_loras = 1`. The slot is always index 0, so the RankFuse op can ignore
  `token_lora_mapping` except for -1 (base-only tokens). Base-only tokens can be given rank 0 in
  `_token_lora_ranks`, which unifies the two cases.
* The `y` that `add_lora_linear` receives is the base layer output, `[T, sum(output_slices)]`, laid
  out in slice order (q|k|v or gate|up). `slice_sizes = output_slices` maps directly to
  `rankfuse_lora`'s `slice_sizes`.
* The RankFuse op must be graph-safe (P3):
  * no `.item()` or host syncs inside it, unless it is wrapped as a custom op like vLLM's, which
    reads CPU tensors;
  * no per-call tensors whose addresses are baked into a graph (vLLM itself allocates the fp32
    buffer per call, which is fine under graph capture because it comes from the graph pool).

### Baseline B3 (P1.3): vLLM's ops called directly (`rankfuse/baselines/vllm_ops.py`)
No copying is needed. vLLM 0.32.0 is installed, so the plan's step 1 becomes
`from vllm.lora.ops.triton_ops import lora_shrink, lora_expand, LoRAKernelMeta`. This imports
`vllm.envs` and `vllm.platforms` but builds no engine.

Recipe (one layer, one slice), matching the docstring's plan:
1. Let the distinct ranks in the workload be `ranks = [r_1, ..., r_k]` and let `r_pad` be the vLLM
   `max_lora_rank`: the smallest allowed `MaxLoRARanks` value that is at least `r_max`. Use 64 for
   our adapters.
2. Build stacked buffers the way `create_lora_weights` and `set_lora` do:
   * `A_stack = zeros(k, r_pad, h_in)` with `A_stack[j, :r_j] = A_t[:r_j]`;
   * `B_stack = zeros(k, h_out, r_pad)` with `B_stack[j, :, :r_j] = B[:r_j].T`.

   Both must be contiguous. 3-D tensors are accepted, and so are 4-D ones with a singleton dim 1.
   Each prefix is stored again for every rank: that duplication is what Claim 1 measures.
3. `meta = LoRAKernelMeta.make(max_loras=k, max_num_tokens=T_max, device)` (no `captured_lora_counts`).
   Then `meta.prepare_tensors(token_adapter_idx)`, where `token_adapter_idx` is an int tensor on the
   device with value `j` = the index of the token's rank in `ranks`.
   * `prepare_tensors` syncs (`torch.all`, `torch.unique`), so run it outside the timed region and
     outside any CUDA graph, exactly as vLLM does once per step.
4. `buf = torch.empty(1, T, r_pad, dtype=float32)`. Then:
   * `lora_shrink(x, [A_stack], buf, *meta.meta_args(T, False), scale)`;
   * `lora_expand(buf, [B_stack], y, *meta.meta_args(T, False), offset_start=0, add_inputs=True)`.
5. For merged layers, pass per-slice lists: `[A_q, A_k, A_v]` / `[B_q, B_k, B_v]`, with
   `buf = (3, T, r_pad)`. `y` has width `sum(out_s)`.
6. Graphing B3 is valid: capture after `prepare_tensors`, as vLLM does. `meta_args(T, False)` passes
   the constant `default_num_active_loras_cpu = k + 1`, which makes grid dimension 3 equal `k + 1`.
   The extra program sees `lora_ids == -1` (filled by `_reset`) and exits early.
7. The bytes claim for the roofline (`rankfuse.roofline`, strategy `"separate_padded"`) holds:
   * shrink reads A rows `0..r_pad-1` (`N = r_pad`), so zero rows are included;
   * expand loops `K = r_pad` over B;
   * every distinct rank is a separate adapter slice, so a shared prefix is read once per rank
     present in the batch.

   Confirm it with ncu (P1.6).
8. Registration: `b3_vllm`, `requires_cuda=True`.

---

## 7. Pitfalls found while reading

* **The pointer cache is keyed by `data_ptr` only** (`ops/triton_ops/utils.py:21–22, L31, L83`).
  If B3 (or RankFuse tests) build several stacks whose first tensor starts at the same address but
  has a different shape or stride (for example views of one pool, or re-allocation after `del`), the
  op reuses stale strides or slice offsets without any error.
  * For a single slice, the cached entry keeps a reference to the tensor alive, so a plain
    reallocation cannot collide. Views can, and so can multi-slice lists whose first pointer matches.
  * Fix: keep one set of stacks per configuration alive for the whole run, or call
    `_LORA_A_PTR_DICT.clear()` / `_LORA_B_PTR_DICT.clear()` between configurations.
* **Scaling is folded into B** at load time (`LoRALayerWeights.optimize`, `lora_weights.py:37`), and
  layers pass `scale=1.0`. This is fine for our `alpha == r` adapters. A fixed-alpha Matryoshka
  adapter would need a per-rank scale that stock vLLM cannot express.
* **Prefix-cache aliasing:** `_gen_lora_extra_hash_keys` keys on `(lora_name, lora_path)`. Ranks
  that share a name share KV blocks unless the rank is added to the key.
* **Split-K atomics make shrink nondeterministic** (`kernel_utils.py:356`) unless
  `VLLM_BATCH_INVARIANT=1`. Exact-match tests against stock vLLM (P4.6) should either set that flag
  or compare at a tolerance on logits, with exact match only on greedy tokens.
* **Default configs are the untuned ones** (`get_lora_op_configs` L276). For a fair B3 comparison,
  report them as "vLLM default configs" or generate tuned JSONs into `VLLM_TUNED_CONFIG_FOLDER`.
* **The fp32 buffer of shape `(n_slices, T, max_lora_rank)` is allocated in every layer call**
  (`punica_gpu.py:251`). This is cheap under CUDA graphs but appears in eager-mode traces.
* `_mcp_apply` (`column_parallel_linear.py:72`), `RowParallelLinearWithShardedLoRA.apply`
  (`row_parallel_linear.py:152`) and `vocab_parallel_embedding.py:102` pass `add_input=True` (a
  misspelling). The keyword is silently swallowed by `**kwargs`, and `add_inputs` keeps its default
  of True, so the behavior is correct, but grepping for `add_inputs` will miss these call sites.
* `PunicaWrapperGPU` is marked `@final`. Python does not enforce this, but a RankFuse wrapper
  should subclass `PunicaWrapperBase` (or monkeypatch methods) to keep type checkers quiet.
