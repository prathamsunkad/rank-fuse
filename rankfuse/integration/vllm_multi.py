"""Multi-adapter RankFuse in vLLM 0.32 (RANKFUSE_VLLM=1 RANKFUSE_MULTI=1), optionally with prefix
paging (RANKFUSE_PAGING=1). Entry point: vllm_plugin.register() calls patch_multi().

Stage 1 (vllm_plugin.py) serves one Matryoshka adapter in one vLLM slot. This module keeps the
Stage 1 patches that carry a rank per request (MatryoshkaLoRARequest decoding, the rank in the
prefix-cache block hash, LoraState.lora_ranks, TokenRankMapping, the V1-runner refusal) and
replaces the ones that assumed one slot:

  wrapper   get_punica_wrapper -> MultiPunicaWrapper (multi_punica.py): per-step RF1-MA metadata
            (MultiMeta) built on the host from vLLM's token -> LoRA id mapping and the token ->
            rank mapping, copied into fixed buffers; one torch.ops.rankfuse.multi_lora_linear per
            dense LoRA layer (MULTI_REGISTRY['rf1_ma'] = rankfuse_multi.multi_lora, mode "group")
            instead of vLLM's shrink + expand.
  mapping   _set_active_loras builds a MultiLoRAMapping: LoRAMapping + per-token ranks + the
            step's padded token count (stashed by a wrapped GPUModelRunner.prepare_inputs) + the
            largest CUDA-graph size + the pager epoch. Dataclass equality covers all of them, so
            vLLM's "mapping unchanged, skip update_metadata" shortcut stays correct.
  weights   dense mode (RANKFUSE_PAGING=0): vLLM's own slots (max_loras of them, padded to
            max_lora_rank) plus rank-major copies [slots, S, r_max, *] refreshed in
            activate_adapter; PagedWeights.dense over the copies (identity page table).
            paged mode (RANKFUSE_PAGING=1): vLLM's slot buffers shrink to [1, 1, 16, *]
            placeholders (a proxy config while the layers are created), one PluginPageManager
            pool for the whole model is allocated right after the layers are wrapped, inside
            load_model, i.e. before vLLM measures free memory for the KV cache; every layer reads
            the pool through PluginPageManager.weights(layer key).
  adapters  paged mode: LRUCacheWorkerLoRAManager._apply_adapters loads adapters into vLLM's CPU
            cache (max_cpu_loras) but never into slots; LRUCacheLoRAModelManager.add_adapter
            packs pinned host pages and registers them; removal from the CPU cache unregisters
            (deactivate_adapter / remove_all_adapters). Warm-up / capture dummy adapters map to
            the pager's zero row.
  steps     paged mode, in _set_active_loras: begin_step(adapters) -> ensure({adapter: max rank})
            (copies on the pager's copy stream) -> set_adapter_mapping (metadata) ->
            wait_for_copies() (compute stream waits); GPUModelRunner.execute_model is wrapped to
            call end_step() once the forward is enqueued (before sample_tokens).
  scheduler paged mode: Scheduler._handle_blocked_waiting_request also applies a page budget:
            admit a request only if sum over adapters of the running set plus the request of
            ceil(max rank / 16) <= frames - RANKFUSE_PAGE_RESERVE. max_loras only bounds the
            number of distinct adapters in one step (metadata sizes, capture dummies).
            Cold requests (RANKFUSE_DEFER_COLD=1, default): a request whose adapter is registered
            on the host but not resident to its rank is deferred while other requests run, and
            its pages are prefetched after the current step (wait policy: it never runs at a
            lower rank); it is admitted anyway after RANKFUSE_DEFER_STEPS deferrals or when
            nothing else is running.
  compile   LoRAConfig.compute_hash adds "rankfuse-multi" + paging settings.

Limits: tensor parallel 1, no fully sharded LoRA, no dual-stream LoRA, model runner V2, every
LoRA module a dense linear layer of a supported type (paged mode requires it; dense mode lets
other modules use vLLM's kernels), max_lora_rank a multiple of 16.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os

import numpy as np


class _CfgProxy:
    """A LoRAConfig view with some fields overridden (used only while vLLM creates the layers)."""

    def __init__(self, cfg, **over):
        object.__setattr__(self, "_cfg", cfg)
        object.__setattr__(self, "_over", over)

    def __getattr__(self, name):
        over = object.__getattribute__(self, "_over")
        if name in over:
            return over[name]
        return getattr(object.__getattribute__(self, "_cfg"), name)


def defer_cold() -> bool:
    return os.environ.get("RANKFUSE_DEFER_COLD", "1") == "1"


def patch_multi() -> None:
    import torch
    from vllm.config.lora import LoRAConfig
    from vllm.logger import init_logger
    from vllm.lora import model_manager as mm
    from vllm.lora import worker_manager as wm
    from vllm.lora.layers import LoRAMapping, LoRAMappingType
    from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
    from vllm.platforms import current_platform
    from vllm.utils.gpu_sync_debug import gpu_sync_allowed
    from vllm.v1.core.sched import scheduler as sched_mod
    from vllm.v1.worker import gpu_input_batch, lora_model_runner_mixin
    from vllm.v1.worker.gpu import model_runner as v2_runner

    from rankfuse.integration import multi_ops, paging
    from rankfuse.integration import vllm_plugin as p1
    from rankfuse.integration.punica import FULL_RANK
    from rankfuse.multi import PagedWeights

    logger = init_logger("vllm.rankfuse_plugin")
    paged = paging.paging_enabled()

    # vLLM's own functions, before Stage 1 wraps them
    vllm_create = mm.LoRAModelManager._create_lora_modules
    vllm_activate = mm.LoRAModelManager.activate_adapter

    p1._patch_all()  # rank-carrying patches; the slot-specific ones are replaced below
    supported = p1._SUPPORTED if hasattr(p1, "_SUPPORTED") else None
    if supported is None:
        from vllm.lora.layers.column_parallel_linear import (
            ColumnParallelLinearWithLoRA,
            MergedColumnParallelLinearWithLoRA,
            MergedQKVParallelLinearWithLoRA,
            QKVParallelLinearWithLoRA,
        )
        from vllm.lora.layers.replicated_linear import ReplicatedLinearWithLoRA
        from vllm.lora.layers.row_parallel_linear import RowParallelLinearWithLoRA

        supported = (
            ColumnParallelLinearWithLoRA, MergedColumnParallelLinearWithLoRA, QKVParallelLinearWithLoRA,
            MergedQKVParallelLinearWithLoRA, RowParallelLinearWithLoRA, ReplicatedLinearWithLoRA,
        )

    # ---- compile cache -----------------------------------------------------------------------
    stage1_hash = LoRAConfig.compute_hash
    tag = f"rankfuse-multi|paging={int(paged)}|kernel={multi_ops.KERNEL}|{p1._source_hash()}"

    def compute_hash(self) -> str:
        # The pool and page table are inputs of the compiled graph with fixed shapes: frames (from
        # RANKFUSE_PAGE_FRAMES / RANKFUSE_PAGE_POOL_GB and the model) and rows (max_cpu_loras + 1).
        # A cached graph from an engine with another pool size fails its shape guards.
        pool = (f"|frames={os.environ.get('RANKFUSE_PAGE_FRAMES', '')}|gb={os.environ.get('RANKFUSE_PAGE_POOL_GB', '')}"
                f"|cpu_loras={self.max_cpu_loras}|loras={self.max_loras}" if paged else "")
        return hashlib.sha256(f"{stage1_hash(self)}|{tag}{pool}".encode()).hexdigest()

    LoRAConfig.compute_hash = compute_hash

    # ---- wrapper -----------------------------------------------------------------------------
    type(current_platform).get_punica_wrapper = classmethod(
        lambda cls: "rankfuse.integration.multi_punica.MultiPunicaWrapper"
    )

    # ---- V1 runner: unsupported ----------------------------------------------------------------
    def v1_make_lora_inputs(self, *args, **kwargs):
        raise NotImplementedError("RankFuse multi-adapter mode needs model runner V2")

    gpu_input_batch.InputBatch.make_lora_inputs = v1_make_lora_inputs

    # ---- mapping -----------------------------------------------------------------------------
    @dataclasses.dataclass
    class MultiLoRAMapping(LoRAMapping):
        index_rank_mapping: tuple[int, ...] = ()
        num_tokens_padded: int = 0
        graph_max_tokens: int = 0
        epoch: int = 0

        def __post_init__(self):
            super().__post_init__()
            self.index_rank_mapping = tuple(self.index_rank_mapping)

    globals()["MultiLoRAMapping"] = MultiLoRAMapping

    orig_prepare_inputs = v2_runner.GPUModelRunner.prepare_inputs

    def prepare_inputs(self, *args, **kwargs):
        ib = orig_prepare_inputs(self, *args, **kwargs)
        self._rf_num_tokens_padded = int(ib.num_tokens_after_padding)
        return ib

    v2_runner.GPUModelRunner.prepare_inputs = prepare_inputs

    def graph_max_tokens(runner) -> int:
        if getattr(runner, "cudagraph_manager", None) is None:
            return 0
        sizes = getattr(runner.compilation_config, "cudagraph_capture_sizes", None) or []
        return max(sizes) if sizes else 0

    def _set_active_loras(self, prompt_lora_mapping, token_lora_mapping, lora_requests,
                          mapping_type=LoRAMappingType.LANGUAGE):
        self._ensure_lora_enabled()
        n = len(token_lora_mapping)
        ranks = getattr(token_lora_mapping, "ranks", None)
        if ranks is None:
            # Warm-up / capture (maybe_select_dummy_loras): cycle the rank tiers over adapter tokens.
            r_max = self.lora_config.max_lora_rank
            tiers = [t for t in p1.RANK_TIERS if t <= r_max]
            ranks = tuple(tiers[i % len(tiers)] if lid > 0 else 0 for i, lid in enumerate(token_lora_mapping))
        T_pad = self.__dict__.pop("_rf_num_tokens_padded", None) or n
        T_pad = max(int(T_pad), n)
        worker = self.lora_manager
        am = worker._adapter_manager
        pager = getattr(am, "_rf_pager", None)
        kw = dict(is_prefill=True, type=mapping_type, index_rank_mapping=ranks, num_tokens_padded=T_pad,
                  graph_max_tokens=graph_max_tokens(self))
        if pager is None:
            mapping = MultiLoRAMapping(tuple(token_lora_mapping), prompt_lora_mapping, **kw)
            worker.set_active_adapters(lora_requests, mapping)
            return
        worker._apply_adapters(lora_requests)  # host registration only (patched below)
        ids = np.asarray(tuple(token_lora_mapping), dtype=np.int64)
        need = {}
        for a, r in paging.max_rank_per_adapter(ids, np.asarray(ranks, dtype=np.int64)).items():
            if a in am._rf_dummy or not pager.is_registered(a):
                continue
            need[a] = min(r, pager.adapter_rank(a))
        pager.begin_step(need)
        pager.ensure(need)
        am._rf_stats["steps"] += 1
        mapping = MultiLoRAMapping(tuple(token_lora_mapping), prompt_lora_mapping, epoch=pager.epoch, **kw)
        am.set_adapter_mapping(mapping)
        pager.wait_for_copies()
        _serve_prefetch(am, pager)

    lora_model_runner_mixin.LoRAModelRunnerMixin._set_active_loras = _set_active_loras

    orig_execute = v2_runner.GPUModelRunner.execute_model

    def execute_model(self, *args, **kwargs):
        out = orig_execute(self, *args, **kwargs)
        am = getattr(getattr(self, "lora_manager", None), "_adapter_manager", None)
        pager = getattr(am, "_rf_pager", None)
        if pager is not None:
            pager.end_step()
        return out

    v2_runner.GPUModelRunner.execute_model = execute_model

    # ---- layers, weights and the pool ----------------------------------------------------------
    def _create_lora_modules(self):
        cfg = self.lora_config
        if cfg.fully_sharded_loras:
            raise NotImplementedError("RankFuse does not support fully_sharded_loras")
        if cfg.max_lora_rank % 16:
            raise NotImplementedError(f"RankFuse multi-adapter mode needs max_lora_rank % 16 == 0, got {cfg.max_lora_rank}")
        if paged:
            # vLLM's per-layer slot buffers become [1, 1, 16, *] placeholders: the pool holds the weights.
            self.lora_config = _CfgProxy(cfg, max_loras=1, max_lora_rank=16)
            try:
                vllm_create(self)
            finally:
                self.lora_config = cfg
        else:
            vllm_create(self)
        mods = []
        all_rf = True
        for name, module in self.modules.items():
            if type(module) not in supported or getattr(module, "tp_size", 1) != 1:
                all_rf = False
                if paged:
                    raise NotImplementedError(f"RankFuse paging: {name} ({type(module).__name__}) is not supported")
                logger.warning("RankFuse: %s (%s) keeps vLLM's LoRA kernels", name, type(module).__name__)
                continue
            assert len(module.output_slices) == len(module.lora_a_stacked)
            mods.append((name, module))
        wrappers = [w for w in self.punica_wrapper_mapping.values() if hasattr(w, "configure")]
        if len(wrappers) != 1:
            raise NotImplementedError("RankFuse multi-adapter mode expects one language-model Punica wrapper")
        wrapper = wrappers[0]
        wrapper.rankfuse_only = all_rf
        r_max = cfg.max_lora_rank
        multi_ops.CONTEXT.update(manager=self, wrapper=wrapper)
        self._rf_stats = {"steps": 0, "registered": 0, "unregistered": 0}
        self._rf_dummy = set()
        if paged:
            dtype = cfg.lora_dtype if isinstance(cfg.lora_dtype, torch.dtype) else mods[0][1].lora_a_stacked[0].dtype
            dev = mods[0][1].lora_a_stacked[0].device
            blocks, keys = [], {}
            for i, (name, m) in enumerate(mods):
                key = (i, name)
                blocks.append((key, m.lora_a_stacked[0].shape[-1], tuple(m.output_slices)))
                keys[name] = key
            layout = paging.ModuleLayout(blocks)
            frames = paging.pool_frames(layout.page_bytes(dtype))
            pager = paging.PluginPageManager(layout, frames, num_adapter_slots=cfg.max_cpu_loras, r_max=r_max,
                                             dtype=dtype, device=dev)
            self._rf_pager, self._rf_keys = pager, keys
            paging.set_current_pager(pager)
            scale = wrapper.scale_table(pager.num_rows)
            for name, m in mods:
                W = pager.weights(*keys[name], scale_table=scale)
                m._rankfuse_wid = multi_ops.register_weights(W, name)
                m._rankfuse_tensors = (W.A_pool, W.B_pool, W.page_table)
            dummy_row = pager.dummy_row

            def resolver(ids, lora_index_to_id, am=self, pager=pager):
                u, inv = np.unique(ids, return_inverse=True)
                row_u = np.full(u.shape[0], -1, dtype=np.int64)
                cap_u = np.zeros(u.shape[0], dtype=np.int64)
                for j, a in enumerate(u.tolist()):
                    if a <= 0:
                        continue
                    if a in am._rf_dummy:
                        row_u[j], cap_u[j] = dummy_row, r_max
                    elif pager.is_registered(a):
                        row_u[j], cap_u[j] = pager.slot(a), pager.adapter_rank(a)
                return row_u[inv], cap_u[inv]

            wrapper.configure(pager.num_rows, resolver, paged=True)
            logger.info(
                "RankFuse paging: pool of %d frames + 1 zero frame, %.1f MB per page (%.2f GB), %d adapter rows, "
                "r_max %d, %d LoRA modules", frames, layout.page_bytes(dtype) / 1e6,
                (frames + 1) * layout.page_bytes(dtype) / 1e9, cfg.max_cpu_loras, r_max, len(mods))
        else:
            slots = self.lora_slots
            self._rf_slot_rank = [0] * slots
            for name, m in mods:
                a0, b0 = m.lora_a_stacked[0], m.lora_b_stacked[0]
                S = len(m.lora_a_stacked)
                _, _, rr, h_in = a0.shape
                n_max = max(m.output_slices)
                A = torch.zeros(slots, S, rr, h_in, dtype=a0.dtype, device=a0.device)
                B = torch.zeros(slots, S, rr, n_max, dtype=b0.dtype, device=b0.device)
                m._rankfuse_pack = (A, B)
                scale = wrapper.scale_table(slots)
                W = PagedWeights.dense(A, B, slice_sizes=tuple(m.output_slices), scale_per_rank=scale)
                m._rankfuse_wid = multi_ops.register_weights(W, name)
                m._rankfuse_tensors = (W.A_pool, W.B_pool, W.page_table)

            def resolver(ids, lora_index_to_id, am=self):
                idx = {lid: i for i, lid in enumerate(lora_index_to_id) if lid is not None}
                u, inv = np.unique(ids, return_inverse=True)
                row_u = np.full(u.shape[0], -1, dtype=np.int64)
                cap_u = np.zeros(u.shape[0], dtype=np.int64)
                for j, a in enumerate(u.tolist()):
                    i = idx.get(a) if a > 0 else None
                    if i is not None:
                        row_u[j], cap_u[j] = i, am._rf_slot_rank[i] or r_max
                return row_u[inv], cap_u[inv]

            wrapper.configure(slots, resolver, paged=False)
            logger.info("RankFuse multi-adapter (dense slots): %d slots x r_max %d on %d LoRA modules%s", slots,
                        r_max, len(mods), "" if all_rf else " (others use vLLM's kernels)")

    mm.LoRAModelManager._create_lora_modules = _create_lora_modules

    def activate_adapter(self, lora_id: int) -> bool:
        if getattr(self, "_rf_pager", None) is not None:
            return False  # paged adapters never take a vLLM slot
        result = vllm_activate(self, lora_id)
        if result:
            index = self.lora_index_to_id.index(lora_id)
            lora = self._registered_adapters[lora_id]
            self._rf_slot_rank[index] = paging.lora_model_rank(lora, self.modules)
            for module in self.modules.values():
                pack = getattr(module, "_rankfuse_pack", None)
                if pack is not None:
                    A, B = pack
                    for s, (a, b) in enumerate(zip(module.lora_a_stacked, module.lora_b_stacked)):
                        A[index, s].copy_(a[index, 0])
                        B[index, s, :, : b.shape[2]].copy_(b[index, 0].T)
        return result

    mm.LoRAModelManager.activate_adapter = activate_adapter

    # ---- paged adapters: host registration, never slots --------------------------------------
    orig_lru_add = mm.LRUCacheLoRAModelManager.add_adapter

    def lru_add_adapter(self, lora) -> bool:
        added = orig_lru_add(self, lora)
        pager = getattr(self, "_rf_pager", None)
        if pager is not None and lora.id not in self._rf_dummy and not pager.is_registered(lora.id):
            host, rank = paging.pack_lora_model(pager.layout, self._rf_keys, lora, self._get_lora_layer_weights,
                                                dtype=pager.pool.pages.dtype)
            if rank > pager.r_max:
                raise ValueError(f"adapter {lora.id} has rank {rank} > max_lora_rank {pager.r_max}")
            pager.register(lora.id, host, rank)
            self._rf_stats["registered"] += 1
            lora.loras = {}  # the host pages are the CPU copy now; vLLM never reads these weights again
        return added

    mm.LRUCacheLoRAModelManager.add_adapter = lru_add_adapter

    orig_deactivate = mm.LoRAModelManager.deactivate_adapter

    def deactivate_adapter(self, adapter_id: int) -> bool:
        pager = getattr(self, "_rf_pager", None)
        if pager is not None:
            self._rf_dummy.discard(adapter_id)
            if pager.unregister(adapter_id):
                self._rf_stats["unregistered"] += 1
        return orig_deactivate(self, adapter_id)

    mm.LoRAModelManager.deactivate_adapter = deactivate_adapter

    orig_remove_all = mm.LoRAModelManager.remove_all_adapters

    def remove_all_adapters(self):
        pager = getattr(self, "_rf_pager", None)
        if pager is not None:
            for a in pager.registered():
                pager.unregister(a)
            self._rf_dummy.clear()
        return orig_remove_all(self)

    mm.LoRAModelManager.remove_all_adapters = remove_all_adapters

    orig_add_dummy = wm.WorkerLoRAManager.add_dummy_lora

    def add_dummy_lora(self, lora_request, rank: int) -> bool:
        am = self._adapter_manager
        if getattr(am, "_rf_pager", None) is not None:
            am._rf_dummy.add(lora_request.lora_int_id)
        return orig_add_dummy(self, lora_request, rank)

    wm.WorkerLoRAManager.add_dummy_lora = add_dummy_lora

    orig_apply = wm.LRUCacheWorkerLoRAManager._apply_adapters

    def _apply_adapters(self, lora_requests) -> None:
        am = self._adapter_manager
        if getattr(am, "_rf_pager", None) is None:
            return orig_apply(self, lora_requests)
        loras = {r.lora_int_id: r for r in lora_requests if r}
        if len(loras) > am.capacity:
            raise RuntimeError(f"{len(loras)} adapters in one step exceed max_cpu_loras ({am.capacity})")
        present = self.list_adapters()
        new = []
        for lid, r in loras.items():  # touch first, so loading new ones never evicts this step's
            if lid in am._rf_dummy:
                continue
            if lid in present and not r.load_inplace:
                am._registered_adapters.touch(lid)
            else:
                new.append(r)
        for r in new:
            with gpu_sync_allowed():
                lora = self._load_adapter(r)
                am.remove_adapter(lora.id)
                if len(am) + 1 > am.capacity:
                    am.remove_oldest_adapter()
                am.add_adapter(lora)

    wm.LRUCacheWorkerLoRAManager._apply_adapters = _apply_adapters

    # ---- the kernel swap -------------------------------------------------------------------------
    stage1_apply = BaseLinearLayerWithLoRA._apply_lora_to_output

    def _apply_lora_to_output(self, x, output):
        wid = getattr(self, "_rankfuse_wid", None)
        if wid is None:
            return stage1_apply(self, x, output)
        original_shape = output.shape if output.ndim == 3 else None
        if x.ndim == 3 and output.ndim == 3:
            output = output.flatten(0, 1)
            x = x.flatten(0, 1)
        self.punica_wrapper.add_lora_linear(output, x, self.lora_a_stacked, self.lora_b_stacked, 1.0,
                                            self.output_slices, rankfuse_wid=wid,
                                            rankfuse_tensors=self._rankfuse_tensors)
        if original_shape is not None:
            output = output.reshape(original_shape)
        return output

    BaseLinearLayerWithLoRA._apply_lora_to_output = _apply_lora_to_output

    # ---- scheduler: page budget, deferred cold requests --------------------------------------
    if paged:
        orig_blocked = sched_mod.Scheduler._handle_blocked_waiting_request
        defer_steps = int(os.environ.get("RANKFUSE_DEFER_STEPS", "8"))

        def _handle_blocked_waiting_request(self, request) -> bool:
            ok = orig_blocked(self, request)
            lr = getattr(request, "lora_request", None)
            if not ok or lr is None or lr.lora_int_id <= 0:
                return ok
            pager = paging.current_pager()
            frames = pager.usable_frames if pager is not None else (
                int(os.environ["RANKFUSE_PAGE_FRAMES"]) if os.environ.get("RANKFUSE_PAGE_FRAMES") else None)
            if frames is None:
                return ok
            r_max = self.lora_config.max_lora_rank
            need = paging.step_pages(self.running, r_max, FULL_RANK)
            a, p = paging.request_pages(lr, r_max, FULL_RANK)
            if p > frames - paging.reserve_frames():
                raise ValueError(f"request {request.request_id} needs {p} pages; the pool has {frames}")
            need[a] = max(need.get(a, 0), p)
            st = _sched_stats(self)
            if sum(need.values()) > frames - paging.reserve_frames():
                st["budget_skips"] += 1
                return False
            if defer_cold() and pager is not None and self.running and pager.is_registered(a):
                cold = pager.resident_pages(a) < p
                n_def = getattr(request, "_rf_deferred", 0)
                if cold and n_def < defer_steps:
                    request._rf_deferred = n_def + 1
                    _PREFETCH[a] = max(_PREFETCH.get(a, 0), p * 16)
                    st["deferred"] += 1
                    return False
            return True

        sched_mod.Scheduler._handle_blocked_waiting_request = _handle_blocked_waiting_request

    logger.info("RankFuse multi-adapter plugin active (RANKFUSE_MULTI=1, paging=%s, kernel=%s)", paged,
                multi_ops.KERNEL)


# Prefetch hints from the scheduler (same process as the worker with UniProcExecutor); served by
# the worker after the step's metadata is set: free frames and idle adapters' pages only.
_PREFETCH: dict[int, int] = {}  # adapter -> rank, deduplicated per step
SCHED_STATS: dict = {}


def _sched_stats(sched) -> dict:
    if not SCHED_STATS:
        SCHED_STATS.update(budget_skips=0, deferred=0, prefetched=0)
    return SCHED_STATS


def _serve_prefetch(am, pager) -> None:
    hints = list(_PREFETCH.items())
    _PREFETCH.clear()
    for a, r in hints:
        if pager.is_registered(a):
            pager.prefetch(a, min(r, pager.adapter_rank(a)))
            SCHED_STATS["prefetched"] = SCHED_STATS.get("prefetched", 0) + 1
