"""RankFuse in vLLM 0.32: a `vllm.general_plugins` entry point, active only with RANKFUSE_VLLM=1.

Stage 1 (docs/VLLM_LORA_NOTES.md, section 6): one Matryoshka adapter loaded once at r_max into
one GPU slot, a rank carried by each request, a token-to-rank buffer built once per step, and
RankFuse's fused kernel in place of vLLM's shrink + expand pair in every dense LoRA layer.

vLLM calls `load_general_plugins()` in every process (API front end, engine core, workers)
before it builds the model, so `register()` runs everywhere and must be idempotent. Without
RANKFUSE_VLLM=1 it returns before importing anything from vLLM: the engine is stock.

What `register()` patches (all by monkeypatching; site-packages is not edited):

  compile    LoRAConfig.compute_hash (part of vLLM's torch.compile / AOT cache key) also hashes
  cache      "rankfuse" and the plugin + kernel sources, so cached graphs never cross between
             stock and RankFuse engines (found the hard way: a RankFuse engine silently reloaded
             a stock engine's compiled graph).
  request    MatryoshkaLoRARequest (request.py) is LoRARequest plus a trailing
             matryoshka_rank. EngineCoreRequest's lora_request annotation is pointed at it so the
             engine core decodes the rank instead of dropping the extra array element.
  prefix     kv_cache_utils._gen_lora_extra_hash_keys: the rank joins (name, path) in the block
  cache      hash, so two ranks of one adapter never share KV blocks.
  per-step   gpu.lora_utils.LoraState (model runner V2, the default): a lora_ranks array next to
  ranks      lora_ids; make_lora_inputs returns the token-id tuple as a TokenRankMapping that
             also carries the token-rank tuple (same `repeat` as the ids).
             LoRAModelRunnerMixin._set_active_loras builds a RankedLoRAMapping (LoRAMapping +
             index_rank_mapping). Dataclass equality includes the ranks, so the manager's
             "mapping unchanged, skip update" check stays correct. Mappings built without ranks
             (warm-up and the CUDA-graph capture hook, maybe_select_dummy_loras) get a cyclic
             pattern of rank tiers for adapter tokens.
             gpu_input_batch.InputBatch.make_lora_inputs (V1 runner): refuses Matryoshka ranks.
  wrapper    <CUDA platform class>.get_punica_wrapper -> RankFusePunicaWrapper (punica.py): a static
             int32 token-rank buffer refilled in update_metadata (also during capture, which
             goes through the same _set_active_loras -> update_metadata path) and the step's
             r_limit in a CPU tensor, computed on the host from the request ranks: no GPU sync.
  weights    LoRAModelManager._create_lora_modules: every dense LoRA linear layer of a supported
             type gets rank-major copies [slots, n_slices, r_max, h_in] / [slots, n_slices,
             r_max, n_max] (vLLM stores B as [slot, 1, out, r]). activate_adapter (the one place
             set_lora/reset_lora run) refreshes the copies of the slot it filled.
  kernel     BaseLinearLayerWithLoRA._apply_lora_to_output passes those copies to
             add_lora_linear, which runs torch.ops.rankfuse.lora_linear (ops.py): one fused
             launch pair for q|k|v, o, gate|up and down, adding into the base output.

Limits (checked at engine start): max_loras = 1, tensor parallel size 1, no fully sharded
LoRA, no VLLM_LORA_ENABLE_DUAL_STREAM, model runner V2 for Matryoshka ranks.
"""

import hashlib
import os
import threading
from pathlib import Path

_LOCK = threading.Lock()
_DONE = False
RANK_TIERS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def enabled() -> bool:
    return os.environ.get("RANKFUSE_VLLM", "0") == "1"


def _source_hash() -> str:
    """Hash of the RankFuse code that ends up inside vLLM's compiled graph."""
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "integration").glob("*.py")) + [root / "kernels" / "rankfuse_fused.py"]
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def register() -> None:
    """Entry point (pyproject: [project.entry-points."vllm.general_plugins"] rankfuse)."""
    global _DONE
    if not enabled():
        return
    with _LOCK:
        if _DONE:
            return
        _patch_all()
        _DONE = True


class TokenRankMapping(tuple):
    """The per-token adapter-id tuple of make_lora_inputs, carrying the per-token rank tuple.

    The model runner unpacks make_lora_inputs' result straight into _set_active_loras, so the
    ranks ride along on the id tuple instead of changing the call's arity.
    """

    ranks: tuple[int, ...]

    def __new__(cls, ids, ranks):
        obj = super().__new__(cls, ids)
        obj.ranks = ranks
        return obj


def _patch_all() -> None:
    import dataclasses

    import numpy as np
    import torch
    from vllm import envs
    from vllm.logger import init_logger
    from vllm.lora import model_manager as mm
    from vllm.lora.layers import LoRAMapping, LoRAMappingType
    from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
    from vllm.lora.layers.column_parallel_linear import (
        ColumnParallelLinearWithLoRA,
        MergedColumnParallelLinearWithLoRA,
        MergedQKVParallelLinearWithLoRA,
        QKVParallelLinearWithLoRA,
    )
    from vllm.lora.layers.replicated_linear import ReplicatedLinearWithLoRA
    from vllm.lora.layers.row_parallel_linear import RowParallelLinearWithLoRA
    from vllm.platforms import current_platform
    from vllm.v1 import engine as v1_engine
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.worker import gpu_input_batch, lora_model_runner_mixin
    from vllm.v1.worker.gpu import lora_utils

    from rankfuse.integration.punica import FULL_RANK
    from rankfuse.integration.request import MatryoshkaLoRARequest

    logger = init_logger("vllm.rankfuse_plugin")  # under "vllm" so vLLM's log handler prints it
    if envs.VLLM_LORA_ENABLE_DUAL_STREAM:
        raise RuntimeError("RANKFUSE_VLLM=1 does not support VLLM_LORA_ENABLE_DUAL_STREAM")

    # ---- compile cache: RankFuse graphs never mix with stock ones --------------------------
    # vLLM stores torch.compile / AOT artifacts under a hash of the config; it cannot see these
    # monkeypatches, so without this a RankFuse engine reloads a graph compiled by a stock
    # engine (which calls vLLM's own LoRA kernels) and vice versa. LoRAConfig.compute_hash is
    # part of that hash; add the plugin and the source of the code it traces into.
    from vllm.config.lora import LoRAConfig

    orig_lora_hash = LoRAConfig.compute_hash
    code = _source_hash()

    def compute_hash(self) -> str:
        return hashlib.sha256(f"{orig_lora_hash(self)}|rankfuse-stage1|{code}".encode()).hexdigest()

    LoRAConfig.compute_hash = compute_hash

    # ---- request: decode the trailing rank in the engine core ------------------------------
    ann = v1_engine.EngineCoreRequest.__annotations__
    ann["lora_request"] = MatryoshkaLoRARequest | None

    # ---- prefix cache: the rank is part of the block hash ----------------------------------
    orig_hash_keys = kv_cache_utils._gen_lora_extra_hash_keys

    def _gen_lora_extra_hash_keys(request):
        keys = orig_hash_keys(request)
        rank = getattr(request.lora_request, "matryoshka_rank", None) if keys else None
        if rank is not None:
            keys = [keys[0] + ("matryoshka_rank", rank)] + keys[1:]
        return keys

    kv_cache_utils._gen_lora_extra_hash_keys = _gen_lora_extra_hash_keys

    # ---- per-request ranks (model runner V2) -----------------------------------------------
    LoraState = lora_utils.LoraState
    orig_state_init = LoraState.__init__
    orig_add_request = LoraState.add_request

    def state_init(self, max_num_reqs: int):
        orig_state_init(self, max_num_reqs)
        self.lora_ranks = np.zeros(max_num_reqs, dtype=np.int32)

    def add_request(self, req_id, req_index, lora_request):
        orig_add_request(self, req_id, req_index, lora_request)
        if lora_request is None:
            self.lora_ranks[req_index] = 0
        else:
            rank = getattr(lora_request, "matryoshka_rank", None)
            self.lora_ranks[req_index] = FULL_RANK if rank is None else rank

    def make_lora_inputs(self, req_ids, idx_mapping, num_scheduled_tokens):
        lora_ids = self.lora_ids[idx_mapping]
        ranks = self.lora_ranks[idx_mapping]
        prompt_lora_mapping = tuple(lora_ids)
        token_lora_mapping = TokenRankMapping(
            lora_ids.repeat(num_scheduled_tokens).tolist(), tuple(ranks.repeat(num_scheduled_tokens).tolist())
        )
        return prompt_lora_mapping, token_lora_mapping, self.get_activate_loras(req_ids)

    LoraState.__init__ = state_init
    LoraState.add_request = add_request
    LoraState.make_lora_inputs = make_lora_inputs

    # ---- V1 runner: not supported for Matryoshka ranks -------------------------------------
    orig_v1_inputs = gpu_input_batch.InputBatch.make_lora_inputs

    def v1_make_lora_inputs(self, *args, **kwargs):
        prompt, token, requests = orig_v1_inputs(self, *args, **kwargs)
        if any(getattr(r, "matryoshka_rank", None) is not None for r in requests):
            raise NotImplementedError("RankFuse Matryoshka ranks need model runner V2 (VLLM_USE_V2_MODEL_RUNNER)")
        return prompt, token, requests

    gpu_input_batch.InputBatch.make_lora_inputs = v1_make_lora_inputs

    # ---- LoRAMapping with ranks ------------------------------------------------------------
    @dataclasses.dataclass
    class RankedLoRAMapping(LoRAMapping):
        index_rank_mapping: tuple[int, ...] = ()

        def __post_init__(self):
            super().__post_init__()
            self.index_rank_mapping = tuple(self.index_rank_mapping)

    globals()["RankedLoRAMapping"] = RankedLoRAMapping

    def _set_active_loras(self, prompt_lora_mapping, token_lora_mapping, lora_requests,
                          mapping_type=LoRAMappingType.LANGUAGE):
        self._ensure_lora_enabled()
        ranks = getattr(token_lora_mapping, "ranks", None)
        if ranks is None:
            # Warm-up / CUDA-graph capture (maybe_select_dummy_loras): adapter tokens cycle
            # through the rank tiers so the capture runs with a real mixed-rank buffer.
            r_max = self.lora_config.max_lora_rank
            tiers = [t for t in RANK_TIERS if t <= r_max]
            ranks = tuple(tiers[i % len(tiers)] if lid > 0 else 0 for i, lid in enumerate(token_lora_mapping))
        mapping = RankedLoRAMapping(
            tuple(token_lora_mapping), prompt_lora_mapping, is_prefill=True, type=mapping_type,
            index_rank_mapping=ranks,
        )
        self.lora_manager.set_active_adapters(lora_requests, mapping)

    lora_model_runner_mixin.LoRAModelRunnerMixin._set_active_loras = _set_active_loras

    # ---- Punica wrapper --------------------------------------------------------------------
    if not current_platform.is_cuda():
        raise RuntimeError("RANKFUSE_VLLM=1 needs a CUDA platform")
    type(current_platform).get_punica_wrapper = classmethod(
        lambda cls: "rankfuse.integration.punica.RankFusePunicaWrapper"
    )

    # ---- RankFuse weight copies, kept in sync with vLLM's slot buffers ---------------------
    supported = (
        ColumnParallelLinearWithLoRA, MergedColumnParallelLinearWithLoRA, QKVParallelLinearWithLoRA,
        MergedQKVParallelLinearWithLoRA, RowParallelLinearWithLoRA, ReplicatedLinearWithLoRA,
    )
    orig_create = mm.LoRAModelManager._create_lora_modules

    def _create_lora_modules(self):
        orig_create(self)
        cfg = self.lora_config
        if self.lora_slots != 1:
            raise NotImplementedError(f"RankFuse Stage 1 serves one adapter slot (max_loras=1), got {self.lora_slots}")
        if cfg.fully_sharded_loras:
            raise NotImplementedError("RankFuse does not support fully_sharded_loras")
        all_rankfuse = True
        for name, module in self.modules.items():
            if type(module) not in supported or getattr(module, "tp_size", 1) != 1:
                all_rankfuse = False
                logger.warning("RankFuse: %s (%s) keeps vLLM's LoRA kernels", name, type(module).__name__)
                continue
            a0, b0 = module.lora_a_stacked[0], module.lora_b_stacked[0]
            n_slices = len(module.lora_a_stacked)
            assert len(module.output_slices) == n_slices
            assert all(b.shape[2] == w for b, w in zip(module.lora_b_stacked, module.output_slices))
            slots, _, r_max, h_in = a0.shape
            n_max = max(module.output_slices)
            A = torch.zeros(slots, n_slices, r_max, h_in, dtype=a0.dtype, device=a0.device)
            B = torch.zeros(slots, n_slices, r_max, n_max, dtype=b0.dtype, device=b0.device)
            module._rankfuse_pack = (A, B)
            module._rankfuse_weights = (A[0], B[0])  # Stage 1: slot 0
        for wrapper in self.punica_wrapper_mapping.values():
            if hasattr(wrapper, "rankfuse_only"):
                wrapper.rankfuse_only = all_rankfuse
        logger.info("RankFuse: fused Matryoshka LoRA kernels on %d LoRA modules%s",
                    sum(hasattr(m, "_rankfuse_pack") for m in self.modules.values()),
                    "" if all_rankfuse else " (others use vLLM's kernels)")

    orig_activate = mm.LoRAModelManager.activate_adapter

    def activate_adapter(self, lora_id: int) -> bool:
        result = orig_activate(self, lora_id)
        if result:
            index = self.lora_index_to_id.index(lora_id)
            for module in self.modules.values():
                pack = getattr(module, "_rankfuse_pack", None)
                if pack is not None:
                    A, B = pack
                    for s, (a, b) in enumerate(zip(module.lora_a_stacked, module.lora_b_stacked)):
                        A[index, s].copy_(a[index, 0])
                        B[index, s, :, : b.shape[2]].copy_(b[index, 0].T)
        return result

    mm.LoRAModelManager._create_lora_modules = _create_lora_modules
    mm.LoRAModelManager.activate_adapter = activate_adapter

    # ---- the kernel swap -------------------------------------------------------------------
    orig_apply_lora = BaseLinearLayerWithLoRA._apply_lora_to_output

    def _apply_lora_to_output(self, x, output):
        weights = getattr(self, "_rankfuse_weights", None)
        if weights is None:
            return orig_apply_lora(self, x, output)
        original_shape = output.shape if output.ndim == 3 else None
        if x.ndim == 3 and output.ndim == 3:
            output = output.flatten(0, 1)
            x = x.flatten(0, 1)
        self.punica_wrapper.add_lora_linear(
            output, x, self.lora_a_stacked, self.lora_b_stacked, 1.0, self.output_slices,
            rankfuse_weights=weights,
        )
        if original_shape is not None:
            output = output.reshape(original_shape)
        return output

    BaseLinearLayerWithLoRA._apply_lora_to_output = _apply_lora_to_output
    logger.info("RankFuse vLLM plugin active (RANKFUSE_VLLM=1)")
