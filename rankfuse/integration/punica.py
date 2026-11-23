"""RankFusePunicaWrapper: vLLM's GPU Punica wrapper plus a per-token rank buffer and the RankFuse op.

vLLM keeps all per-step LoRA metadata in one wrapper object shared by every LoRA layer, in
device buffers allocated once (so CUDA graphs can replay them) and refilled each step by
`update_metadata`. This subclass adds:

    _token_lora_ranks  int32 [max_num_batched_tokens] on the GPU: each token's Matryoshka rank,
                       0 for tokens without an adapter (or whose adapter is not in a slot).
                       Refilled each step with one host-to-device copy from pinned memory.
    _r_limit_cpu       int32 [1] on the CPU: the step's maximum rank (0 = no LoRA this step).
    scale_table        optional fp32 [max_lora_rank + 1] per-rank scale (RANKFUSE_RANK_SCALE).

and replaces `add_lora_linear` (the one call every dense LoRA layer makes) with one
`torch.ops.rankfuse.lora_linear` launch for layers the plugin prepared (they pass
`rankfuse_weights`). Everything else (logits, embeddings, MoE, layers without RankFuse
weights, max_loras > 1) is the stock PunicaWrapperGPU path.

Stage 2 (several Matryoshka adapters per batch): `_rankfuse_apply` is the only place that
calls the kernel; a multi-adapter kernel would take the per-token slot buffer
(`self._token_lora_indices`, already filled by the base class) next to `_token_lora_ranks`
and all slots of the layer's RankFuse weights instead of slot 0.
"""

from typing import Any

import numpy as np
import torch

from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU
from vllm.utils.gpu_sync_debug import gpu_sync_allowed

from rankfuse.integration import ops

# A request without matryoshka_rank uses the whole adapter: its rank is clamped to max_lora_rank.
FULL_RANK = 1 << 30


def rank_scale_table(mode: str, r_max: int, device) -> torch.Tensor | None:
    """Per-rank scale applied on top of vLLM's load-time scaling (alpha / r_max folded into B).

    "none"          scale 1 at every rank: adapters whose every rank uses alpha == r, like
                    ours (each tier copy has lora_alpha == r), so stock vLLM also uses 1.
    "alpha_over_r"  a Matryoshka adapter trained with one fixed alpha, where rank r should use
                    alpha / r: relative to the folded alpha / r_max that is r_max / r.
    """
    if mode in ("", "none"):
        return None
    if mode != "alpha_over_r":
        raise ValueError(f"RANKFUSE_RANK_SCALE must be 'none' or 'alpha_over_r', got {mode!r}")
    table = torch.zeros(r_max + 1, dtype=torch.float32)
    table[1:] = r_max / torch.arange(1, r_max + 1, dtype=torch.float32)
    return table.to(device)


class RankFusePunicaWrapper(PunicaWrapperGPU):
    def __init__(self, max_num_batched_tokens: int, max_batches: int, device: torch.device | str, **kwargs):
        super().__init__(max_num_batched_tokens, max_batches, device, **kwargs)
        import os

        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_lora_rank = self.lora_config.max_lora_rank
        self._token_lora_ranks = torch.zeros(max_num_batched_tokens, dtype=torch.int32, device=device)
        self._r_limit_cpu = torch.zeros(1, dtype=torch.int32)
        self._ranks_filled = 0  # length of the nonzero prefix written last step
        self.scale_table = rank_scale_table(os.environ.get("RANKFUSE_RANK_SCALE", "none"), self.max_lora_rank, device)
        # Set by the plugin once every LoRA module of the model is a RankFuse-prepared linear
        # layer: the stock kernel metadata (two torch.unique/sort passes and a host sync per
        # step) is then unused and not built.
        self.rankfuse_only = False

    # ------------------------------------------------------------------ per-step metadata

    def update_metadata(self, mapping, lora_index_to_id, max_loras, vocab_size, **kwargs):
        self.is_prefill = mapping.is_prefill
        self._update_base_metadata(mapping, lora_index_to_id, max_loras, vocab_size)
        self._update_rank_metadata(mapping, lora_index_to_id)
        if self.rankfuse_only and self.max_loras == 1:
            return
        with gpu_sync_allowed():
            self.token_mapping_meta.prepare_tensors(self.token_lora_indices)
            self.prompt_mapping_meta.prepare_tensors(self.sampler_indices)

    def _update_rank_metadata(self, mapping, lora_index_to_id) -> None:
        ids = np.asarray(mapping.index_mapping, dtype=np.int64)
        n = ids.shape[0]
        ranks = getattr(mapping, "index_rank_mapping", ())
        if len(ranks) == n:
            ranks = np.asarray(ranks, dtype=np.int64)
        else:  # a mapping built without ranks (e.g. by a path the plugin does not patch): full rank
            ranks = np.where(ids > 0, FULL_RANK, 0)
        in_slot = [i for i in lora_index_to_id if i is not None]
        ranks = np.where(np.isin(ids, in_slot) if in_slot else False, np.minimum(ranks, self.max_lora_rank), 0)
        r_limit = int(ranks.max()) if n else 0
        # Zero what the previous step wrote beyond this step's tokens, so padded tokens (CUDA
        # graphs pad the batch) read rank 0.
        hi = max(n, self._ranks_filled)
        buf = np.zeros(hi, dtype=np.int32)
        buf[:n] = ranks
        if hi:
            self._token_lora_ranks[:hi].copy_(torch.from_numpy(buf).pin_memory(), non_blocking=True)
        self._ranks_filled = n
        self._r_limit_cpu[0] = r_limit

    # ------------------------------------------------------------------ the kernel swap

    def add_lora_linear(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        scale: float,
        output_slices: tuple[int, ...],
        *,
        buffer: torch.Tensor | None = None,
        rankfuse_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> None:
        if rankfuse_weights is None or self.max_loras != 1:
            return super().add_lora_linear(
                y, x, lora_a_stacked, lora_b_stacked, scale, output_slices, buffer=buffer, **kwargs
            )
        assert scale == 1.0, "vLLM folds the LoRA scaling into B at load time"
        self._rankfuse_apply(y, x, rankfuse_weights, output_slices, kwargs.get("add_inputs", True))

    def _rankfuse_apply(self, y, x, rankfuse_weights, output_slices, add: bool) -> None:
        # Stage 1: one adapter slot (slot 0). See the module docstring for Stage 2.
        A_t, B = rankfuse_weights
        ops.lora_linear(
            y, x.view(-1, x.shape[-1]), A_t, B, self._token_lora_ranks, self._r_limit_cpu,
            list(output_slices), self.scale_table, add, self.max_num_batched_tokens,
        )
