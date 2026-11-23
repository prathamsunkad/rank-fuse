"""B3-multi: many Matryoshka adapters served the way stock vLLM 0.32 serves them, with its own kernels.

vLLM does not know that ranks are nested, so every (adapter, rank) pair a request asks for is a
separate LoRA adapter in a slot of its own, stored padded to max_lora_rank (= r_max) in vLLM's
stacked layouts (as Stage 1's B3, rankfuse/baselines/vllm_ops.py, does for one adapter):

    A [num_pairs, r_max, h_in]        pair p = (a, r): A[a, :r], zero rows below
    B [num_pairs, h_out, r_max]       B[a, :r]^T * scale(a, r), zero columns after

Only the pairs present in the step are built (as if vLLM had a slot for every one of them, the
most favourable case for it: no adapter loads from CPU). Each token maps to its pair's slot
(-1: no LoRA); vLLM's LoRAKernelMeta sorts tokens by slot once per step, then _lora_shrink
(split-K, fp32 buffer) and _lora_expand (add_inputs=True) run, one launch each for all slices.
Merged slices are passed as per-slice weight lists, as vLLM's merged layers do.

vLLM's ops take fp16/bf16 only.
"""

from __future__ import annotations

import torch

from rankfuse.baselines.vllm_ops import _vllm_ops
from rankfuse.multi import PagedWeights, output_buffer, register_multi


def _clear_ptr_cache():
    # vLLM memoizes weight pointers by data_ptr only (docs/VLLM_LORA_NOTES.md, section 7); copies of
    # a new configuration can reuse a freed address with another shape, so start empty.
    from vllm.lora.ops.triton_ops import utils

    utils._LORA_A_PTR_DICT.clear()
    utils._LORA_B_PTR_DICT.clear()


class PairCopies:
    """One padded vLLM adapter per (adapter, rank) pair in `pairs`."""

    def __init__(self, W: PagedWeights, pairs: list[tuple[int, int]]):
        dev, dt, r_max = W.device, W.dtype, W.r_max
        pa = torch.tensor([a for a, _ in pairs], dtype=torch.long, device=dev)
        pr = torch.tensor([r for _, r in pairs], dtype=torch.long, device=dev)
        mask = (torch.arange(r_max, device=dev) < pr[:, None]).float()  # [P, r_max]
        if W.scale_table is not None:
            scale = W.scale_table[pa, pr].float()
        else:
            scale = torch.ones(len(pairs), device=dev)
        self.A, self.B = [], []
        for s, n in enumerate(W.slice_sizes):
            A = W.A_dense[pa, s] * mask[:, :, None].to(dt)  # [P, r_max, h_in]
            B = (W.B_dense[pa, s, :, :n].float() * (mask * scale[:, None])[:, :, None]).to(dt)
            self.A.append(A.contiguous())
            self.B.append(B.transpose(1, 2).contiguous())  # [P, n, r_max]
        self.num = len(pairs)
        self.r_max = r_max


class Step:
    def __init__(self, num_pairs: int, slot: torch.Tensor, n_slices: int, r_max: int):
        LoRAKernelMeta, _, _ = _vllm_ops()
        T = slot.numel()
        self.meta = LoRAKernelMeta.make(max_loras=num_pairs, max_num_tokens=max(T, 1), device=slot.device)
        self.meta.prepare_tensors(slot)
        self.args = self.meta.meta_args(T, specialize_active_lora=False)
        self.buf = torch.zeros((n_slices, T, r_max), dtype=torch.float32, device=slot.device)


def prepare(x, W: PagedWeights, token_adapter, token_ranks) -> dict:
    """Pair copies and vLLM's per-step metadata, built outside the timed (or captured) call."""
    ad = token_adapter.long().cpu()
    rk = token_ranks.long().cpu()
    live = (rk > 0) & (ad >= 0)
    if not live.any():
        return dict(copies=None, step=None)
    uniq, inv = torch.unique(torch.stack([ad[live], rk[live]], 1), dim=0, return_inverse=True)
    pairs = [tuple(p) for p in uniq.tolist()]
    slot = torch.full((x.shape[0],), -1, dtype=torch.int32)
    slot[live] = inv.to(torch.int32)
    _clear_ptr_cache()
    copies = PairCopies(W, pairs)
    return dict(copies=copies, step=Step(copies.num, slot.to(x.device), W.n_slices, W.r_max))


@register_multi("b3_multi", prepare=prepare, dtypes=(torch.float16, torch.bfloat16))
def b3_multi(x, W: PagedWeights, token_adapter, token_ranks, copies: PairCopies | None = None, step: Step | None = None,
             out=None, add=False):
    if copies is None and step is None:
        meta = prepare(x, W, token_adapter, token_ranks)
        copies, step = meta["copies"], meta["step"]
    out = output_buffer(x, W, out, add)
    if not add:
        out.zero_()
    if step is None or x.shape[0] == 0:  # every token at rank 0
        return out
    _, lora_shrink, lora_expand = _vllm_ops()
    lora_shrink(x, copies.A, step.buf, *step.args, 1.0)
    lora_expand(step.buf, copies.B, out, *step.args, offset_start=0, add_inputs=True)
    return out
