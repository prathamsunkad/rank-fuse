"""B3: a Matryoshka adapter served the way stock vLLM serves it, with vLLM's own Triton kernels.

vLLM does not know that ranks are nested, so every rank tier (1, 2, 4, ..., r_max) has to be
registered as its own adapter. Like vLLM's LoRA manager, each copy is stored padded to
max_lora_rank (= r_max) in stacked buffers, in vLLM's layouts (docs/VLLM_LORA_NOTES.md,
sections 2 and 4):

    A [num_copies, r_max, h_in]     copy k = A_t[:r_k], zero rows below
    B [num_copies, h_out, r_max]    copy k = B[:r_k]^T * scale(r_k), zero columns after

The duplicated, zero-padded storage is what this baseline measures (Claim 1): the shrink reads
r_max rows of every active copy. Each token is mapped to the copy for its rank (rank 0 -> -1,
no LoRA). vLLM's LoRAKernelMeta builds the per-step metadata (once per step, before the layers
run), then _lora_shrink (split-K into an fp32 buffer) and _lora_expand (add_inputs=True: out +=
delta, as vLLM's LoRA layers call it) compute the delta. Scales are folded into B, as vLLM folds
lora_alpha / r into its stored weights.

vLLM's shrink takes fp16/bf16 only. Measured against vLLM 0.32.0 on an H200 (4096 x 4096, r_max
64, add-into-output, CUDA-graph replay): rf1_fused is 1.9-2.5x faster at decode and 2.0-2.8x on
prompts; at T=32 B3 reads 7.9 MB from HBM against rf1_fused's 1.7 MB.

vLLM is imported on first use, so `import rankfuse` stays fast and works without vLLM.
"""

from __future__ import annotations

from importlib.util import find_spec

import torch

from rankfuse.interface import MatryoshkaLoRA, register
from rankfuse.workloads import default_tiers


def _vllm_ops():
    from vllm.lora.ops.triton_ops import LoRAKernelMeta
    from vllm.lora.ops.triton_ops.lora_expand_op import _lora_expand
    from vllm.lora.ops.triton_ops.lora_shrink_op import _lora_shrink

    return LoRAKernelMeta, _lora_shrink, _lora_expand


class AdapterCopies:
    """One Matryoshka adapter registered as one padded vLLM adapter per rank in `ranks`."""

    def __init__(self, lora: MatryoshkaLoRA, ranks: list[int]):
        r_max, h_in = lora.A_t.shape
        dev, dt = lora.A_t.device, lora.A_t.dtype
        self.ranks = sorted(set(ranks))
        self.r_max = r_max
        n = len(self.ranks)
        self.A = torch.zeros(n, r_max, h_in, device=dev, dtype=dt)
        self.B = torch.zeros(n, lora.h_out, r_max, device=dev, dtype=dt)
        scales = lora.scale_per_rank or {}
        for i, r in enumerate(self.ranks):
            self.A[i, :r] = lora.A_t[:r]
            self.B[i, :, :r] = (lora.B[:r].float() * scales.get(r, 1.0)).T.to(dt)
        self.copy_of_rank = torch.full((r_max + 1,), -1, dtype=torch.int32, device=dev)
        for i, r in enumerate(self.ranks):
            self.copy_of_rank[r] = i


class Step:
    """Per-step metadata, shared by every layer whose copies use the same rank list."""

    def __init__(self, copy_of_rank: torch.Tensor, num_copies: int, r_max: int, token_ranks: torch.Tensor):
        LoRAKernelMeta, _, _ = _vllm_ops()
        T = token_ranks.numel()
        self.num_tokens = T
        self.meta = LoRAKernelMeta.make(max_loras=num_copies, max_num_tokens=max(T, 1), device=token_ranks.device)
        self.meta.prepare_tensors(copy_of_rank[token_ranks.long()])
        self.args = self.meta.meta_args(T, specialize_active_lora=False)
        self.buf = torch.zeros((1, T, r_max), dtype=torch.float32, device=token_ranks.device)


def apply(x: torch.Tensor, copies: AdapterCopies, step: Step, out: torch.Tensor, add: bool = True) -> torch.Tensor:
    """out (+)= delta with vLLM's shrink and expand kernels. Rows of rank-0 tokens are not touched."""
    _, lora_shrink, lora_expand = _vllm_ops()
    lora_shrink(x, [copies.A], step.buf, *step.args, 1.0)
    lora_expand(step.buf, [copies.B], out, *step.args, offset_start=0, add_inputs=add)
    return out


def _copies_for(lora: MatryoshkaLoRA, present: list[int]) -> AdapterCopies:
    """The adapter's copies, built once: the default tiers plus any other rank seen so far."""
    key = (lora.A_t.data_ptr(), lora.B.data_ptr(), tuple(sorted((lora.scale_per_rank or {}).items())))
    cached = getattr(lora, "_b3_copies", None)
    if cached is None or cached[0] != key or not set(present) <= set(cached[1].ranks):
        ranks = set(default_tiers(lora.r_max)) | set(present)
        if cached is not None and cached[0] == key:
            ranks |= set(cached[1].ranks)
        cached = (key, AdapterCopies(lora, sorted(ranks)))
        lora._b3_copies = cached
    return cached[1]


def prepare(x: torch.Tensor, lora: MatryoshkaLoRA, token_ranks: torch.Tensor) -> dict:
    """Copies and per-step metadata, built outside the timed (or captured) call."""
    present = [r for r in torch.unique(token_ranks).tolist() if r > 0]
    copies = _copies_for(lora, present)
    if not present:
        return dict(copies=copies, step=None)
    return dict(copies=copies, step=Step(copies.copy_of_rank, len(copies.ranks), copies.r_max, token_ranks))


if find_spec("vllm") is not None:

    @register("b3_vllm", requires_cuda=True, dtypes=(torch.float16, torch.bfloat16), prepare=prepare)
    def b3_vllm(
        x: torch.Tensor,
        lora: MatryoshkaLoRA,
        token_ranks: torch.Tensor,
        copies: AdapterCopies | None = None,
        step: Step | None = None,
        out: torch.Tensor | None = None,
        add: bool = False,
    ) -> torch.Tensor:
        """B3. Without copies/step, builds them here (one host sync for the ranks present)."""
        if copies is None:
            meta = prepare(x, lora, token_ranks)
            copies, step = meta["copies"], meta["step"]
        if out is None:
            if add:
                raise ValueError("add=True needs an existing out to add into")
            out = torch.zeros((x.shape[0], lora.h_out), dtype=x.dtype, device=x.device)
        elif not add:
            out.zero_()
        if step is None or x.shape[0] == 0:  # every token at rank 0
            return out
        return apply(x, copies, step, out, add=True)
