"""Reference multi-adapter baselines used by correctness and paging experiments.

Provides:
- b0_ma: FP32 reference oracle
- b1_ma: existing PyTorch baseline
- b1_ma_padded: real per-adapter padded baseline (no per-token weight gathers)
- b3_ma: vLLM's LoRA kernels with duplicate padded copy per (adapter, rank)
- b4_ma_copies: S-LoRA BGMV kernel with duplicate unpadded copy per (adapter, rank)
- b4_ma_shared: S-LoRA BGMV kernel with one shared copy per adapter
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import importlib.util
import os
import torch

from rankfuse.interface import MatryoshkaLoRA
from rankfuse.baselines.slora_bgmv import bgmv_forward
from rankfuse.workloads import default_tiers


def _adapter(
    adapters: Mapping[int, MatryoshkaLoRA] | Sequence[MatryoshkaLoRA] | tuple[torch.Tensor, torch.Tensor],
    index: int,
) -> MatryoshkaLoRA:
    if isinstance(adapters, tuple) and len(adapters) == 2 and isinstance(adapters[0], torch.Tensor):
        a_pool, b_pool = adapters
        if a_pool.dim() == 4:
            a_pool = a_pool.squeeze(0)
        if b_pool.dim() == 4:
            b_pool = b_pool.squeeze(0)
        return MatryoshkaLoRA(a_pool[index], b_pool[index])
    return adapters[index] if not isinstance(adapters, Mapping) else adapters[index]


def _first_adapter(
    adapters: Mapping[int, MatryoshkaLoRA] | Sequence[MatryoshkaLoRA] | tuple[torch.Tensor, torch.Tensor],
) -> MatryoshkaLoRA:
    if isinstance(adapters, tuple) and len(adapters) == 2 and isinstance(adapters[0], torch.Tensor):
        a_pool, b_pool = adapters
        if a_pool.dim() == 4:
            a_pool = a_pool.squeeze(0)
        if b_pool.dim() == 4:
            b_pool = b_pool.squeeze(0)
        return MatryoshkaLoRA(a_pool[0], b_pool[0])
    return next(iter(adapters.values())) if isinstance(adapters, Mapping) else adapters[0]


def _adapter_count(
    adapters: Mapping[int, MatryoshkaLoRA] | Sequence[MatryoshkaLoRA] | tuple[torch.Tensor, torch.Tensor],
) -> int:
    if isinstance(adapters, tuple) and len(adapters) == 2 and isinstance(adapters[0], torch.Tensor):
        a_pool = adapters[0]
        if a_pool.dim() == 4:
            return a_pool.shape[1]
        return a_pool.shape[0]
    return len(adapters)


def _extract_pools(
    adapters: Mapping[int, MatryoshkaLoRA] | Sequence[MatryoshkaLoRA] | tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """Return (a_pool, b_pool, num_adapters, r_max, h_out).

    a_pool: [num_adapters, r_max, h_in]
    b_pool: [num_adapters, r_max, h_out]
    """
    if isinstance(adapters, tuple) and len(adapters) == 2 and isinstance(adapters[0], torch.Tensor):
        a_pool, b_pool = adapters
        if a_pool.dim() == 4:
            a_pool = a_pool.squeeze(0)
        if b_pool.dim() == 4:
            b_pool = b_pool.squeeze(0)
        num_adapters, r_max, _ = a_pool.shape
        h_out = b_pool.shape[2]
        return a_pool.to(device=device, dtype=dtype), b_pool.to(device=device, dtype=dtype), num_adapters, r_max, h_out

    num_adapters = len(adapters)
    first = _first_adapter(adapters)
    r_max, h_in, h_out = first.r_max, first.h_in, first.h_out
    a_pool = torch.zeros(num_adapters, r_max, h_in, device=device, dtype=dtype)
    b_pool = torch.zeros(num_adapters, r_max, h_out, device=device, dtype=dtype)
    for i in range(num_adapters):
        item = _adapter(adapters, i)
        a_pool[i] = item.A_t.to(device=device, dtype=dtype)
        b_pool[i] = item.B.to(device=device, dtype=dtype)
    return a_pool, b_pool, num_adapters, r_max, h_out


# =====================================================================
# 0. Reference Oracle (FP32)
# =====================================================================

def b0_ma(
    x: torch.Tensor,
    adapters: Mapping[int, MatryoshkaLoRA] | Sequence[MatryoshkaLoRA] | tuple[torch.Tensor, torch.Tensor],
    adapter_ids: torch.Tensor,
    token_ranks: torch.Tensor,
) -> torch.Tensor:
    """FP32 per-token oracle for a batch containing multiple adapters."""
    if x.shape[0] != adapter_ids.numel() or x.shape[0] != token_ranks.numel():
        raise ValueError("x, adapter_ids and token_ranks must have the same token count")
    first = _first_adapter(adapters)
    out = torch.zeros(
        x.shape[0], first.h_out,
        device=x.device, dtype=torch.float32
    )
    for i in range(x.shape[0]):
        lora = _adapter(adapters, int(adapter_ids[i]))
        rank = int(token_ranks[i])
        if rank <= 0:
            continue
        if not 0 < rank <= lora.r_max:
            raise ValueError(f"rank {rank} is outside adapter {int(adapter_ids[i])}'s range")
        out[i] = x[i].float() @ lora.A_t[:rank].float().T @ lora.B[:rank].float()
        scale = lora.scale_per_rank.get(rank, 1.0) if lora.scale_per_rank else 1.0
        out[i] *= scale
    return out.to(x.dtype)


# =====================================================================
# 1. Existing PyTorch Baseline (b1_ma)
# =====================================================================

def b1_ma(
    x: torch.Tensor,
    adapters: Mapping[int, MatryoshkaLoRA] | Sequence[MatryoshkaLoRA] | tuple[torch.Tensor, torch.Tensor],
    adapter_ids: torch.Tensor,
    token_ranks: torch.Tensor,
) -> torch.Tensor:
    """Per-adapter padded PyTorch baseline with one shared weight copy per adapter."""
    out = torch.zeros(
        x.shape[0], _first_adapter(adapters).h_out,
        device=x.device, dtype=x.dtype
    )
    for adapter_id in torch.unique(adapter_ids, sorted=True).tolist():
        rows = adapter_ids == adapter_id
        lora = _adapter(adapters, int(adapter_id))
        limit = int(token_ranks[rows].max())
        if limit <= 0:
            continue
        v = x[rows] @ lora.A_t[:limit].T
        ranks = token_ranks[rows]
        cols = torch.arange(limit, device=x.device)
        v = v.masked_fill(cols.unsqueeze(0) >= ranks.unsqueeze(1), 0)
        out[rows] = v @ lora.B[:limit]
        if lora.scale_per_rank:
            scales = torch.tensor(
                [lora.scale_per_rank.get(int(rank), 1.0) for rank in ranks.tolist()],
                device=x.device, dtype=out.dtype
            )
            out[rows] *= scales[:, None]
    return out


# =====================================================================
# 2. Real Padded Multi-Adapter Baseline (b1_ma_padded)
# =====================================================================

def b1_ma_padded(
    x: torch.Tensor,
    adapters: Mapping[int, MatryoshkaLoRA] | Sequence[MatryoshkaLoRA] | tuple[torch.Tensor, torch.Tensor],
    adapter_ids: torch.Tensor,
    token_ranks: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    add: bool = False,
    prepared: dict | None = None,
) -> torch.Tensor:
    """Per adapter pad to its max rank in the batch, mask, two matmuls.

    No per-token weight gathers. Graph-safe when `prepared` metadata is supplied.
    """
    first = _first_adapter(adapters)
    h_out = first.h_out
    if out is None:
        out = torch.zeros(x.shape[0], h_out, device=x.device, dtype=x.dtype)
    elif not add:
        out.zero_()

    if x.shape[0] == 0:
        return out

    if prepared is not None:
        # Fast path for CUDA graph replay
        a_pool = prepared["a_pool"]
        b_pool = prepared["b_pool"]
        slices = prepared["slices"]  # list of (adapter_idx, start, end, max_rank)
        for a_idx, start, end, r_lim in slices:
            if start >= end or r_lim <= 0:
                continue
            x_sub = x[start:end]
            v = x_sub @ a_pool[a_idx, :r_lim].T
            r_sub = token_ranks[start:end]
            cols = torch.arange(r_lim, device=x.device)
            v = v.masked_fill(cols.unsqueeze(0) >= r_sub.unsqueeze(1), 0)
            y = v @ b_pool[a_idx, :r_lim]
            if add:
                out[start:end] += y
            else:
                out[start:end] = y
        return out

    # Eager path
    for adapter_id in torch.unique(adapter_ids, sorted=True).tolist():
        rows = adapter_ids == adapter_id
        if not rows.any():
            continue
        lora = _adapter(adapters, int(adapter_id))
        limit = int(token_ranks[rows].max())
        if limit <= 0:
            continue
        v = x[rows] @ lora.A_t[:limit].T
        ranks = token_ranks[rows]
        cols = torch.arange(limit, device=x.device)
        v = v.masked_fill(cols.unsqueeze(0) >= ranks.unsqueeze(1), 0)
        y = v @ lora.B[:limit]
        if lora.scale_per_rank:
            scales = torch.tensor(
                [lora.scale_per_rank.get(int(r), 1.0) for r in ranks.tolist()],
                device=x.device, dtype=out.dtype
            )
            y *= scales[:, None]
        if add:
            out[rows] += y
        else:
            out[rows] = y
    return out


def prepare_b1_ma_padded(
    x: torch.Tensor,
    adapters: Mapping[int, MatryoshkaLoRA] | Sequence[MatryoshkaLoRA] | tuple[torch.Tensor, torch.Tensor],
    adapter_ids: torch.Tensor,
    token_ranks: torch.Tensor,
) -> dict:
    """Build graph-safe slice metadata for b1_ma_padded."""
    a_pool, b_pool, _, _, _ = _extract_pools(adapters, x.device, x.dtype)
    slices = []
    # Identify contiguous slices of the same adapter
    diffs = torch.where(adapter_ids[:-1] != adapter_ids[1:])[0] + 1
    starts = torch.cat([torch.tensor([0], device=x.device), diffs])
    ends = torch.cat([diffs, torch.tensor([adapter_ids.numel()], device=x.device)])
    for s, e in zip(starts.tolist(), ends.tolist()):
        a_id = int(adapter_ids[s])
        r_lim = int(token_ranks[s:e].max()) if s < e else 0
        slices.append((a_id, s, e, r_lim))
    return {"a_pool": a_pool, "b_pool": b_pool, "slices": slices}


# =====================================================================
# 3. vLLM's Multi-Adapter Baseline (b3_ma)
# =====================================================================

def _get_vllm_ops():
    if importlib.util.find_spec("vllm") is None:
        return None, None, None
    from vllm.lora.ops.triton_ops import LoRAKernelMeta
    from vllm.lora.ops.triton_ops.lora_expand_op import _lora_expand
    from vllm.lora.ops.triton_ops.lora_shrink_op import _lora_shrink
    return LoRAKernelMeta, _lora_shrink, _lora_expand


class VLLMMultiAdapterCopies:
    """Every (adapter, rank) pair registered as a separate padded copy in stacked buffers."""

    def __init__(
        self,
        adapters: Mapping[int, MatryoshkaLoRA] | Sequence[MatryoshkaLoRA] | tuple[torch.Tensor, torch.Tensor],
        ranks: Sequence[int],
        device: torch.device,
        dtype: torch.dtype,
    ):
        num_adapters = _adapter_count(adapters)
        first = _first_adapter(adapters)
        r_max, h_in, h_out = first.r_max, first.h_in, first.h_out
        self.ranks = sorted(set(ranks))
        self.num_adapters = num_adapters
        self.r_max = r_max
        self.num_copies = num_adapters * len(self.ranks)

        self.A = torch.zeros(self.num_copies, r_max, h_in, device=device, dtype=dtype)
        self.B = torch.zeros(self.num_copies, h_out, r_max, device=device, dtype=dtype)

        self.copy_map: dict[tuple[int, int], int] = {}
        copy_idx = 0
        for a_idx in range(num_adapters):
            lora = _adapter(adapters, a_idx)
            scales = lora.scale_per_rank or {}
            for r in self.ranks:
                self.A[copy_idx, :r] = lora.A_t[:r].to(device=device, dtype=dtype)
                scaled_b = lora.B[:r].float() * scales.get(r, 1.0)
                self.B[copy_idx, :, :r] = scaled_b.T.to(device=device, dtype=dtype)
                self.copy_map[(a_idx, r)] = copy_idx
                copy_idx += 1


def prepare_b3_ma(
    x: torch.Tensor,
    adapters: Mapping[int, MatryoshkaLoRA] | Sequence[MatryoshkaLoRA] | tuple[torch.Tensor, torch.Tensor],
    adapter_ids: torch.Tensor,
    token_ranks: torch.Tensor,
) -> dict:
    """Build copies and Step metadata for b3_ma before graph capture."""
    first = _first_adapter(adapters)
    tiers = default_tiers(first.r_max)
    present_ranks = set(torch.unique(token_ranks).tolist()) - {0}
    all_ranks = sorted(set(tiers) | present_ranks)
    copies = VLLMMultiAdapterCopies(adapters, all_ranks, x.device, x.dtype)

    T = token_ranks.numel()
    copy_indices = torch.full((T,), -1, dtype=torch.int32, device=x.device)
    for i in range(T):
        a_id = int(adapter_ids[i])
        r = int(token_ranks[i])
        if r > 0 and (a_id, r) in copies.copy_map:
            copy_indices[i] = copies.copy_map[(a_id, r)]

    LoRAKernelMeta, _, _ = _get_vllm_ops()
    if LoRAKernelMeta is not None and x.is_cuda:
        meta = LoRAKernelMeta.make(max_loras=copies.num_copies, max_num_tokens=max(T, 1), device=x.device)
        meta.prepare_tensors(copy_indices)
        args = meta.meta_args(T, specialize_active_lora=False)
        buf = torch.zeros((1, T, copies.r_max), dtype=torch.float32, device=x.device)
    else:
        meta, args, buf = None, None, None

    return {
        "copies": copies,
        "copy_indices": copy_indices,
        "meta": meta,
        "args": args,
        "buf": buf,
    }


def b3_ma(
    x: torch.Tensor,
    adapters: Mapping[int, MatryoshkaLoRA] | Sequence[MatryoshkaLoRA] | tuple[torch.Tensor, torch.Tensor],
    adapter_ids: torch.Tensor,
    token_ranks: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    add: bool = False,
    prepared: dict | None = None,
) -> torch.Tensor:
    """vLLM multi-adapter baseline with every (adapter, rank) a separate padded copy."""
    first = _first_adapter(adapters)
    h_out = first.h_out
    if out is None:
        out = torch.zeros(x.shape[0], h_out, device=x.device, dtype=x.dtype)
    elif not add:
        out.zero_()

    if x.shape[0] == 0:
        return out

    if prepared is None:
        prepared = prepare_b3_ma(x, adapters, adapter_ids, token_ranks)

    LoRAKernelMeta, lora_shrink, lora_expand = _get_vllm_ops()
    if LoRAKernelMeta is not None and x.is_cuda and prepared["args"] is not None:
        copies = prepared["copies"]
        args = prepared["args"]
        buf = prepared["buf"]
        lora_shrink(x, [copies.A], buf, *args, 1.0)
        lora_expand(buf, [copies.B], out, *args, offset_start=0, add_inputs=add)
        return out

    # Eager fallback
    copies = prepared["copies"]
    copy_indices = prepared["copy_indices"]
    for i in range(x.shape[0]):
        c = int(copy_indices[i])
        if c < 0:
            continue
        v = x[i].float() @ copies.A[c].float().T
        y = v @ copies.B[c].float().T
        if add:
            out[i] += y.to(out.dtype)
        else:
            out[i] = y.to(out.dtype)
    return out


# =====================================================================
# 4. S-LoRA Multi-Adapter Baselines (b4_ma_copies and b4_ma_shared)
# =====================================================================

class SLoRACopiesPool:
    """One unpadded copy per (adapter, rank) in a contiguous pool."""

    def __init__(
        self,
        adapters: Mapping[int, MatryoshkaLoRA] | Sequence[MatryoshkaLoRA] | tuple[torch.Tensor, torch.Tensor],
        ranks: Sequence[int],
        device: torch.device,
        dtype: torch.dtype,
    ):
        num_adapters = _adapter_count(adapters)
        first = _first_adapter(adapters)
        r_max, h_in, h_out = first.r_max, first.h_in, first.h_out
        self.ranks = sorted(set(ranks))
        self.num_adapters = num_adapters
        self.r_max = r_max
        self.num_copies = num_adapters * len(self.ranks)

        self.A = torch.zeros(self.num_copies, r_max, h_in, device=device, dtype=dtype)
        self.B = torch.zeros(self.num_copies, r_max, h_out, device=device, dtype=dtype)

        self.copy_map: dict[tuple[int, int], int] = {}
        copy_idx = 0
        for a_idx in range(num_adapters):
            lora = _adapter(adapters, a_idx)
            for r in self.ranks:
                self.A[copy_idx, :r] = lora.A_t[:r].to(device=device, dtype=dtype)
                self.B[copy_idx, :r] = lora.B[:r].to(device=device, dtype=dtype)
                self.copy_map[(a_idx, r)] = copy_idx
                copy_idx += 1


def prepare_b4_ma_copies(
    x: torch.Tensor,
    adapters: Mapping[int, MatryoshkaLoRA] | Sequence[MatryoshkaLoRA] | tuple[torch.Tensor, torch.Tensor],
    adapter_ids: torch.Tensor,
    token_ranks: torch.Tensor,
) -> dict:
    """Build unpadded copies pool and copy indices for b4_ma_copies."""
    first = _first_adapter(adapters)
    tiers = default_tiers(first.r_max)
    present_ranks = set(torch.unique(token_ranks).tolist()) - {0}
    all_ranks = sorted(set(tiers) | present_ranks)
    pool = SLoRACopiesPool(adapters, all_ranks, x.device, x.dtype)

    T = token_ranks.numel()
    copy_indices = torch.full((T,), -1, dtype=torch.int32, device=x.device)
    for i in range(T):
        a_id = int(adapter_ids[i])
        r = int(token_ranks[i])
        if r > 0 and (a_id, r) in pool.copy_map:
            copy_indices[i] = pool.copy_map[(a_id, r)]

    scales = torch.ones(pool.r_max + 1, dtype=torch.float32, device=x.device)
    if first.scale_per_rank:
        for r, s in first.scale_per_rank.items():
            if r <= pool.r_max:
                scales[r] = s

    v_buf = torch.zeros(T, pool.r_max, dtype=torch.float32, device=x.device)
    return {
        "pool": pool,
        "copy_indices": copy_indices,
        "scales": scales,
        "v_buf": v_buf,
    }


def b4_ma_copies(
    x: torch.Tensor,
    adapters: Mapping[int, MatryoshkaLoRA] | Sequence[MatryoshkaLoRA] | tuple[torch.Tensor, torch.Tensor],
    adapter_ids: torch.Tensor,
    token_ranks: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    add: bool = False,
    prepared: dict | None = None,
) -> torch.Tensor:
    """S-LoRA baseline with one copy per (adapter, rank)."""
    if prepared is None:
        prepared = prepare_b4_ma_copies(x, adapters, adapter_ids, token_ranks)

    pool = prepared["pool"]
    return bgmv_forward(
        x, pool.A, pool.B, prepared["copy_indices"], token_ranks,
        scales=prepared["scales"], out=out, add=add, v_buf=prepared["v_buf"]
    )


def prepare_b4_ma_shared(
    x: torch.Tensor,
    adapters: Mapping[int, MatryoshkaLoRA] | Sequence[MatryoshkaLoRA] | tuple[torch.Tensor, torch.Tensor],
    token_ranks: torch.Tensor,
) -> dict:
    """Build shared weight pool for b4_ma_shared."""
    a_pool, b_pool, _, r_max, _ = _extract_pools(adapters, x.device, x.dtype)
    first = _first_adapter(adapters)
    scales = torch.ones(r_max + 1, dtype=torch.float32, device=x.device)
    if first.scale_per_rank:
        for r, s in first.scale_per_rank.items():
            if r <= r_max:
                scales[r] = s

    T = token_ranks.numel()
    v_buf = torch.zeros(T, r_max, dtype=torch.float32, device=x.device)
    return {
        "a_pool": a_pool,
        "b_pool": b_pool,
        "scales": scales,
        "v_buf": v_buf,
    }


def b4_ma_shared(
    x: torch.Tensor,
    adapters: Mapping[int, MatryoshkaLoRA] | Sequence[MatryoshkaLoRA] | tuple[torch.Tensor, torch.Tensor],
    adapter_ids: torch.Tensor,
    token_ranks: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    add: bool = False,
    prepared: dict | None = None,
) -> torch.Tensor:
    """S-LoRA baseline with one shared copy per adapter."""
    if prepared is None:
        prepared = prepare_b4_ma_shared(x, adapters, token_ranks)

    return bgmv_forward(
        x, prepared["a_pool"], prepared["b_pool"], adapter_ids, token_ranks,
        scales=prepared["scales"], out=out, add=add, v_buf=prepared["v_buf"]
    )
