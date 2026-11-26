"""B4-multi: many Matryoshka adapters with S-LoRA's mixed-rank decode kernel (bgmv), two ways.

The kernel, its contract and its limits (fp16/bf16, compiled hidden sizes, r_max <= 64 and a
multiple of 8) are those of Stage 1's B4 (rankfuse/baselines/slora_bgmv.py, whose extension,
Pool and apply are reused here). Each token points at one entry (start row, rank, scale) of
S-LoRA's unified pool; the kernel does one CUDA-core dot product per (token, rank row).

    b4_multi_copies  how S-LoRA would serve it today: every (adapter, rank) pair a request asks
                     for is its own adapter at its true rank (rounded up to a multiple of 8 for
                     the expand), with its own A rows and B^T rows in the pool. Only the pairs
                     present in the step are stored.
    b4_multi_shared  one shared copy per adapter (Stage 1's "shared" trick): each adapter's A_t
                     and B^T [h_out, r_max] stored once; entry (a, r) points at adapter a's rows,
                     shrinks at rank r and expands at r_max (the extra v columns are zero).

S-LoRA runs bgmv only for decode and switches to a separate Triton kernel for prompts (>= 200
tokens); prefill rows for B4 measure the bgmv kernel itself, not how S-LoRA serves prompts.

Registered only on request (`enable()`; rankfuse.multi.load_all does it), like Stage 1's B4,
because the first use JIT-builds the CUDA extension.
"""

from __future__ import annotations

import torch

from rankfuse.baselines import slora_bgmv as sb
from rankfuse.multi import MULTI_REGISTRY, PagedWeights, output_buffer, register_multi

NAMES = ("b4_multi_copies", "b4_multi_shared")
# RankFuse A100 r_max 128/256 (2026-11-23): the multi-adapter B4 also takes delta widths 128 and
# 256, compiled into slora_ext only with hidden sizes 512 and up (bgmv_config.h); the width must be
# below both hidden sizes (bgmv_kernel picks expand vs shrink by feat_in < feat_out). Stage 1's B4
# keeps sb.NARROW_SIZES (r_max <= 64).
NARROW_SIZES = sb.NARROW_SIZES + (128, 256)


def check(W: PagedWeights) -> str | None:
    if W.n_slices != 1:
        return "S-LoRA bgmv: one slice per call"
    for side, h in (("h_in", W.h_in), ("h_out", W.h_out)):
        if h not in sb.WIDE_SIZES:
            return f"S-LoRA bgmv: {side}={h} is not compiled in (WIDE_SIZES={sb.WIDE_SIZES})"
    if W.r_max > NARROW_SIZES[-1] or W.r_max % 8:
        return f"S-LoRA bgmv: r_max={W.r_max} must be a multiple of 8 up to {NARROW_SIZES[-1]}"
    width = next(n for n in NARROW_SIZES if n >= W.r_max)
    if width > sb.NARROW_SIZES[-1] and width >= min(W.h_in, W.h_out):
        return f"S-LoRA bgmv: delta width {width} must be below h_in and h_out (got {W.h_in} -> {W.h_out})"
    return None


def _scales(W: PagedWeights, pa: torch.Tensor, pr: torch.Tensor) -> torch.Tensor:
    if W.scale_table is None:
        return (pr > 0).float()
    return W.scale_table[pa.clamp(min=0), pr] * (pr > 0).float()


def pool_copies(W: PagedWeights, pairs: list[tuple[int, int]]):
    """Entries 0..P-1 = the pairs at their true rank (stored at a multiple of 8); entry P = rank 0."""
    dev, dt = W.device, W.dtype
    width = next(n for n in NARROW_SIZES if n >= W.r_max)
    stored = [-(-r // 8) * 8 for _, r in pairs]
    rows = sum(stored)
    A = torch.zeros(rows + 1, 1, W.h_in, dtype=dt, device=dev)
    B = torch.zeros(rows + 1, 1, W.h_out, dtype=dt, device=dev)
    start = [0] * len(pairs)
    row = 0
    for i, s in enumerate(stored):
        start[i] = row
        row += s
    # Pairs with the same stored width are copied in one batched op.
    for s_w in sorted(set(stored)):
        sel = [i for i, s in enumerate(stored) if s == s_w]
        pa = torch.tensor([pairs[i][0] for i in sel], device=dev)
        pr = torch.tensor([pairs[i][1] for i in sel], device=dev)
        mask = (torch.arange(s_w, device=dev) < pr[:, None]).to(dt)[:, :, None]  # [P_s, s_w, 1]
        a_rows = W.A_dense[pa, 0, :s_w] * mask  # [P_s, s_w, h_in]
        bt = (W.B_dense[pa, 0, :s_w, : W.h_out] * mask).transpose(1, 2).contiguous()  # B^T [P_s, h_out, s_w]
        dst = torch.cat([torch.arange(start[i], start[i] + s_w) for i in sel]).to(dev)
        A[dst, 0] = a_rows.reshape(-1, W.h_in)
        B[dst, 0] = bt.reshape(-1, W.h_out)  # B^T [h_out, s] flattened into s rows of h_out
    pa = torch.tensor([a for a, _ in pairs] + [0], device=dev)
    pr = torch.tensor([r for _, r in pairs] + [0], device=dev)
    return sb.Pool(A, B, start + [rows], stored + [0], stored + [0], _scales(W, pa, pr).tolist(), [0], width, dt, dev)


def pool_shared(W: PagedWeights):
    """Adapter a's A_t and B^T stored once at rows a*r_max..; entry a*(r_max+1) + r = (a, r)."""
    dev, dt, N, R = W.device, W.dtype, W.num_adapters, W.r_max
    width = next(n for n in NARROW_SIZES if n >= R)
    A = W.A_dense[:, 0].reshape(N * R, 1, W.h_in).contiguous()
    B = W.B_dense[:, 0, :, : W.h_out].transpose(1, 2).contiguous().reshape(N * R, 1, W.h_out)
    pa = torch.arange(N, device=dev).repeat_interleave(R + 1)
    pr = torch.arange(R + 1, device=dev).repeat(N)
    start = (pa * R).tolist()
    shrink = pr.tolist()
    expand = torch.where(pr > 0, R, 0).tolist()
    return sb.Pool(A, B, start, shrink, expand, _scales(W, pa, pr).tolist(), [0], width, dt, dev)


def _prepare(mode: str):
    def prepare(x, W: PagedWeights, token_adapter, token_ranks) -> dict:
        reason = check(W)
        if reason:
            raise ValueError(reason)
        ad = token_adapter.long().cpu()
        rk = token_ranks.long().cpu()
        live = (rk > 0) & (ad >= 0)
        if mode == "shared":
            key = ("b4_shared_pool", W.A_dense.data_ptr(), W.B_dense.data_ptr())
            if key not in W._cache:
                W._cache[key] = pool_shared(W)
            pool = W._cache[key]
            entries = torch.where(live, ad * (W.r_max + 1) + rk, torch.zeros_like(ad))
        else:
            if live.any():
                uniq, inv = torch.unique(torch.stack([ad[live], rk[live]], 1), dim=0, return_inverse=True)
                pairs = [tuple(p) for p in uniq.tolist()]
            else:
                pairs, inv = [], torch.zeros(0, dtype=torch.long)
            pool = pool_copies(W, pairs)
            entries = torch.full_like(ad, len(pairs))
            entries[live] = inv
        sb._ext()  # build or load the extension here, not inside a timed or captured call
        return dict(pool=pool, entries=entries.to(x.device),
                    v=torch.zeros(x.shape[0], pool.width, dtype=x.dtype, device=x.device))

    return prepare


def _impl(mode: str):
    prepare = _prepare(mode)

    def b4(x, W: PagedWeights, token_adapter, token_ranks, pool=None, entries=None, v=None, out=None, add=False):
        if x.shape[0] > sb.MAX_TOKENS:
            raise ValueError(f"{x.shape[0]} tokens: the bgmv grid holds at most {sb.MAX_TOKENS}")
        if pool is None:
            meta = prepare(x, W, token_adapter, token_ranks)
            pool, entries, v = meta["pool"], meta["entries"], meta["v"]
        out = output_buffer(x, W, out, add)
        if not add:
            out.zero_()
        if x.shape[0] == 0:
            return out
        return sb.apply(x.contiguous(), pool, entries, v, out)

    return b4, prepare


def enable() -> None:
    """Register b4_multi_copies and b4_multi_shared (idempotent)."""
    for name in NAMES:
        if name not in MULTI_REGISTRY:
            fn, prepare = _impl(name.removeprefix("b4_multi_"))
            register_multi(name, prepare=prepare, dtypes=(torch.float16, torch.bfloat16), supports_slices=False,
                           check=check)(fn)


def disable() -> None:
    for name in NAMES:
        MULTI_REGISTRY.pop(name, None)
