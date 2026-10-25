"""RF2, bucketed (`rf2_bucketed`): one specialised kernel pair per rank bucket.

Every token goes to the bucket of its own rank (16/32/64/128/256 rank rows; rank 0 = no
adapter). A Plan built on the CPU once per step lists the tokens of each bucket in their
original order (a prompt's tokens stay contiguous). For each bucket present, one split-K shrink
and one expand kernel are launched, each compiled for exactly R rows (no runtime branch, fewer
registers for low ranks); low-rank buckets use larger token blocks (128 tokens for buckets 16
and 32 above 128 tokens). Each bucket's kernels run on their own CUDA stream, forked from and
joined back to the current stream (capturable in a CUDA graph), heaviest bucket first, so light
buckets fill the SMs alongside it. Run one after another they are 1.1-2.2x slower than rf1_fused.

Slice count, tile sizes and shared-memory guards follow rf1_fused (rankfuse_fused.py, section 8).

Measured on an H200, GPU time by CUDA-graph replay (4096 x 4096, r_max 64): 1.10-1.15x faster
than rf1_fused on prompt batches mixing low ranks with a rank-64 request. Slower when the batch
maximum rank is already low, for small decode batches, on small models (whole Qwen2.5-0.5B:
0.95-1.01x of rf1_fused) and in eager mode (stream fork/join and per-bucket launches from
Python, up to 2x slower). use_rf2 below is the CPU-side rule that picks it only where it paid.
"""

import torch
import triton
import triton.language as tl

from rankfuse.interface import MatryoshkaLoRA, register
from rankfuse.kernels import rankfuse_fused as rf
from rankfuse.kernels._common import check_layout


@triton.jit
def _bk_shrink_kernel(x_ptr, a_ptr, idx_ptr, count, ws_ptr, T, H_IN, r_limit, K_PER,
                      BM: tl.constexpr, BH: tl.constexpr, R: tl.constexpr, R_PAD: tl.constexpr):
    pos = tl.program_id(0) * BM + tl.arange(0, BM)
    row_ok = pos < count
    tok = tl.load(idx_ptr + pos, mask=row_ok, other=0)
    k = tl.arange(0, R)
    k_ok = k < r_limit
    h_lo = tl.program_id(1) * K_PER
    v = tl.zeros((BM, R), dtype=tl.float32)
    for h_start in range(0, K_PER, BH):
        h = h_lo + h_start + tl.arange(0, BH)
        h_ok = h < H_IN
        x = tl.load(x_ptr + tok[:, None] * H_IN + h[None, :], mask=row_ok[:, None] & h_ok[None, :], other=0.0)
        a = tl.load(a_ptr + k[:, None] * H_IN + h[None, :], mask=k_ok[:, None] & h_ok[None, :], other=0.0)
        v += tl.dot(x, tl.trans(a), input_precision="ieee")
    ws = ws_ptr + tl.program_id(1) * T * R_PAD + tok[:, None] * R_PAD + k[None, :]
    tl.store(ws, v, mask=row_ok[:, None])


@triton.jit
def _bk_expand_kernel(ws_ptr, ranks_ptr, idx_ptr, count, scale_ptr, b_ptr, out_ptr, T, N, OUT_STRIDE, r_limit,
                      SPLIT_K, cols_per_prog, BM: tl.constexpr, BN: tl.constexpr, R: tl.constexpr,
                      R_PAD: tl.constexpr, HAS_SCALE: tl.constexpr, ADD_OUT: tl.constexpr, LOOP: tl.constexpr):
    pos = tl.program_id(0) * BM + tl.arange(0, BM)
    row_ok = pos < count
    tok = tl.load(idx_ptr + pos, mask=row_ok, other=0)
    ranks = tl.load(ranks_ptr + tok, mask=row_ok, other=0)
    k = tl.arange(0, R)
    k_ok = k < r_limit
    v = tl.zeros((BM, R), dtype=tl.float32)
    for s in range(0, SPLIT_K):
        v += tl.load(ws_ptr + s * T * R_PAD + tok[:, None] * R_PAD + k[None, :], mask=row_ok[:, None], other=0.0)
    v = tl.where(k[None, :] < ranks[:, None], v, 0.0)
    if HAS_SCALE:
        v = v * tl.load(scale_ptr + ranks, mask=row_ok, other=0.0)[:, None]
    v_in = v.to(out_ptr.dtype.element_ty)
    col_base = tl.program_id(1) * cols_per_prog
    if LOOP:
        for col_start in range(0, cols_per_prog, BN):
            cols = col_base + col_start + tl.arange(0, BN)
            col_ok = cols < N
            b = tl.load(b_ptr + k[:, None] * N + cols[None, :], mask=k_ok[:, None] & col_ok[None, :], other=0.0)
            acc = tl.dot(v_in, b, input_precision="ieee")
            ptrs = out_ptr + tok[:, None] * OUT_STRIDE + cols[None, :]
            mask = row_ok[:, None] & col_ok[None, :]
            if ADD_OUT:
                acc += tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
            tl.store(ptrs, acc, mask=mask)
    else:
        cols = col_base + tl.arange(0, BN)
        col_ok = cols < N
        b = tl.load(b_ptr + k[:, None] * N + cols[None, :], mask=k_ok[:, None] & col_ok[None, :], other=0.0)
        acc = tl.dot(v_in, b, input_precision="ieee")
        ptrs = out_ptr + tok[:, None] * OUT_STRIDE + cols[None, :]
        mask = row_ok[:, None] & col_ok[None, :]
        if ADD_OUT:
            acc += tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        tl.store(ptrs, acc, mask=mask)


def bucket_of(r: int) -> int:
    """Rank bucket of a token: 0 for no adapter, else the smallest of 16/32/64/128/256 >= r."""
    if r <= 0:
        return 0
    return max(16, triton.next_power_of_2(r))


def block_m(bucket: int, num_tokens: int) -> int:
    if num_tokens <= rf.EXPAND_TILED_MIN_TOKENS:
        return 16
    return {16: 128, 32: 128, 64: 64, 128: 32, 256: 16}[bucket]


class Plan:
    """Per-step plan from the token ranks (a CPU list): token indices per bucket, heaviest first."""

    def __init__(self, token_ranks_cpu: list[int], device="cuda"):
        self.T = len(token_ranks_cpu)
        self.r_limit = max(token_ranks_cpu) if token_ranks_cpu else 0
        groups: dict[int, list[int]] = {}
        for t, r in enumerate(token_ranks_cpu):
            groups.setdefault(bucket_of(r), []).append(t)
        self.zero_idx = torch.tensor(groups.pop(0, []), dtype=torch.int64, device=device)
        self.buckets = []  # (R, idx tensor, count, BM)
        for R in sorted(groups, reverse=True):
            idx = torch.tensor(groups[R], dtype=torch.int32, device=device)
            self.buckets.append((R, idx, len(groups[R]), block_m(R, self.T)))
        full = bucket_of(self.r_limit)
        # bucketed rank work relative to rf1_fused (every token at the batch maximum's bucket)
        self.work_ratio = (sum(bucket_of(r) for r in token_ranks_cpu) / (self.T * full)) if full else 1.0


def use_rf2(plan: Plan, tokens_per_request: int, graphed: bool = True) -> bool:
    """CPU-side choice between rf2_bucketed and rf1_fused. Picks rf2_bucketed only under CUDA
    graphs, for batches above 128 tokens whose bucketed rank work is <= 45% of rf1_fused's and
    that have >= 4096 tokens or >= 64 tokens per request (prompt batches). Fitted on one H200
    sweep of 4096 x 4096 layers; not yet validated on other shapes."""
    return (graphed and plan.T > rf.EXPAND_TILED_MIN_TOKENS and plan.work_ratio <= 0.45
            and (plan.T >= 4096 or tokens_per_request >= 64))


_STREAMS: dict[tuple, torch.cuda.Stream] = {}


def _stream(device, R):
    key = (device, R)
    if key not in _STREAMS:
        _STREAMS[key] = torch.cuda.Stream(device=device)
    return _STREAMS[key]


def rf2_bucketed(x, A_t, B, token_ranks, plan: Plan, out=None, add=False, scale_table=None, concurrent=True):
    """Bucketed RF2 for one adapter: out (+)= delta, following a per-step Plan."""
    T, h_in = x.shape
    n = B.shape[1]
    if out is None:
        if add:
            raise ValueError("add=True needs an existing out to add into")
        out = torch.empty((T, n), dtype=x.dtype, device=x.device)
    if plan.zero_idx.numel() and not add:
        out.index_fill_(0, plan.zero_idx, 0)
    if not plan.buckets:
        return out
    r_limit = plan.r_limit
    r_pad = max(16, triton.next_power_of_2(r_limit))
    dev = x.device.index or 0
    num_sms = rf._num_sms(dev)
    tiled = T > rf.EXPAND_TILED_MIN_TOKENS
    total_blocks = sum(triton.cdiv(c, bm) for _, _, c, bm in plan.buckets)
    bh_min = min(rf._splitk_shrink_tiles(bm, R, x.element_size(), dev)[0] for R, _, _, bm in plan.buckets)
    split_k = max(rf.MIN_SPLIT_K_TILED if tiled else 1, min(rf.MAX_SPLIT_K, rf.SPLITK_WAVES * num_sms // total_blocks))
    k_per = triton.cdiv(triton.cdiv(h_in, split_k), bh_min) * bh_min  # a multiple of every bucket's BH used
    split_k = triton.cdiv(h_in, k_per)
    ws = torch.empty((split_k, T, r_pad), dtype=torch.float32, device=x.device)
    scales = out if scale_table is None else scale_table

    def launch(R, idx, count, bm):
        nb = triton.cdiv(count, bm)
        bh, stages = rf._splitk_shrink_tiles(bm, R, x.element_size(), dev)
        if k_per % bh:
            bh = bh_min
        _bk_shrink_kernel[(nb, split_k)](x, A_t, idx, count, ws, T, h_in, r_limit, k_per,
                                         BM=bm, BH=bh, R=R, R_PAD=r_pad, num_warps=4, num_stages=stages)
        if tiled:
            bn = 256 if bm <= 64 else 128
            while bn > 64 and R * bn * x.element_size() > 0.35 * rf._max_shared_mem(dev):
                bn //= 2
            cols, warps = bn, rf.EXPAND_TILED_WARPS
        else:
            bn, warps = rf.EXPAND_BLOCK_N, rf.EXPAND_WARPS
            n_tiles = triton.cdiv(n, bn)
            n_split = min(n_tiles, max(1, rf.EXPAND_WAVES * num_sms // total_blocks))
            cols = triton.cdiv(n_tiles, n_split) * bn
        _bk_expand_kernel[(nb, triton.cdiv(n, cols))](
            ws, token_ranks, idx, count, scales, B, out, T, n, out.stride(0), r_limit, split_k, cols,
            BM=bm, BN=bn, R=R, R_PAD=r_pad, HAS_SCALE=scale_table is not None, ADD_OUT=add, LOOP=not tiled,
            num_warps=warps, num_stages=2)

    if concurrent and len(plan.buckets) > 1:
        main = torch.cuda.current_stream()
        streams = []
        for R, idx, count, bm in plan.buckets:  # heaviest first
            s = _stream(x.device, R)
            s.wait_stream(main)
            with torch.cuda.stream(s):
                launch(R, idx, count, bm)
            streams.append(s)
        for s in streams:
            main.wait_stream(s)
    else:
        for R, idx, count, bm in plan.buckets:
            launch(R, idx, count, bm)
    return out


@register("rf2_bucketed", requires_cuda=True)
def rankfuse_rf2_bucketed(
    x: torch.Tensor,
    lora: MatryoshkaLoRA,
    token_ranks: torch.Tensor,
    plan: Plan | None = None,
    out: torch.Tensor | None = None,
    add: bool = False,
    concurrent: bool = True,
) -> torch.Tensor:
    """Bucketed RF2. A serving engine builds the Plan once per step from the request ranks it
    already knows; by default it is built here from token_ranks (one host sync)."""
    check_layout(x, lora, token_ranks)
    if plan is None:
        plan = Plan(token_ranks.tolist(), x.device)
    return rf2_bucketed(x, lora.A_t, lora.B, token_ranks, plan, out=out, add=add,
                        scale_table=rf._scale_table(lora, x.device), concurrent=concurrent)
