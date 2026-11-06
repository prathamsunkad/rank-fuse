"""RF2, bucketed (`rf2_bucketed`): one launch of rf1_fused's kernels per rank bucket.

Every token goes to the bucket of its own rank (16/32/64/128/256 rank rows; rank 0 = no adapter).
A Plan built on the CPU once per step lists the tokens of each bucket in their original order (a
prompt's tokens stay contiguous). For each bucket present, the split-K kernels (rankfuse_fused.py,
section 8) are launched compiled for exactly R rows, reading tokens through the bucket's index. Each
bucket runs on its own CUDA stream, forked from and joined back to the current stream (capturable
in a CUDA graph), heaviest bucket first, so light buckets fill the SMs alongside it.

Block size, tile and shared-memory rules are rf1_fused's, so rf2_bucketed differs from it only in
rank width, index reads and launches. `rf2_bucketed_tuned` adds what bucketing makes possible:
larger token blocks for low buckets (128 tokens for buckets 16 and 32 above 128 tokens) with their
tiles and pipeline sized for each bucket. Report it as extra tuning, not as skipping.

use_rf2 below is the CPU-side rule that picks rf2_bucketed over rf1_fused; it was fitted on the
earlier tuned kernel and one H200 sweep.
"""

import torch
import triton

from rankfuse.interface import MatryoshkaLoRA, register
from rankfuse.kernels import rankfuse_fused as rf
from rankfuse.kernels._common import check_layout


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
    """Per-step plan from the token ranks (a list or a tensor): token indices per bucket,
    heaviest first. Built with a few vectorised CPU ops (about 0.1 ms for thousands of tokens)."""

    def __init__(self, token_ranks, device="cuda"):
        r = torch.as_tensor(token_ranks).to("cpu", torch.int64).flatten()
        self.T = r.numel()
        self.r_limit = int(r.max()) if self.T else 0
        # bucket_of for every token: 0, or the smallest of 16, 32, 64, ... >= r
        pow2 = torch.ceil(torch.log2(r.clamp(min=1).double())).long()
        buckets = torch.where(r > 0, torch.clamp(1 << pow2, min=16), 0)
        self.zero_idx = torch.nonzero(buckets == 0).flatten().to(device)
        self.buckets = []  # (R, idx tensor, count, BM)
        for R in sorted(torch.unique(buckets[buckets > 0]).tolist(), reverse=True):
            idx = torch.nonzero(buckets == R).flatten().to(torch.int32)
            self.buckets.append((R, idx.to(device), idx.numel(), block_m(R, self.T)))
        full = bucket_of(self.r_limit)
        # bucketed rank work relative to rf1_fused (every token at the batch maximum's bucket)
        self.work_ratio = float(buckets.sum()) / (self.T * full) if full else 1.0


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


def rf2_bucketed(x, A_t, B, token_ranks, plan: Plan, out=None, add=False, scale_table=None, concurrent=True, tuned=False, slice_sizes=None):
    """Bucketed RF2 for one adapter or several output slices: out (+)= delta, following a per-step Plan."""
    if A_t.dim() == 2:
        A_t, B = A_t.unsqueeze(0), B.unsqueeze(0)
    T, h_in = x.shape
    n_slices = A_t.shape[0]
    sizes = tuple(int(s) for s in slice_sizes) if slice_sizes is not None else (B.shape[2],) * n_slices
    if out is None:
        if add:
            raise ValueError("add=True needs an existing out to add into")
        out = torch.empty((T, sum(sizes)), dtype=x.dtype, device=x.device)
    if plan.zero_idx.numel() and not add:
        out.index_fill_(0, plan.zero_idx, 0)
    if not plan.buckets:
        return out
    r_limit = plan.r_limit
    r_pad = max(16, triton.next_power_of_2(r_limit))
    dev = x.device.index or 0
    tiled = T > rf.EXPAND_TILED_MIN_TOKENS
    # (R, idx, count, BM, BH, stages, cfg): rf1_fused's settings at the batch width, or each
    # bucket's own when tuned (rf._config at R; the hand table in block_m when nothing is tuned).
    groups = []
    for R, idx, count, tuned_bm in plan.buckets:
        if tuned:
            cfg = rf._config(R, T, dev, bm=tuned_bm, bn=256 if tuned_bm <= 64 else 128)
        else:
            cfg = rf._config(r_pad, T, dev)
        bm = cfg["bm"]
        bh, stages = rf._splitk_shrink_tiles(bm, R if tuned else r_pad, x.element_size(), dev, cfg["bh"], cfg["stages"])
        groups.append((R, idx, count, bm, bh, stages, cfg))
    grid_tiles = sum(triton.cdiv(g[2], g[3]) for g in groups)
    bh_min = min(g[4] for g in groups)
    split_k, k_per = rf._split_count(h_in, tiled, grid_tiles, n_slices, rf._num_sms(dev), bh_min)  # k_per: a multiple of bh_min
    ws = torch.empty((n_slices, split_k, T, r_pad), dtype=torch.float32, device=x.device)

    def launch(R, idx, count, bm, bh, stages, cfg, ws_row0):
        rf.splitk_pass(x, A_t, B, token_ranks, out, add, sizes, scale_table, ws, r_limit, r_pad, split_k, k_per, bm,
                       bh if k_per % bh == 0 else bh_min, stages, index=idx, count=count, ws_row0=ws_row0, grid_tiles=grid_tiles,
                       rank=R, block_n=cfg["bn"], shrink_warps=cfg["shrink_warps"], expand_warps=cfg["expand_warps"])

    ws_row0 = 0  # each bucket takes its own workspace rows, in its own order
    if concurrent and len(groups) > 1:
        main = torch.cuda.current_stream()
        streams = []
        for g in groups:  # heaviest first
            s = _stream(x.device, g[0])
            s.wait_stream(main)
            with torch.cuda.stream(s):
                launch(*g, ws_row0)
            ws_row0 += g[2]
            streams.append(s)
        for s in streams:
            main.wait_stream(s)
    else:
        for g in groups:
            launch(*g, ws_row0)
            ws_row0 += g[2]
    return out


def _prepare_plan(x: torch.Tensor, lora: MatryoshkaLoRA, token_ranks: torch.Tensor) -> dict:
    return dict(plan=Plan(token_ranks, x.device))


@register("rf2_bucketed", requires_cuda=True, prepare=_prepare_plan)
def rankfuse_rf2_bucketed(
    x: torch.Tensor,
    lora: MatryoshkaLoRA,
    token_ranks: torch.Tensor,
    plan: Plan | None = None,
    out: torch.Tensor | None = None,
    add: bool = False,
    concurrent: bool = True,
) -> torch.Tensor:
    """Bucketed RF2 on rf1_fused's block and tile rules. A serving engine builds the Plan once per
    step from the request ranks it already knows; by default it is built here (one host sync)."""
    check_layout(x, lora, token_ranks)
    if plan is None:
        plan = Plan(token_ranks, x.device)
    return rf2_bucketed(x, lora.A_t, lora.B, token_ranks, plan, out=out, add=add,
                        scale_table=rf._scale_table(lora, x.device), concurrent=concurrent)


@register("rf2_bucketed_tuned", requires_cuda=True, prepare=_prepare_plan)
def rankfuse_rf2_bucketed_tuned(
    x: torch.Tensor,
    lora: MatryoshkaLoRA,
    token_ranks: torch.Tensor,
    plan: Plan | None = None,
    out: torch.Tensor | None = None,
    add: bool = False,
    concurrent: bool = True,
) -> torch.Tensor:
    """Bucketed RF2 with per-bucket block sizes, tiles and pipeline depth."""
    check_layout(x, lora, token_ranks)
    if plan is None:
        plan = Plan(token_ranks, x.device)
    return rf2_bucketed(x, lora.A_t, lora.B, token_ranks, plan, out=out, add=add,
                        scale_table=rf._scale_table(lora, x.device), concurrent=concurrent, tuned=True)


@register("rf2_bucketed_seq", requires_cuda=True, prepare=_prepare_plan)
def rankfuse_rf2_bucketed_seq(
    x: torch.Tensor,
    lora: MatryoshkaLoRA,
    token_ranks: torch.Tensor,
    plan: Plan | None = None,
    out: torch.Tensor | None = None,
    add: bool = False,
) -> torch.Tensor:
    """rf2_bucketed with buckets one after another: measures what concurrent streams change."""
    return rankfuse_rf2_bucketed(x, lora, token_ranks, plan=plan, out=out, add=add, concurrent=False)
