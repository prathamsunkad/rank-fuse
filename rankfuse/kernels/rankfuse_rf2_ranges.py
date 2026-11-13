"""RF2, per-bucket ranges (`rf2_ranges`): one RF1 launch per rank bucket over sorted rows.

On a batch whose requests are physically sorted by rank, every rank bucket (16/32/64/128/256 rows;
rank 0 = no adapter) is one contiguous run of rows. A RangePlan, built on the host once per step,
finds those runs: (bucket, start, end). Each layer then runs rf1_fused's split-K pass on the row
views x[start:end], out[start:end] at exactly that bucket's width, with that width's launch
settings (rankfuse_fused._config). Views only move the pointers, so this needs no kernel changes,
no index list and no run-time width branch: each bucket is a plain RF1 call on its own rows. The
runs go on concurrent CUDA streams, heaviest first (the fork/join of rf2_bucketed, capturable in a
graph); `rf2_ranges_seq` runs them one after another to measure what the streams add.

Unsorted batches have a bucket in more than one run. The plan then falls back to one run over all
rows at the batch's width, which is exactly rf1_fused: correct in any order, but nothing skipped.
"""

import torch
import triton

from rankfuse.interface import MatryoshkaLoRA, register
from rankfuse.kernels import rankfuse_fused as rf
from rankfuse.kernels._common import check_layout
from rankfuse.kernels.rankfuse_rf2_bucketed import _stream


class RangePlan:
    """Per-step plan from the token ranks: runs of rows sharing a rank bucket, heaviest first.

    runs: (R, start, end, run_max_rank) for R > 0; zero_runs: (start, end) of rank-0 rows.
    sorted is False when some bucket appears in more than one run; then runs is a single run over
    every row at the batch's bucket (rf1_fused) and zero_runs is empty."""

    def __init__(self, token_ranks):
        r = torch.as_tensor(token_ranks).to("cpu", torch.int64).flatten()
        self.T = r.numel()
        self.r_limit = int(r.max()) if self.T else 0
        self.runs, self.zero_runs, self.sorted = [], [], True
        if not self.T:
            return
        pow2 = torch.ceil(torch.log2(r.clamp(min=1).double())).long()
        buckets = torch.where(r > 0, torch.clamp(1 << pow2, min=16), 0)
        change = torch.nonzero(buckets[1:] != buckets[:-1]).flatten() + 1
        starts = [0, *change.tolist()]
        ends = [*change.tolist(), self.T]
        run_buckets = [int(buckets[s]) for s in starts]
        if len(set(run_buckets)) != len(run_buckets):  # not sorted by bucket: plain RF1
            self.sorted = False
            if self.r_limit:
                self.runs = [(max(16, triton.next_power_of_2(self.r_limit)), 0, self.T, self.r_limit)]
            return
        for R, s, e in zip(run_buckets, starts, ends):
            if R == 0:
                self.zero_runs.append((s, e))
            else:
                self.runs.append((R, s, e, int(r[s:e].max())))
        self.runs.sort(key=lambda run: (run[2] - run[1]) * run[0], reverse=True)  # heaviest first


def rf2_ranges(x, A_t, B, token_ranks, plan: RangePlan, out=None, add=False, scale_table=None, concurrent=True,
               slice_sizes=None):
    """Per-bucket RF2 for one adapter or several output slices: out (+)= delta, following a RangePlan."""
    if A_t.dim() == 2:
        A_t, B = A_t.unsqueeze(0), B.unsqueeze(0)
    T = x.shape[0]
    n_max = B.shape[2]
    sizes = tuple(int(s) for s in slice_sizes) if slice_sizes is not None else (n_max,) * A_t.shape[0]
    if out is None:
        if add:
            raise ValueError("add=True needs an existing out to add into")
        out = torch.empty((T, sum(sizes)), dtype=x.dtype, device=x.device)
    if not add:
        for s, e in plan.zero_runs:
            out[s:e].zero_()

    def launch(R, s, e, run_max):
        rf._splitk(x[s:e], A_t, B, token_ranks[s:e], out[s:e], add, sizes, run_max, R, scale_table, None,
                   False, 0, 0)

    if concurrent and len(plan.runs) > 1:
        main = torch.cuda.current_stream()
        streams = []
        for run in plan.runs:
            st = _stream(x.device, run[0])
            st.wait_stream(main)
            with torch.cuda.stream(st):
                launch(*run)
            streams.append(st)
        for st in streams:
            main.wait_stream(st)
    else:
        for run in plan.runs:
            launch(*run)
    return out


def _prepare_plan(x: torch.Tensor, lora: MatryoshkaLoRA, token_ranks: torch.Tensor) -> dict:
    return dict(plan=RangePlan(token_ranks))


@register("rf2_ranges", requires_cuda=True, prepare=_prepare_plan)
def rankfuse_rf2_ranges(
    x: torch.Tensor,
    lora: MatryoshkaLoRA,
    token_ranks: torch.Tensor,
    plan: RangePlan | None = None,
    out: torch.Tensor | None = None,
    add: bool = False,
    concurrent: bool = True,
) -> torch.Tensor:
    """Per-bucket RF2 over sorted rows. A serving engine builds the RangePlan once per step from
    the request ranks it already knows; by default it is built here (one host sync)."""
    check_layout(x, lora, token_ranks)
    if plan is None:
        plan = RangePlan(token_ranks)
    return rf2_ranges(x, lora.A_t, lora.B, token_ranks, plan, out=out, add=add,
                      scale_table=rf._scale_table(lora, x.device), concurrent=concurrent)


@register("rf2_ranges_seq", requires_cuda=True, prepare=_prepare_plan)
def rankfuse_rf2_ranges_seq(
    x: torch.Tensor,
    lora: MatryoshkaLoRA,
    token_ranks: torch.Tensor,
    plan: RangePlan | None = None,
    out: torch.Tensor | None = None,
    add: bool = False,
) -> torch.Tensor:
    """rf2_ranges with runs one after another: measures what concurrent streams change."""
    return rankfuse_rf2_ranges(x, lora, token_ranks, plan=plan, out=out, add=add, concurrent=False)
