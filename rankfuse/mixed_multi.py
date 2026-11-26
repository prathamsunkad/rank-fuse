"""Mixed (chunked-prefill) serving steps for the multi-adapter path.

A separate module: nothing in rankfuse.workloads or rankfuse.multi changes. A step has the same
shape as Stage 1's mixed steps (rankfuse.workloads, `lengths`): `num_decodes` one-token decode
requests, then prompts or prompt chunks that fill (chunked_<budget>_<chunk>) or partly fill
(serving_<k>_<budget>) the step's token budget, in a seeded random request order. The four step
types of bench/configs/rf2_mixed.yaml:

    1. decode only            lengths None
    2. one prompt fills       chunked_<budget>_<budget>
    3. short prompts          serving_<k>_<budget>
    4. many partial prefills  chunked_<budget>_<chunk>

The request lengths and order come from rankfuse.workloads.Workload with the same num_requests,
lengths and seed, so a multi-adapter step has exactly the shape of the single-adapter step with
the same parameters. Ranks also come from there (sample_request_ranks: decodes from `dist`,
prompts from `prefill_dist` when set). Each request is also given an adapter, drawn from a Zipf
popularity over `num_adapters` ids (weight 1 / (i + 1)^zipf_alpha; adapter 0 is the most popular),
independently of its rank and of whether it is a prompt. A request's tokens are consecutive and
share its adapter and rank. Synthetic, with no trace behind it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from rankfuse.multi import MultiWorkload
from rankfuse.workloads import DTYPES, Workload, sample_request_ranks


def step_type(lengths: str | None) -> str:
    """Which of the four step types a lengths preset is."""
    if lengths is None:
        return "decode_only"
    if lengths.startswith("serving_"):
        return "short_prompts"
    budget, chunk = lengths.removeprefix("chunked_").split("_")
    return "one_prompt_fills" if budget == chunk else "partial_prefills"


@dataclass(frozen=True)
class MixedMultiWorkload:
    num_decodes: int
    lengths: str | None  # None: decode only; else a rankfuse.workloads lengths preset
    num_adapters: int
    zipf_alpha: float = 1.0
    dist: str = "zipf_1"  # rank mix of decodes (of every request without prefill_dist)
    prefill_dist: str | None = None  # rank mix of prompts
    h_in: int = 4096
    h_out: int = 4096
    r_max: int = 64
    dtype: str = "fp16"
    seed: int = 0
    pages: str = "shuffled"

    def shape_workload(self) -> Workload:
        """The single-adapter workload with this step's shape and rank mix."""
        return Workload(num_requests=self.num_decodes, tokens_per_request=1, h_in=self.h_in, h_out=self.h_out,
                        r_max=self.r_max, dist=self.dist, dtype=self.dtype, seed=self.seed, lengths=self.lengths,
                        prefill_dist=self.prefill_dist if self.lengths is not None else None)

    def weights_workload(self) -> MultiWorkload:
        """For rankfuse.multi.make_weights: the adapters depend on seed, count, shape, r_max, dtype, pages."""
        return MultiWorkload(1, 1, self.num_adapters, popularity="zipf", zipf_alpha=self.zipf_alpha, dist="uniform",
                             h_in=self.h_in, h_out=self.h_out, r_max=self.r_max, dtype=self.dtype, seed=self.seed,
                             pages=self.pages)

    @property
    def step_type(self) -> str:
        return step_type(self.lengths)

    def as_row(self) -> dict:
        row = asdict(self)
        sw = self.shape_workload()
        lens, pre = sw.request_lengths, sw.request_is_prefill
        dec = sum(n for n, p in zip(lens, pre) if not p)
        row.update(step_type=self.step_type, num_tokens=sum(lens), decode_tokens=dec, prefill_tokens=sum(lens) - dec,
                   num_prompts=sum(pre), decode_share=dec / max(sum(lens), 1))
        return row


def sample_requests(w: MixedMultiWorkload) -> tuple[list[int], list[int], list[int], list[bool]]:
    """(tokens, adapter, rank, is_prefill) per request in batch order, seeded."""
    sw = w.shape_workload()
    lens, pre = sw.request_lengths, sw.request_is_prefill
    ranks = sample_request_ranks(sw, torch.Generator().manual_seed(w.seed + 1))
    gen = torch.Generator().manual_seed(w.seed + 7)
    probs = 1.0 / torch.arange(1, w.num_adapters + 1, dtype=torch.float64) ** w.zipf_alpha
    adapters = torch.multinomial(probs, len(lens), replacement=True, generator=gen).tolist()
    return lens, adapters, ranks, pre


def make_batch(w: MixedMultiWorkload, device="cuda"):
    """(x [T, h_in], token_adapter [T] int32, token_ranks [T] int32): requests in batch order, a
    request's tokens consecutive (not sorted by adapter or rank)."""
    lens, adapters, ranks, _ = sample_requests(w)
    gen = torch.Generator().manual_seed(w.seed + 3)
    x = torch.randn(sum(lens), w.h_in, generator=gen).to(device, DTYPES[w.dtype])
    n = torch.tensor(lens)
    rep = lambda v: torch.tensor(v, dtype=torch.int32).repeat_interleave(n).to(device)  # noqa: E731
    return x, rep(adapters), rep(ranks)
