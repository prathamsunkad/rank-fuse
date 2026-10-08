"""Compare an implementation against the B0 reference and explain any mismatch.

Used by tests/test_correctness.py, and handy while developing a kernel:

    from rankfuse.debug import compare
    report = compare("rf1_masked", lora, batch.x, batch.token_ranks, dtype="fp16",
                     seg_starts=batch.seg_starts)
    print(report)            # summary, per-rank error table, diagnosed mismatches
    assert report.ok

For each mismatched token the report tries to name the likely bug:
  * output row is zero                    -> token dropped / never scattered back
  * matches the result at another rank r' -> mask or grouping used the wrong rank
  * matches another token's expected row  -> scatter/index bug
  * matches expected times a constant     -> per-rank scale applied wrong
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import torch

from rankfuse.interface import REGISTRY, MatryoshkaLoRA

TOL = {"fp32": (1e-4, 1e-4), "fp16": (2e-2, 2e-2), "bf16": (8e-2, 8e-2)}
MAX_LISTED = 5  # mismatched tokens listed individually


@dataclass
class Report:
    impl: str
    dtype: str
    device: str
    ok: bool
    lines: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return "\n".join(self.lines)


def _close(a: torch.Tensor, b: torch.Tensor, rtol: float, atol: float) -> bool:
    return bool(torch.allclose(a, b, rtol=rtol, atol=atol))


def _row_at_rank(xf, A_t, B, r: int) -> torch.Tensor:
    return (xf @ A_t[:r].T) @ B[:r] if r > 0 else torch.zeros(B.shape[1], device=xf.device)


def _diagnose(i, got, expected, xf, A_t, B, token_ranks, scale, rtol, atol) -> str:
    """Best guess at why token i is wrong."""
    g, e, r_i = got[i], expected[i], int(token_ranks[i])
    if torch.count_nonzero(g) == 0 and torch.count_nonzero(e) > 0:
        return "output row is all zeros -> token dropped or never written back"

    # Computed at the wrong rank? Try every rank present in the batch, plus r_max.
    candidates = sorted(set(token_ranks.tolist()) | {A_t.shape[0]})
    s_i = scale(r_i)
    for r in candidates:
        if r == r_i:
            continue
        y_r = _row_at_rank(xf[i], A_t, B, r)
        for s in {s_i, scale(r), 1.0}:
            if _close(g, y_r * s, rtol, atol):
                return f"matches the rank-{r} result instead of rank {r_i} -> mask/grouping used the wrong rank"

    # Landed in the wrong row?
    for j in range(expected.shape[0]):
        if j != i and _close(g, expected[j], rtol, atol):
            return f"matches token {j}'s expected output (rank {int(token_ranks[j])}) -> scatter/index bug"

    # Off by a constant factor (scaling)?
    denom = float((e * e).sum())
    if denom > 0:
        c = float((g * e).sum()) / denom
        if abs(c - 1.0) > 1e-3 and _close(g, e * c, rtol, atol):
            return f"equals expected x {c:.4g} -> per-rank scale applied wrong"

    return "no simple pattern -> numerical bug or tolerance too tight for this dtype"


def compare(
    impl_name: str,
    lora: MatryoshkaLoRA,
    x: torch.Tensor,
    token_ranks: torch.Tensor,
    dtype: str = "fp32",
    seg_starts: torch.Tensor | None = None,
) -> Report:
    rtol, atol = TOL[dtype]
    device = x.device.type
    expected = REGISTRY["b0_reference"](x, lora, token_ranks)
    got = REGISTRY[impl_name](x, lora, token_ranks)

    T = x.shape[0]
    counts = Counter(token_ranks.tolist())
    ranks_str = "  ".join(f"{r}x{n}" for r, n in sorted(counts.items()))
    lines = [
        f"{impl_name} vs b0_reference  [{dtype}, {device}]",
        f"  batch: T={T} tokens, h_in={lora.h_in}, h_out={lora.h_out}, r_max={lora.r_max}",
        f"  ranks (rank x tokens): {ranks_str}   (batch max {max(counts)})",
        f"  tolerance: rtol={rtol:.0e} atol={atol:.0e}",
    ]

    if tuple(got.shape) != (T, lora.h_out):
        lines.insert(0, "FAILED: wrong output shape")
        lines.append(f"  got shape {tuple(got.shape)}, expected {(T, lora.h_out)}")
        return Report(impl_name, dtype, device, False, lines)
    if not torch.isfinite(got).all():
        bad = (~torch.isfinite(got)).any(dim=1).nonzero().squeeze(1).tolist()
        lines.insert(0, "FAILED: output contains NaN/inf")
        lines.append(f"  non-finite rows (tokens): {bad[:20]}{' ...' if len(bad) > 20 else ''}")
        return Report(impl_name, dtype, device, False, lines)

    got = got.float()
    err = (got - expected).abs()
    bad_elem = err > atol + rtol * expected.abs()
    bad_tok = bad_elem.any(dim=1)
    n_bad = int(bad_tok.sum())
    ok = n_bad == 0

    flat = int(err.argmax())
    ti, col = divmod(flat, lora.h_out)
    lines.insert(0, "PASSED" if ok else "FAILED")
    lines.append(
        f"  max |err| = {float(err.max()):.2e}  (token {ti}, rank {int(token_ranks[ti])}, col {col})"
        f"   mismatched tokens: {n_bad}/{T}"
    )

    lines.append("  per rank:   rank  tokens  bad   max|err|   max|expected|")
    bad_ranks = []
    for r in sorted(counts):
        sel = token_ranks == r
        nb = int(bad_tok[sel].sum())
        if nb:
            bad_ranks.append(r)
        lines.append(
            f"             {r:>5}  {int(sel.sum()):>6}  {nb:>3}   {float(err[sel].max()):.2e}"
            f"   {float(expected[sel].abs().max()):.2e}{'   <-- ' if nb else ''}"
        )

    if not ok:
        xf, A_t, B = x.float(), lora.A_t.float(), lora.B.float()
        table = lora.scale_per_rank or {}
        scale = lambda r: float(table.get(r, 1.0)) if lora.scale_per_rank else 1.0  # noqa: E731
        req_of = None
        if seg_starts is not None:
            req_of = torch.searchsorted(seg_starts[1:].cpu(), torch.arange(T), right=True)
        lines.append("  first mismatches:")
        for i in bad_tok.nonzero().squeeze(1).tolist()[:MAX_LISTED]:
            where = f"request {int(req_of[i])}, " if req_of is not None else ""
            why = _diagnose(i, got, expected, xf, A_t, B, token_ranks, scale, rtol, atol)
            lines.append(f"    token {i} ({where}rank {int(token_ranks[i])}): max|err| {float(err[i].max()):.2e} -> {why}")
        if n_bad > MAX_LISTED:
            lines.append(f"    ... and {n_bad - MAX_LISTED} more")
        if len(bad_ranks) < len(counts):
            lines.append(f"  hint: errors confined to rank(s) {bad_ranks}; other ranks are correct")
        elif n_bad < T:
            lines.append("  hint: every rank has some bad tokens but not all -> look at indexing, not the rank logic")
        else:
            lines.append("  hint: every token is wrong -> look at the core math/layout (A_t is [r_max, h_in])")

    return Report(impl_name, dtype, device, ok, lines)
