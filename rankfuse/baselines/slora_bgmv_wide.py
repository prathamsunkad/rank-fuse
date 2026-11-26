"""B4 (S-LoRA's real bgmv CUDA kernel) for one adapter at r_max 128 and 256.

Stage 1's `slora_bgmv` accepts delta widths 8/16/32/64 only (`slora_bgmv.NARROW_SIZES`). The A100
r_max 128/256 sweeps (2026-11-23) compiled narrow sizes 128 and 256 into `slora_ext` (with hidden
sizes 512 and up, see `slora_ext/bgmv/bgmv_config.h`) for the multi-adapter B4. `enable()` here
lets the single-adapter `b4_slora_copies` / `b4_slora_shared` use them too, by widening
`slora_bgmv.NARROW_SIZES` in this process and then registering B4 as usual. Nothing else about the
kernel or the pool layout changes: rank tiers are still stored at multiples of 8, the shrink runs at
each token's rank and the expand reads B^T rows in 8-element vectors.

The kernel picks the expand by feat_in < feat_out, so a delta width must be below both hidden sizes
(`check_shape`); bench/run_mixed_sweep.py only runs 4096 x 4096. Used by bench/run_mixed_sweep.py
and tests/test_mixed_single.py; `import rankfuse` does not call it.
"""

from __future__ import annotations

from rankfuse.baselines import slora_bgmv as sb

STAGE1_NARROW_SIZES = (8, 16, 32, 64)
WIDE_NARROW_SIZES = STAGE1_NARROW_SIZES + (128, 256)


def check_shape(r_max: int, h_in: int, h_out: int) -> str | None:
    """Why B4 cannot run this shape, or None."""
    if r_max > WIDE_NARROW_SIZES[-1] or r_max % 8:
        return f"r_max={r_max}: needs a multiple of 8 up to {WIDE_NARROW_SIZES[-1]}"
    width = next(n for n in WIDE_NARROW_SIZES if n >= r_max)
    if width > STAGE1_NARROW_SIZES[-1] and (width >= min(h_in, h_out) or min(h_in, h_out) < 512):
        return f"delta width {width} is compiled only with hidden sizes of 512 and up, above the width"
    return None


def enable() -> None:
    """Widen B4's delta widths to 128 and 256 in this process, then register b4_slora_* (idempotent)."""
    sb.NARROW_SIZES = WIDE_NARROW_SIZES
    sb.enable()
