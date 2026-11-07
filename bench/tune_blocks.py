"""Tune the split-K launch settings per (width, token range) and write the shared table.

    python bench/tune_blocks.py --csv results/raw/tune/a100.csv --table rankfuse/kernels/tuned_a100.py

Tuning inputs are single-rank batches: every token at rank W, so rf1_fused runs at width W. These
batches are only used here, never reported as results. Every split-K kernel then looks its settings
up in the table (rankfuse_fused._config): RF1, rf2_segment and rf2_index at the batch width, the
per-bucket kernels at each bucket's width, so RF1 gets every tuned setting RF2 does.

Search, per width and token count (h 4096 x 4096, fp16, add mode, graph timing, L2 flushed):
  1. BM x shrink warps x tiled-expand tile BN x expand warps, at the default shrink tiles.
  2. Shrink BH x pipeline stages, at the best of step 1.
Then for each token range the per-size winners and the default are re-timed on every size in the
range over 3 seeds; the best geometric mean is kept only if it beats the default by more than 2%
(the repeat noise), else the default stays. Ranges up to 128 tokens (decode, loop expand) are not
tuned. Combinations keeping v [BM, W] above 64 KiB of fp32 registers are skipped.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import statistics
from pathlib import Path

import torch

from run_sweep import gpu_name, git_sha, time_graph  # also puts the repo on sys.path
from rf1_ablation import settings

from rankfuse.interface import REGISTRY  # noqa: E402
from rankfuse.kernels import rankfuse_fused as rf  # noqa: E402
from rankfuse.workloads import Workload, make_batch, make_lora  # noqa: E402

WIDTHS = [16, 32, 64, 128, 256]
SIZES = [256, 1024, 2048, 8192]
STEP1 = dict(bm=[16, 32, 64, 128], shrink_warps=[4, 8], bn=[128, 256], expand_warps=[4, 8])
STEP2 = dict(bh=[64, 128], stages=[2, 3, 4])
MAX_V_ELEMS = 64 * 1024 // 4
KEEP_DEFAULT_WITHIN = 0.02


def default_cfg(width: int, tokens: int) -> dict:
    return dict(bm=rf._block_m(tokens, width), shrink_warps=4, bn=rf.EXPAND_TILED_BLOCK_N,
                expand_warps=rf.EXPAND_TILED_WARPS, bh=rf.SPLITK_BLOCK_H, stages=rf.SPLITK_STAGES)


class Bench:
    def __init__(self, writer, gname: str, sha: str):
        self.writer, self.gname, self.sha = writer, gname, sha
        self.cases = {}

    def case(self, width: int, tokens: int, seed: int):
        key = (width, tokens, seed)
        if key not in self.cases:
            w = Workload(tokens, 1, r_max=width, dist="all_max", seed=seed)
            lora, batch = make_lora(w, "cuda"), make_batch(w, "cuda")
            impl = REGISTRY["rf1_fused"]
            args = (batch.x, lora, batch.token_ranks)
            kwargs = impl.prepare(*args)
            target = torch.zeros(batch.x.shape[0], w.h_out, dtype=batch.x.dtype, device="cuda")

            def fn(x, l, ranks, **kw):
                return impl(x, l, ranks, out=target, add=True, **kw)

            self.cases[key] = (fn, args, kwargs)
        return self.cases[key]

    def time(self, width: int, tokens: int, seed: int, cfg: dict, stage: str, reps=(5, 10, 10)) -> float | None:
        fn, args, kwargs = self.case(width, tokens, seed)
        table = {(width, rf._token_range(tokens)): cfg}
        try:
            with settings({"TUNED": table, "TUNED_GPU": self.gname}):
                us = statistics.median(time_graph(fn, args, kwargs, *reps, True))
            error = ""
        except Exception as e:  # out of resources etc.: record and skip
            torch.cuda.synchronize()
            us, error = None, f"{type(e).__name__}: {e}"[:200]
        self.writer.writerow(dict(stage=stage, gpu=self.gname, git_sha=self.sha, width=width, tokens=tokens, seed=seed,
                                  **cfg, median_us="" if us is None else f"{us:.2f}", error=error))
        return us


def search(b: Bench, width: int, tokens: int) -> dict:
    base = default_cfg(width, tokens)
    best, best_us = base, b.time(width, tokens, 0, base, "default")
    for combo in itertools.product(*STEP1.values()):
        cfg = dict(base, **dict(zip(STEP1, combo)))
        if cfg["bm"] * width > MAX_V_ELEMS:
            continue
        us = b.time(width, tokens, 0, cfg, "step1")
        if us is not None and (best_us is None or us < best_us):
            best, best_us = cfg, us
    step1 = best
    for combo in itertools.product(*STEP2.values()):
        cfg = dict(step1, **dict(zip(STEP2, combo)))
        us = b.time(width, tokens, 0, cfg, "step2")
        if us is not None and us < best_us:
            best, best_us = cfg, us
    return best


def pick(b: Bench, width: int, sizes: list[int], candidates: list[dict]) -> tuple[dict | None, float]:
    """Re-time candidates on every size over 3 seeds; return the winner (None = keep the default)
    and its geometric-mean speedup over the default."""
    def geo(cfg_for_size):
        logs = []
        for t in sizes:
            for seed in range(3):
                us = b.time(width, t, seed, cfg_for_size(t), "confirm", reps=(10, 30, 20))
                if us is None:
                    return None
                logs.append(math.log(us))
        return math.exp(sum(logs) / len(logs))

    default_us = geo(lambda t: default_cfg(width, t))
    best, best_us = None, default_us
    for cfg in candidates:
        us = geo(lambda t, c=cfg: c)
        if us is not None and us < best_us:
            best, best_us = cfg, us
    gain = default_us / best_us
    return (best if gain > 1 + KEEP_DEFAULT_WITHIN else None), gain


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True)
    p.add_argument("--table", required=True)
    p.add_argument("--widths", type=int, nargs="*", default=WIDTHS)
    p.add_argument("--sizes", type=int, nargs="*", default=SIZES)
    a = p.parse_args()
    Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
    gname, sha = gpu_name(), git_sha()
    fields = ["stage", "gpu", "git_sha", "width", "tokens", "seed", *STEP1, *STEP2, "median_us", "error"]
    table, notes = {}, []
    with open(a.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        b = Bench(writer, gname, sha)
        for width in a.widths:
            ranges: dict[int, list[int]] = {}
            for t in a.sizes:
                if t > rf.TOKEN_RANGES[0]:
                    ranges.setdefault(rf._token_range(t), []).append(t)
            for rng, sizes in sorted(ranges.items()):
                winners = []
                for t in sizes:
                    w = search(b, width, t)
                    if w not in winners:
                        winners.append(w)
                    f.flush()
                cfg, gain = pick(b, width, sizes, winners)
                notes.append(f"# width {width}, range {rng} ({sizes} tokens): "
                             + (f"{gain:.3f}x over default" if cfg else f"default kept (best {gain:.3f}x)"))
                print(notes[-1], flush=True)
                if cfg:
                    table[(width, rng)] = {k: cfg[k] for k in fields if k in cfg}
                f.flush()
    lines = [f'"""Launch settings tuned by bench/tune_blocks.py on {gname} (commit {sha}).',
             "",
             "Keys: (width, token range), ranges from rankfuse_fused.TOKEN_RANGES. Missing entries use the",
             'default rules. Set RANKFUSE_TUNED=0 to ignore this table."""',
             "", *notes, "", f"GPU = {gname!r}", "TUNED = {"]
    lines += [f"    {k!r}: {v!r}," for k, v in sorted(table.items())]
    lines += ["}", ""]
    Path(a.table).write_text("\n".join(lines))
    print(f"-> {a.table}")


if __name__ == "__main__":
    main()
