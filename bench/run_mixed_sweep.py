"""bench/run_sweep.py with B4 (S-LoRA bgmv) widened to delta widths 128 and 256.

    python bench/run_mixed_sweep.py bench/configs/rf2_mixed_a100_r128.yaml --out results/.../r128.csv --resume

Exactly run_sweep.main() (same arguments, timing and CSV), after rankfuse.baselines.slora_bgmv_wide
.enable(), so b4_slora_copies / b4_slora_shared also run at r_max 128 and 256 (Stage 1's B4 stops at
64). For the mixed-batching sweeps (bench/configs/rf2_mixed_a100_*.yaml).
After the sweep, with --out, it measures the per-step sort / prepare cost of the same workloads
(bench/mixed_prepare_cost.py) into `<out stem>-prepare.csv`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rankfuse.baselines import slora_bgmv_wide  # noqa: E402

import run_sweep  # noqa: E402

if __name__ == "__main__":
    slora_bgmv_wide.enable()
    run_sweep.main()
    if "--out" in sys.argv:
        import mixed_prepare_cost

        mixed_prepare_cost.after_sweep("single", sys.argv[1], sys.argv[sys.argv.index("--out") + 1])
