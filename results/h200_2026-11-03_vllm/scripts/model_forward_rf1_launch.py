"""bench/model_forward.py restricted to none / b1 / b3 / rf1, with rankfuse_lora either as it is
now (launch-path caches) or as it was before (the saved copy of rankfuse_fused.py given as
the second argument). Same workloads, timing and output format as bench/model_forward.py.

    python results/h200_2026-11-03_vllm/scripts/model_forward_rf1_launch.py out.json [old_rankfuse_fused.py]
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "bench"))
import model_forward as mf  # noqa: E402
from rankfuse.kernels import rankfuse_fused as rf  # noqa: E402

mf.BACKENDS = ["none", "b1", "b3", "rf1"]
if len(sys.argv) > 2:
    from rankfuse.interface import REGISTRY

    for name in ("rf1_fused", "rf1_single", "rf2_fused"):  # the saved copy registers them again
        REGISTRY.pop(name, None)
    spec = importlib.util.spec_from_file_location("rankfuse_fused_before", sys.argv[2])
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    rf.rankfuse_lora = old.rankfuse_lora  # the harness calls rf.rankfuse_lora
    print("using the saved (before) rankfuse_lora from", sys.argv[2], flush=True)
sys.argv = sys.argv[:2]
mf.main()
