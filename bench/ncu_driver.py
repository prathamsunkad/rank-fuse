"""Run one registered implementation 3 times on one workload, for Nsight Compute to profile.

    python bench/ncu_driver.py <impl> [requests tokens_per_request h_in h_out r_max dist dtype]
    # defaults: 32 1 4096 4096 64 uniform fp16

The implementation's per-step metadata (`prepare`: batch maximum rank, rank order, block table,
B3's padded copies and vLLM metadata, ...) is built before the calls, so the profiled call has
no host sync and launches only its own kernels. The first two calls compile and warm up; the
last sits between cudaProfilerStart/Stop, so with `ncu --profile-from-start off` (as in
scripts/ice/ncu_profile.sh) only that call is profiled, whatever kernels cuBLAS picks.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rankfuse.baselines import slora_bgmv  # noqa: E402
from rankfuse.interface import REGISTRY  # noqa: E402
from rankfuse.workloads import Workload, make_batch, make_lora  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    name = sys.argv[1]
    defaults = ["32", "1", "4096", "4096", "64", "uniform", "fp16"]
    a = sys.argv[2:] + defaults[len(sys.argv) - 2:]
    w = Workload(int(a[0]), int(a[1]), h_in=int(a[2]), h_out=int(a[3]), r_max=int(a[4]), dist=a[5], dtype=a[6], seed=0)
    lora, batch = make_lora(w, "cuda"), make_batch(w, "cuda")
    slora_bgmv.enable_if_requested([name])  # B4 is registered only on request
    impl = REGISTRY[name]
    args = (batch.x, lora, batch.token_ranks)
    kwargs = impl.prepare(*args) if impl.prepare else {}
    for _ in range(2):
        impl(*args, **kwargs)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    impl(*args, **kwargs)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(f"{name}: {w.num_requests} x {w.tokens_per_request} tokens, {w.h_in} x {w.h_out}, r_max {w.r_max}, "
          f"{w.dist}, {w.dtype}{' (prepared: ' + ', '.join(kwargs) + ')' if kwargs else ''}", flush=True)


if __name__ == "__main__":
    main()
