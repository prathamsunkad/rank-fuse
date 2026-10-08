#!/bin/bash
# Profile one implementation on one workload with Nsight Compute (run inside a GPU job).
#   bash scripts/phoenix/ncu_profile.sh b1_padded 32 1     # impl, num_requests, tokens_per_request
# Reports DRAM bytes and achieved bandwidth per kernel; compare with `python -m rankfuse.roofline`.
set -euo pipefail

IMPL=${1:?impl name}
REQUESTS=${2:-32}
TPR=${3:-1}
OUT=results/ncu/${IMPL}-r${REQUESTS}-t${TPR}
mkdir -p results/ncu

ncu --set full -o "$OUT" -f python - <<EOF
import torch, rankfuse
from rankfuse.workloads import Workload, make_batch, make_lora
w = Workload(${REQUESTS}, ${TPR}, dist="uniform", dtype="fp16")
lora, b = make_lora(w, "cuda"), make_batch(w, "cuda")
fn = rankfuse.REGISTRY["${IMPL}"]
for _ in range(3):
    fn(b.x, lora, b.token_ranks)
torch.cuda.synchronize()
EOF

ncu --import "${OUT}.ncu-rep" --page raw --metrics dram__bytes.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum
