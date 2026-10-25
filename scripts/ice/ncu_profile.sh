#!/bin/bash
# Profile one implementation on one workload with Nsight Compute. Run inside a GPU job, e.g.
#   salloc -p ice-gpu --gres=gpu:h200:1 -t 1:00:00 --cpus-per-task=8 --mem=64G
#   bash scripts/ice/ncu_profile.sh rf1_fused 32 1 4096 4096 64
# Arguments: impl, num_requests, tokens_per_request, h_in, h_out, r_max (defaults shown above).
# Prints DRAM bytes, achieved bandwidth, L2 traffic and time per kernel; compare the DRAM
# bytes with `python -m rankfuse.roofline` for the same shape.
set -euo pipefail
source scripts/ice/env.sh
module load cuda   # provides ncu
ncu --version | tail -1

IMPL=${1:?impl name}
REQUESTS=${2:-32}
TPR=${3:-1}
H_IN=${4:-4096}
H_OUT=${5:-4096}
R_MAX=${6:-64}
OUT=results/ncu/${IMPL}-r${REQUESTS}-t${TPR}-${H_IN}x${H_OUT}-r${R_MAX}
mkdir -p results/ncu

# Three calls: the first compiles the Triton kernels. ncu profiles every launch, so the CSV
# has one row per kernel per call; read the rows of the last call.
ncu --set full -o "$OUT" -f python - <<EOF
import torch, rankfuse
from rankfuse.workloads import Workload, make_batch, make_lora
w = Workload(${REQUESTS}, ${TPR}, h_in=${H_IN}, h_out=${H_OUT}, r_max=${R_MAX}, dist="uniform", dtype="fp16")
lora, b = make_lora(w, "cuda"), make_batch(w, "cuda")
fn = rankfuse.REGISTRY["${IMPL}"]
for _ in range(3):
    fn(b.x, lora, b.token_ranks)
torch.cuda.synchronize()
EOF

# On ICE, ncu saves a zstd-compressed .ncu-repz that `ncu --import` cannot open; unpack it.
if [ ! -f "${OUT}.ncu-rep" ] && [ -f "${OUT}.ncu-repz" ]; then
    zstd -q -d -f "${OUT}.ncu-repz" -o "${OUT}.ncu-rep"
fi

# Metric names as in ncu 2026.3; lts__t_sectors are 32-byte L2 sectors.
ncu --import "${OUT}.ncu-rep" --page raw --csv --metrics \
gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sectors.sum,lts__t_sector_hit_rate.pct \
> "${OUT}.csv"
echo "wrote ${OUT}.csv"
