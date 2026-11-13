#!/bin/bash
# Profile one implementation on one workload with Nsight Compute. Run inside a GPU job, e.g.
#   salloc -p ice-gpu --gres=gpu:h200:1 -t 1:00:00 --cpus-per-task=8 --mem=64G
#   bash scripts/ice/ncu_profile.sh rf1_fused 32 1 4096 4096 64
#   for impl in b1_padded b3_vllm rf1_fused rf2_segment; do bash scripts/ice/ncu_profile.sh $impl 2048 1; done
# Arguments: impl, num_requests, tokens_per_request, h_in, h_out, r_max, dist, dtype (defaults:
# 32 1 4096 4096 64 uniform fp16). Only the last of three calls is profiled (bench/ncu_driver.py,
# per-step metadata built before it, so no host sync); ncu defaults apply: caches flushed and
# clocks locked to base for each kernel. Prints one row per kernel of that call: time, DRAM bytes
# and bandwidth, L2 traffic, SM and tensor-pipe use, launch shape and occupancy limits. Compare
# the DRAM bytes with `python -m rankfuse.roofline` for the same shape.
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
DIST=${7:-uniform}
DTYPE=${8:-fp16}
PROFILE_SORT=${PROFILE_SORT:-0}
OUT=results/ncu/${IMPL}-r${REQUESTS}-t${TPR}-${H_IN}x${H_OUT}-r${R_MAX}-${DIST}-${DTYPE}
mkdir -p results/ncu

SORT_ARG=()
if [ "$PROFILE_SORT" = 1 ]; then
    SORT_ARG=(--profile-sort)
    OUT="${OUT}-sort"
fi
ncu --profile-from-start off --set full -o "$OUT" -f \
    python bench/ncu_driver.py "$IMPL" "$REQUESTS" "$TPR" "$H_IN" "$H_OUT" "$R_MAX" "$DIST" "$DTYPE" "${SORT_ARG[@]}"

# On ICE, ncu saves a zstd-compressed .ncu-repz that `ncu --import` cannot open; unpack it.
if [ ! -f "${OUT}.ncu-rep" ] && [ -f "${OUT}.ncu-repz" ]; then
    zstd -q -d -f "${OUT}.ncu-repz" -o "${OUT}.ncu-rep"
fi

# Metric names as in ncu 2026.3; lts__t_sectors are 32-byte L2 sectors.
ncu --import "${OUT}.ncu-rep" --page raw --csv --metrics \
gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sectors.sum,lts__t_sector_hit_rate.pct,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,\
launch__grid_size,launch__block_size,launch__registers_per_thread,\
launch__shared_mem_per_block_dynamic,launch__shared_mem_per_block_static,\
launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem \
> "${OUT}.csv"
echo "wrote ${OUT}.csv"
