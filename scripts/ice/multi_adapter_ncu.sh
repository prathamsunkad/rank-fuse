#!/bin/bash
# Profile multi-adapter implementations on H200 with Nsight Compute.
# Usage:
#   bash scripts/ice/multi_adapter_ncu.sh <impl> <out_prefix> [tokens] [adapters] [h_in] [h_out] [r_max] [mix] [pop]
# Or with no arguments (or RUN_ALL=1): profiles all 6 implementations on selected workloads into results/h200_2026-11-12_multi_adapter/ncu/
set -euo pipefail

source scripts/ice/env.sh
module load cuda
ncu --version | tail -1

profile_one() {
    local IMPL=$1
    local OUT=$2
    shift 2
    mkdir -p "$(dirname "$OUT")"

    echo "=== Profiling $IMPL -> $OUT ==="
    ncu --profile-from-start off --set full -o "$OUT" -f \
        python bench/multi_adapter_ncu_driver.py "$IMPL" "$@"

    if [ ! -f "${OUT}.ncu-rep" ] && [ -f "${OUT}.ncu-repz" ]; then
        zstd -q -d -f "${OUT}.ncu-repz" -o "${OUT}.ncu-rep"
    fi

    ncu --import "${OUT}.ncu-rep" --page raw --csv --metrics \
gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sectors.sum,\
lts__t_sector_hit_rate.pct,sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,\
launch__grid_size,launch__block_size,launch__registers_per_thread,\
launch__shared_mem_per_block_dynamic,launch__shared_mem_per_block_static,\
launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem \
    > "${OUT}.csv"
    echo "Wrote ${OUT}.csv"
}

if [ $# -ge 2 ]; then
    profile_one "$@"
else
    OUT_DIR="results/h200_2026-11-12_multi_adapter/ncu"
    mkdir -p "$OUT_DIR"
    IMPLS=("rf1_ma" "rf1_ma_flat" "b1_ma_padded" "b3_ma" "b4_ma_copies" "b4_ma_shared")

    # Selected configs for NCU analysis:
    # 1. Decode: T=32, N=16, mostly_low
    # 2. Decode: T=128, N=16, mostly_low
    # 3. Decode: T=128, N=64, mostly_low
    for IMPL in "${IMPLS[@]}"; do
        profile_one "$IMPL" "${OUT_DIR}/${IMPL}-t32-n16" 32 16 4096 4096 64 mostly_low uniform
        profile_one "$IMPL" "${OUT_DIR}/${IMPL}-t128-n16" 128 16 4096 4096 64 mostly_low uniform
        profile_one "$IMPL" "${OUT_DIR}/${IMPL}-t128-n64" 128 64 4096 4096 64 mostly_low uniform
    done
fi
