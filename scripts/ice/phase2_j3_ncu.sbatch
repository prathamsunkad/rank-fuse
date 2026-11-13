#!/bin/bash
#SBATCH -J rf-p2-j3-ncu
#SBATCH -p ice-gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o results/slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/ice/env.sh
module load cuda

mkdir -p results
mkdir -p results/h200_2026-11-12_multi_adapter/ncu

echo "=== System & GPU Info ==="
nvidia-smi

python -c "import sys; print('Python version:', sys.version)"
python -c "import torch, triton; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'Triton:', triton.__version__)"
python -c "import torch; dev = torch.cuda.get_device_name(); assert 'H200' in dev, f'Expected H200, got {dev}'; print('Device name:', dev)"

COMMIT="${COMMIT:-$(git rev-parse HEAD)}"
SNAP_DIR="$RF_ROOT/snap-$SLURM_JOB_ID"
git worktree add --detach "$SNAP_DIR" "$COMMIT"
cleanup() {
    cd "$SLURM_SUBMIT_DIR"
    git worktree remove --force "$SNAP_DIR" || rm -rf "$SNAP_DIR" || true
}
trap cleanup EXIT

cd "$SNAP_DIR"
export PYTHONPATH="$SNAP_DIR:${PYTHONPATH:-}"

python -c "import rankfuse; print('Snapshot commit:', '$COMMIT'); print('rankfuse file:', rankfuse.__file__); assert 'snap-' + '$SLURM_JOB_ID' in str(rankfuse.__file__), f'{rankfuse.__file__} not inside snap-$SLURM_JOB_ID'"

OUT_DIR="$SLURM_SUBMIT_DIR/results/h200_2026-11-12_multi_adapter/ncu"
mkdir -p "$OUT_DIR"

IMPLS=("rf1_ma" "rf1_ma_flat" "b1_ma_padded" "b3_ma" "b4_ma_copies" "b4_ma_shared")

profile_one() {
    local IMPL=$1
    local OUT=$2
    shift 2

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

# Profile all 6 implementations across 3 key configurations
for IMPL in "${IMPLS[@]}"; do
    profile_one "$IMPL" "${OUT_DIR}/${IMPL}-t32-n16" 32 16 4096 4096 64 mostly_low uniform
    profile_one "$IMPL" "${OUT_DIR}/${IMPL}-t128-n16" 128 16 4096 4096 64 mostly_low uniform
    profile_one "$IMPL" "${OUT_DIR}/${IMPL}-t128-n64" 128 64 4096 4096 64 mostly_low uniform
done

echo "=== J3 NCU Complete ==="
