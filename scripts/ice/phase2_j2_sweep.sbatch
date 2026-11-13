#!/bin/bash
#SBATCH -J rf-p2-j2-sweep
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

mkdir -p results
mkdir -p results/h200_2026-11-12_multi_adapter/source

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

OUT_CSV="$SLURM_SUBMIT_DIR/results/h200_2026-11-12_multi_adapter/source/multi_adapter_sweep.csv"
echo "=== Running Multi-Adapter Sweep -> $OUT_CSV ==="
python bench/multi_adapter_sweep.py --out "$OUT_CSV" --replays 200

echo "=== J2 Sweep Complete ==="
