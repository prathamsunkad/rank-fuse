#!/bin/bash
#SBATCH -J rf-phase1b-verify
#SBATCH -p ice-gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -o results/slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/ice/env.sh

mkdir -p results

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

echo "=== Prohibited Pattern Checks ==="
echo "Checking torch.bmm / torch.matmul..."
! grep -rn "torch\.bmm\|torch\.matmul" rankfuse/kernels/rankfuse_ma.py rankfuse/paging/*.py
echo "Checking .item() / .tolist()..."
! grep -rn "\.item()\|\.tolist()" rankfuse/kernels/rankfuse_ma.py rankfuse/paging/*.py
echo "Checking BM = 1..."
! grep -rn "BM = 1" rankfuse/kernels/rankfuse_ma.py
echo "Prohibited pattern checks passed!"

echo "=== Running Multi-Adapter and Paging Tests ==="
EXIT_CODE=0
python -m pytest -v -rsfE --tb=short tests/test_multi_adapter.py tests/test_paging.py || EXIT_CODE=$?

echo "=== Running Full Test Suite ==="
python -m pytest -v -rsfE --tb=short || EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "=== Verification Suite FAILED with exit code $EXIT_CODE ==="
    exit $EXIT_CODE
fi

echo "=== Phase 1b Verification Complete ==="
