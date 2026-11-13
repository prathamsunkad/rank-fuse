#!/bin/bash
#SBATCH -J rf-p2-j6-tables
#SBATCH -p ice-cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o results/slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/ice/env.sh

mkdir -p results
mkdir -p results/h200_2026-11-12_multi_adapter/tables

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

echo "=== Building All Tables and Reports ==="
python "$SLURM_SUBMIT_DIR/results/h200_2026-11-12_multi_adapter/build_tables.py"

echo "=== J6 Tables Complete ==="
