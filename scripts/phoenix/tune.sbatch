#!/bin/bash
# GPU tests, then per-width tuning of the split-K launch settings (bench/tune_blocks.py).
#   sbatch --constraint=A100-80GB --exclude=atl1-1-01-002-28-0 \
#       --export=ALL,REQUIRE_GPU="NVIDIA A100 80GB PCIe" scripts/phoenix/tune.sbatch
# Writes results/raw/tune/<gpu>.csv and rankfuse/kernels/tuned_a100.py (review it before committing).
#SBATCH -J rankfuse-tune
#SBATCH -A gts-smussmann3
#SBATCH -q inferno
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o results/slurm-tune-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load anaconda3
module load cuda
eval "$(conda shell.bash hook)"
conda activate "${ENV_PREFIX:-$HOME/scratch/envs/rankfuse}"
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$HOME/scratch/.triton}

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if [ -n "${REQUIRE_GPU:-}" ] && [ "$GPU_NAME" != "$REQUIRE_GPU" ]; then
    echo "got GPU '$GPU_NAME', need '$REQUIRE_GPU'" >&2
    exit 1
fi
nvidia-smi
[ -n "${SKIP_TESTS:-}" ] || RANKFUSE_TUNED=0 python -m pytest -q -rs
python bench/tune_blocks.py --csv results/raw/tune/a100.csv --table rankfuse/kernels/tuned_a100.py ${TUNE_ARGS:-}
