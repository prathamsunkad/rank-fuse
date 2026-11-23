#!/bin/bash
# Did a kernel change move rf1_fused? Runs OLD_DIR and NEW_DIR (git worktrees) in turn on one GPU, so
# hardware cannot explain a difference. Repeats of one side give the noise floor.
#   sbatch --constraint=A100-80GB --exclude=atl1-1-01-002-28-0 \
#       --export=ALL,OLD_DIR=...,NEW_DIR=...,OUT_DIR=... scripts/phoenix/rf1_old_vs_new.sbatch
#SBATCH -J rf1-old-new
#SBATCH -A gts-smussmann3
#SBATCH -q embers
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=32G
#SBATCH -t 01:30:00
#SBATCH -o results/slurm-rf1-old-new-%j.out

set -euo pipefail
CONFIG=$(realpath "$SLURM_SUBMIT_DIR/bench/configs/rf1_old_vs_new.yaml")
module load anaconda3
module load cuda
eval "$(conda shell.bash hook)"
conda activate "${ENV_PREFIX:-$HOME/scratch/envs/rankfuse}"
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$HOME/scratch/.triton}

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
REQUIRE_GPU=${REQUIRE_GPU:-NVIDIA A100 80GB PCIe}
if [ "$GPU_NAME" != "$REQUIRE_GPU" ]; then
    echo "got GPU '$GPU_NAME', need '$REQUIRE_GPU'" >&2
    exit 1
fi
mkdir -p "$OUT_DIR"
for round in 1 2 3; do
    for side in old new; do
        if [ "$side" = old ]; then dir=$OLD_DIR; else dir=$NEW_DIR; fi
        cd "$dir"  # run_sweep imports the kernels next to it and records this directory's commit
        python bench/run_sweep.py "$CONFIG" --out "$OUT_DIR/${side}_${round}.csv"
    done
done
