#!/bin/bash
# Kernel benchmark sweep (Level 1) on PACE ICE.
#   sbatch scripts/ice/sweep.sbatch                                                # Llama-3.1-8B decode
#   sbatch --export=ALL,CONFIG=bench/configs/llama3_8b_prefill.yaml scripts/ice/sweep.sbatch
#   sbatch --gres=gpu:h100:1 scripts/ice/sweep.sbatch                              # another GPU type
#
# Headline numbers come from ONE GPU type (H200; H100, then A100 if H200 is unavailable).
# Results land in results/raw/<config>-<gpu>-<time>.csv with the git commit recorded per row.
#SBATCH -J rankfuse-sweep
#SBATCH -p ice-gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o results/slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/ice/env.sh

CONFIG=${CONFIG:-bench/configs/llama3_8b_decode.yaml}
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -m pytest -q
python bench/run_sweep.py "$CONFIG"
