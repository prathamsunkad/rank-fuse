#!/bin/bash
# Run correctness tests on the GPU, then a benchmark sweep.
#   sbatch scripts/phoenix/sweep.sbatch                      # decode sweep
#   sbatch --export=ALL,CONFIG=bench/configs/prefill.yaml scripts/phoenix/sweep.sbatch
#
# EDIT BEFORE FIRST USE:
#   -A : your PI's charge account (`pace-quota` lists yours), e.g. gts-<pi-username>
#   -q : inferno (charged) or embers (free, preemptible backfill)
#   --gres : H100 by default. Report headline numbers on ONE GPU type; options include
#            gpu:A100:1, gpu:V100:1, gpu:L40S:1 depending on what your account can access.
#SBATCH -J rankfuse-sweep
#SBATCH -A gts-CHANGEME
#SBATCH -q embers
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o results/slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load anaconda3
module load cuda
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME:-rankfuse}"

CONFIG=${CONFIG:-bench/configs/decode.yaml}
nvidia-smi
python -m pytest -q
python bench/run_sweep.py "$CONFIG"
