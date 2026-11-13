#!/bin/bash
# Build the pinned environment away from the login-node memory limit.
#SBATCH -J rankfuse-setup
#SBATCH -p ice-cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH -o results/slurm-%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
bash scripts/ice/setup_env.sh
