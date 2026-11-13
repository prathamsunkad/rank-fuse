#!/bin/bash
#SBATCH -J rankfuse-analysis
#SBATCH -p ice-cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH -o results/slurm-%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
DATE=${PROFILE_DATE:?PROFILE_DATE is required}
ROOT="results/h200_ice_${DATE}"
trap 'rc=$?; printf "%s\n" "[$(date -Is)] FAILED analysis job=$SLURM_JOB_ID exit=$rc; inspect results/slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out and resume with the exact dependency chain in STATUS.md" >> "$ROOT/STATUS.md"; exit "$rc"' EXIT
python analysis/profile_plots.py "$ROOT"
printf '%s\n' "[$(date -Is)] DONE analysis job=$SLURM_JOB_ID" >> "$ROOT/STATUS.md"
trap - EXIT
