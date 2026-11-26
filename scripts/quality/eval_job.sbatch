#!/bin/bash
# sbatch eval_job.sbatch <model path> <manifest.json> <results dir>
#SBATCH -J rq-eval
#SBATCH -p ice-gpu,coc-gpu
# Allowed GPUs: never A100-80GB (reserved for timing). For 7B pass --constraint="H100|H200|A100-40GB" if memory-bound.
#SBATCH --gres=gpu:1 --constraint=H100|H200|A100-40GB|L40S|A40
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH -t 12:00:00
#SBATCH -o /storage/ice1/4/7/psunkad3/rankfuse/logs/quality/%x-%j.out
set -u
source /storage/ice1/4/7/psunkad3/rankfuse/wt/quality/scripts/quality/env_mlora.sh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv
D=/storage/ice1/4/7/psunkad3/rankfuse/wt/quality/scripts/quality
python -u $D/eval_ranks.py --model "$1" --manifest "$2" --out "$3" --tmp $RQ/quality_tmp/$SLURM_JOB_ID ${4:-}
rm -rf $RQ/quality_tmp/$SLURM_JOB_ID
