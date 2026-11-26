#!/bin/bash
# Run every line of a queue file (one mlora_train.py argument list per line) in order.
# Finished runs (done.json) are skipped and unfinished ones resume from ckpt.pt, so the same job can
# be resubmitted after a timeout.   sbatch train_queue.sbatch <queue.txt>
#SBATCH -J rq-train
#SBATCH -p ice-gpu,coc-gpu
# Allowed GPUs: never A100-80GB (reserved for timing). For 7B pass --constraint="H100|H200|A100-40GB" if memory-bound.
#SBATCH --gres=gpu:1 --constraint=H100|H200|A100-40GB|L40S|A40
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH -t 16:00:00
#SBATCH -o /storage/ice1/4/7/psunkad3/rankfuse/logs/quality/%x-%j.out
set -u
Q=$1
source /storage/ice1/4/7/psunkad3/rankfuse/wt/quality/scripts/quality/env_mlora.sh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv
D=/storage/ice1/4/7/psunkad3/rankfuse/wt/quality/scripts/quality
while IFS= read -r line; do
  [ -z "$line" ] && continue; [[ "$line" == \#* ]] && continue
  echo "=== $(date +%T) $line"
  eval python -u $D/mlora_train.py $line || echo "FAILED: $line"
done < "$Q"
echo "queue finished $(date)"
