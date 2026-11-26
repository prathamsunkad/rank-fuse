#!/bin/bash
# 7B RankFuse rerun on an H200 with the log-prob rank check, a warm-up grid and a fine load grid,
# plus a 0.5B check-only confirmation of the rank check. One job, <= 4 h.
#   sbatch bench/e2e/jobs/h200_7b_rerun.sbatch
#SBATCH -J rf-e2e-q7fine
#SBATCH -p ice-gpu,coc-gpu
#SBATCH -C H200
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o /storage/ice1/4/7/psunkad3/rankfuse/wt/e2e/results/e2e_2026-11-24/q7_fine/slurm-%j.out
set -uo pipefail
WT=/storage/ice1/4/7/psunkad3/rankfuse/wt/e2e
cd "$WT"
source /storage/ice1/4/7/psunkad3/rankfuse/rf_paths.sh
source scripts/ice/env.sh
export PYTHONPATH=$WT TORCH_EXTENSIONS_DIR=$RF_SCRATCH/cache/torch_extensions_e2e VLLM_CACHE_ROOT=$RF_SCRATCH/cache/vllm_e2e
D=$WT/results/e2e_2026-11-24
A=$RF_ADAPTERS/synthetic_r256
EA="--max-model-len 4096 --max-num-seqs 256 --gpu-memory-utilization 0.85 --seed 0"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee $D/q05_check/gpu.txt > $D/q7_fine/gpu.txt

echo "=== 0.5B rank-check confirmation $(date +%T)"
for s in rankfuse stock-full; do
  python bench/e2e/run_system.py --system $s --model Qwen/Qwen2.5-0.5B-Instruct --adapters-dir $A/qwen2.5-0.5b \
    --check-only --rates inf --out $D/q05_check --engine-args "$EA" 2>&1 | grep -v "ready in" | tee -a $D/q05_check/driver.log
done

echo "=== 7B fine grid $(date +%T)"
export N=3000 FIXED_RANKS="8" RATES="inf 8 12 16 20 24 28 32 40 48" ENGINE_ARGS="$EA" \
  CLIENT_ARGS="--max-duration 45 --min-requests 300 --max-requests 1500 --warmup 64 --warmup-concurrency 32 --warmup-grid --stop-when-saturated --drain-timeout 240"
bash bench/e2e/jobs/single_adapter.sh q7 $D/q7_fine rankfuse-skip rankfuse stock-pad stock-full stock-r8
echo "=== done $(date +%T)"
