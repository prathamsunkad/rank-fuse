#!/bin/bash
# RankFuse in vLLM 0.32, Stage 1: launch-path benchmark, correctness gate, GPU-sync check and
# step-time measurements (results/h200_<date>_vllm/raw/). About 1 h on an H200.
#   sbatch scripts/ice/vllm_stage1.sbatch
# Needs the editable install (pip install --no-deps --no-build-isolation -e .) so vLLM finds the
# `vllm.general_plugins` entry point, and the adapters from scripts/make_adapters.py.
#SBATCH -J rankfuse-vllm-stage1
#SBATCH -p ice-gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o results/slurm-%x-%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/ice/env.sh
OUT=${OUT:-results/h200_$(date +%F)_vllm/raw}
mkdir -p "$OUT/correctness" "$OUT/step_time" "$OUT/sync_check"

# A stale rankfuse.egg-info in the repo root would hide the plugin's entry point.
python -c "from setuptools import setup; setup()" egg_info -q

python bench/vllm_rf1_launch.py "$OUT/vllm_rf1_launch.json"

for g in graphs eager; do
    RANKFUSE_VLLM=0 python bench/vllm_correctness.py run stock $g "$OUT/correctness/stock_$g.json" > "$OUT/correctness/stock_$g.log" 2>&1
    RANKFUSE_VLLM=1 python bench/vllm_correctness.py run rankfuse $g "$OUT/correctness/rankfuse_$g.json" > "$OUT/correctness/rankfuse_$g.log" 2>&1
done
python bench/vllm_correctness.py compare "$OUT/correctness" | tee "$OUT/correctness/compare.txt"

for f in 0 1; do
    for g in graphs eager; do
        VLLM_GPU_SYNC_CHECK=error RANKFUSE_VLLM=$f python results/h200_2026-11-03_vllm/scripts/sync_check.py $g \
            > "$OUT/sync_check/rankfuse${f}_$g.log" 2>&1
        grep -h "SYNC CHECK" "$OUT/sync_check/rankfuse${f}_$g.log" || echo "sync check failed: RANKFUSE_VLLM=$f $g"
    done
done

for g in graphs eager; do
    for m in base stock64 stockmix rfmix rf64; do
        [ "${m:0:2}" = rf ] && f=1 || f=0
        RANKFUSE_VLLM=$f python bench/vllm_step_time_rankfuse.py $m $g "$OUT/step_time/step_time.jsonl" \
            > "$OUT/step_time/${m}_$g.log" 2>&1 || echo "step time failed: $m $g"
    done
done
python bench/vllm_step_time_rankfuse.py summary "$OUT/step_time/step_time.jsonl" "$OUT/step_time/summary.md"
