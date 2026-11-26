#!/bin/bash
# GPU checks of the rank-skip plugin mode, inside an allocation:
#   srun --jobid=<id> --overlap -N1 -n1 bash bench/e2e/jobs/rank_skip_checks.sh <out>
# 1. kernel tests (tests/test_rank_skip_launcher.py) and the CPU harness tests
# 2. graph-replay microbenchmark, 7B and 0.5B layer shapes
# 3. the Stage 1 P4.6 correctness gate (bench/vllm_correctness.py, r_max 64 adapters) with
#    RANKFUSE_VLLM_KERNEL=rf2_index and rf2_segment against stock, CUDA graphs on and off
set -uo pipefail
WT=/storage/ice1/4/7/psunkad3/rankfuse/wt/e2e
cd "$WT"
source /storage/ice1/4/7/psunkad3/rankfuse/rf_paths.sh
source scripts/ice/env.sh
export PYTHONPATH=$WT TORCH_EXTENSIONS_DIR=$RF_SCRATCH/cache/torch_extensions_e2e VLLM_CACHE_ROOT=$RF_SCRATCH/cache/vllm_e2e
OUT=$1; mkdir -p "$OUT/correctness"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$OUT/gpu.txt"
python -m pytest -q -p no:cacheprovider tests/test_rank_skip_launcher.py tests/test_e2e_harness.py 2>&1 | tail -5 | tee "$OUT/pytest.txt"
for m in q7 q05; do python bench/e2e/rank_skip_microbench.py "$OUT/microbench_$m.jsonl" --model $m > "$OUT/microbench_$m.log" 2>&1; tail -45 "$OUT/microbench_$m.log"; done
C=$OUT/correctness
for g in graphs eager; do
  RANKFUSE_VLLM=0 python bench/vllm_correctness.py run stock $g $C/stock_$g.json > $C/stock_$g.log 2>&1
  for k in rf2_index rf2_segment; do
    mkdir -p $C/$k
    RANKFUSE_VLLM=1 RANKFUSE_VLLM_KERNEL=$k python bench/e2e/with_rank_skip.py bench/vllm_correctness.py run rankfuse $g $C/$k/rankfuse_$g.json > $C/$k/rankfuse_$g.log 2>&1
    grep -h "RankFuse rank skipping" $C/$k/rankfuse_$g.log | head -1
    cp $C/stock_$g.json $C/$k/
  done
done
for k in rf2_index rf2_segment; do echo "== $k"; python bench/vllm_correctness.py compare $C/$k | tee $C/$k/compare.txt | tail -25; done
