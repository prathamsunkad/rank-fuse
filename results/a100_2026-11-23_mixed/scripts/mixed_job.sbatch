#!/bin/bash
# RankFuse mixed-batching (chunked-prefill) sweeps. Submit from the wt/mixed worktree root:
#   sbatch --export=ALL,MODE=smoke  results/a100_2026-11-23_mixed/scripts/mixed_job.sbatch
#   sbatch --export=ALL,MODE=single,RMAX=128 results/a100_2026-11-23_mixed/scripts/mixed_job.sbatch
#   sbatch --export=ALL,MODE=multi,RMAX=128  results/a100_2026-11-23_mixed/scripts/mixed_job.sbatch
# A100 80GB PCIe by default (the #SBATCH lines); for a dev run on another GPU, override on the command
# line (-p ice-gpu --gres=gpu:h100:1 -C "" ...) and set DEV=1 (outputs go to dev/, never raw/).
# TESTS=1 (single/multi): run the new tests on this GPU before the sweep.
# MODE smoke:  env, new tests (tests/test_mixed_*.py), both smoke sweeps.
# MODE single: bench/run_mixed_sweep.py bench/configs/rf2_mixed_a100_r$RMAX.yaml (--resume)
# MODE multi:  bench/run_multi_mixed_sweep.py bench/configs/multi_mixed_a100_r$RMAX.yaml (--resume)
#              TAG_SUFFIX names the output (e.g. -rf1ma_v2 after kernels-v2's Phase B step 1); IMPLS a subset.
#SBATCH -J rf-mixed
#SBATCH -p ice-gpu,coc-gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH -C A100-80GB
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o results/a100_2026-11-23_mixed/source/logs/slurm-%x-%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
OUT=results/a100_2026-11-23_mixed
LOGS=$OUT/source/logs
source /storage/ice1/4/7/psunkad3/rankfuse/rf_paths.sh
source scripts/ice/env.sh
export PYTHONPATH=$PWD
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
export TORCH_CUDA_ARCH_LIST=$ARCH
export TORCH_EXTENSIONS_DIR=$HOME/scratch/rankfuse/torch_ext/mixed_sm${ARCH/./}
DEST=$OUT/raw
[ "${DEV:-0}" = 1 ] && DEST=$OUT/dev
mkdir -p "$DEST" "$LOGS"

TAG=${MODE}${RMAX:+-r$RMAX}${TAG_SUFFIX:-}-$SLURM_JOB_ID
{
  echo "job $SLURM_JOB_ID on $(hostname) at $(date -Is), MODE=$MODE RMAX=${RMAX:-} DEV=${DEV:-0} IMPLS=${IMPLS:-}"
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.mem --format=csv
  python - <<'PY'
import torch, triton, rankfuse
from rankfuse.kernels import rankfuse_fused as rf
print("rankfuse", rankfuse.__file__)
print("torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__)
try:
    import vllm; print("vllm", vllm.__version__)
except Exception as e:
    print("vllm import failed:", e)
p = torch.cuda.get_device_properties(0)
print("device_name", repr(torch.cuda.get_device_name()), "sm", f"{p.major}.{p.minor}", "SMs", p.multi_processor_count,
      "mem_GiB", round(p.total_memory / 2**30, 1))
from rankfuse.kernels import tuned_a100
import os
print("tuned_a100 table in force:", torch.cuda.get_device_name() == tuned_a100.GPU and os.environ.get("RANKFUSE_TUNED", "1") != "0")
PY
  echo "git HEAD $(git rev-parse --short HEAD) ($(git log -1 --format=%s | cut -c1-80))"
  echo "uncommitted diff (tracked) sha256 $(git diff HEAD | sha256sum | cut -c1-16)"
  echo "untracked:"; git status --short --untracked-files=all -- bench rankfuse tests | grep '^??' | while read -r _ f; do echo "  $f $(sha256sum "$f" | cut -c1-16)"; done
} 2>&1 | tee "$LOGS/env-$TAG.txt"
git diff HEAD > "$LOGS/diff-$TAG.patch"

NAME=$(python -c "import torch; print(torch.cuda.get_device_name())")
if [ "$NAME" != "NVIDIA A100 80GB PCIe" ] && [ "${DEV:-0}" != 1 ]; then
  echo "device is '$NAME', not 'NVIDIA A100 80GB PCIe' (set DEV=1 for a dev run)"; exit 3
fi
python -c "from rankfuse.baselines import slora_bgmv_wide, slora_bgmv as sb; slora_bgmv_wide.enable(); import time; t=time.time(); sb._ext(); print(f'S-LoRA ext built/loaded in {time.time()-t:.0f}s')" 2>&1 | tee "$LOGS/build_ext-$TAG.out"

run_tests() {
  local rc=0
  for f in tests/test_mixed_single.py tests/test_mixed_multi.py tests/test_workloads.py; do
    b=$(basename "$f" .py)
    python -m pytest -q -rfEs --no-header -p no:cacheprovider --junitxml="$LOGS/$b-$TAG.xml" "$f" > "$LOGS/$b-$TAG.out" 2>&1
    [ $? = 0 ] || rc=1
    echo "== $f"; tail -n 5 "$LOGS/$b-$TAG.out"
  done
  return $rc
}
# TESTS=1: the new tests first (correctness on this GPU), then the sweep.
[ "${TESTS:-0}" = 1 ] && [ "$MODE" != smoke ] && { run_tests 2>&1 | tee "$LOGS/tests-$TAG.out"; }

case "$MODE" in
  smoke)
    run_tests; rc=$?
    python bench/run_mixed_sweep.py bench/configs/rf2_mixed_a100_smoke.yaml --out "$DEST/single-smoke-$TAG.csv" 2>&1 | tee "$LOGS/single_smoke-$TAG.out"
    python bench/run_multi_mixed_sweep.py bench/configs/multi_mixed_smoke.yaml --out "$DEST/multi-smoke-$TAG.jsonl" 2>&1 | tee "$LOGS/multi_smoke-$TAG.out"
    exit $rc
    ;;
  single)
    python bench/run_mixed_sweep.py "bench/configs/rf2_mixed_a100_r${RMAX}.yaml" --resume \
      --out "$DEST/single-mixed-r${RMAX}.csv" 2>&1 | tee "$LOGS/single_r${RMAX}-$TAG.out"
    exit "${PIPESTATUS[0]}"
    ;;
  multi)
    python bench/run_multi_mixed_sweep.py "bench/configs/multi_mixed_a100_r${RMAX}.yaml" --resume ${IMPLS:+--impls "$IMPLS"} \
      --out "$DEST/multi-mixed-r${RMAX}${TAG_SUFFIX:-}.jsonl" 2>&1 | tee "$LOGS/multi_r${RMAX}-$TAG.out"
    exit "${PIPESTATUS[0]}"
    ;;
esac
