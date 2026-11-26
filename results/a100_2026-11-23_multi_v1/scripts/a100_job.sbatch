#!/bin/bash
# RankFuse Stage 2 on one A100 80GB PCIe (PACE ICE). Submit from the repository root:
#   sbatch --export=ALL,MODE=smoke results/a100_2026-11-23_multi_v1/scripts/a100_job.sbatch
#   sbatch --export=ALL,MODE=sweep,RMAX=64 results/a100_2026-11-23_multi_v1/scripts/a100_job.sbatch
#   sbatch --export=ALL,MODE=ncu results/a100_2026-11-23_multi_v1/scripts/a100_job.sbatch
# MODE smoke: device check, the multi-adapter / B4 test files, the S-LoRA extension built for sm_80.
# MODE sweep: bench/run_multi_sweep.py bench/configs/multi_a100_r${RMAX}.yaml -> raw/.
# MODE ncu:   scripts/ncu_all.sh (Nsight Compute bytes, r_max 64/128/256).
#SBATCH -J rf-a100
#SBATCH -p ice-gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH -C A100-80GB
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o results/a100_2026-11-23_multi_v1/source/logs/slurm-%x-%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
OUT=results/a100_2026-11-23_multi_v1
LOGS=$OUT/source/logs
# Per-session build dir, so the S-LoRA extension builds without clashing with other sessions. It is built for
# sm_80 and sm_90 so the smoke job (run on an H100/H200/A100-40GB to keep A100-80GB time for the sweeps) leaves a
# build the A100-80GB sweeps reuse.
export TORCH_EXTENSIONS_DIR=$HOME/scratch/rankfuse/cache/torch_extensions_a100_b60417aa
export TORCH_CUDA_ARCH_LIST="8.0;9.0"
source /storage/ice1/4/7/psunkad3/rankfuse/rf_paths.sh
source scripts/ice/env.sh

STAMP=$(date +%Y%m%d-%H%M%S)
TAG=${MODE}${RMAX:+-r$RMAX}-$SLURM_JOB_ID
{
  echo "job $SLURM_JOB_ID on $(hostname) at $(date -Is), MODE=$MODE RMAX=${RMAX:-}"
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.mem --format=csv
  python - <<'PY'
import torch, triton
print("torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__)
try:
    import vllm; print("vllm", vllm.__version__)
except Exception as e:
    print("vllm import failed:", e)
p = torch.cuda.get_device_properties(0)
print("device_name", repr(torch.cuda.get_device_name()), "sm", f"{p.major}.{p.minor}", "SMs", p.multi_processor_count,
      "mem_GiB", round(p.total_memory / 2**30, 1))
print("max_shared_mem (triton)", triton.runtime.driver.active.utils.get_device_properties(0)["max_shared_mem"])
PY
  echo "git HEAD $(git rev-parse --short HEAD) branch $(git branch --show-current)"
  echo "uncommitted diff (tracked) sha256 $(git diff HEAD | sha256sum | cut -c1-16)"
  echo "untracked:"; git status --short --untracked-files=all -- bench rankfuse tests | grep '^??' | while read -r _ f; do echo "  $f $(sha256sum "$f" | cut -c1-16)"; done
} 2>&1 | tee "$LOGS/env-$TAG.txt"
git diff HEAD > "$LOGS/diff-$TAG.patch"

NAME=$(python -c "import torch; print(torch.cuda.get_device_name())")
if [ "$NAME" != "NVIDIA A100 80GB PCIe" ]; then
  echo "device is '$NAME', not 'NVIDIA A100 80GB PCIe'"
  [ "$MODE" = smoke ] || exit 3
fi

case "$MODE" in
  smoke)
    python -c "from rankfuse.baselines import slora_bgmv as sb; import time; t=time.time(); sb._ext(); print(f'S-LoRA ext built/loaded in {time.time()-t:.0f}s')" 2>&1 | tee "$LOGS/build_ext-$TAG.out"
    rc=0
    for f in tests/test_multi_correctness.py tests/test_multi_graphs.py tests/test_slora_bgmv.py tests/test_multi_rmax.py; do
      b=$(basename "$f" .py)
      python -m pytest -q -rfEs --no-header -p no:cacheprovider --junitxml="$LOGS/$b-$TAG.xml" "$f" > "$LOGS/$b-$TAG.out" 2>&1
      [ $? = 0 ] || rc=1
      echo "== $f"; tail -n 5 "$LOGS/$b-$TAG.out"
    done
    python bench/run_multi_sweep.py bench/configs/multi_smoke.yaml --out "$OUT/source/smoke-$TAG.jsonl" 2>&1 | tee "$LOGS/sweep_smoke-$TAG.out"
    exit $rc
    ;;
  sweep)
    python bench/run_multi_sweep.py "bench/configs/multi_a100_r${RMAX}.yaml" ${IMPLS:+--impls "$IMPLS"} \
      --out "$OUT/raw/multi-a100-r${RMAX}.jsonl" 2>&1 | tee "$LOGS/sweep_r${RMAX}-$TAG.out"
    exit "${PIPESTATUS[0]}"
    ;;
  ncu)
    bash "$OUT/scripts/ncu_all.sh" 2>&1 | tee "$LOGS/ncu-$TAG.out"
    ;;
esac
