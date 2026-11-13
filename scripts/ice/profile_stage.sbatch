#!/bin/bash
#SBATCH -J rankfuse-profile
#SBATCH -p ice-gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH -o results/slurm-%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/ice/env.sh
DATE=${PROFILE_DATE:?PROFILE_DATE is required}
ROOT="results/h200_ice_${DATE}"
mkdir -p "$ROOT"/{tables,figures,raw,ncu}
STAGE=${STAGE:?STAGE is required}
printf '%s\n' "[$(date -Is)] START $STAGE job=$SLURM_JOB_ID" >> "$ROOT/STATUS.md"
trap 'rc=$?; printf "%s\n" "[$(date -Is)] FAILED $STAGE job=$SLURM_JOB_ID exit=$rc; inspect results/slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out and resume with the exact dependency chain in STATUS.md" >> "$ROOT/STATUS.md"; exit "$rc"' EXIT
case "$STAGE" in
  check)
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    python -c "import torch,triton,importlib.util; print('torch',torch.__version__,'cuda',torch.version.cuda,'triton',triton.__version__,'vllm',bool(importlib.util.find_spec('vllm'))); print(torch.cuda.get_device_name(),torch.cuda.get_device_properties(0).multi_processor_count,'SMs')"
    module load cuda
    ncu --version
    python -m pytest -q tests/test_correctness.py tests/test_kernel_features.py
    python -c "import torch,triton,importlib.util; from pathlib import Path; p=Path('$ROOT/env.txt'); p.write_text('\\n'.join([f'torch={torch.__version__}',f'cuda={torch.version.cuda}',f'triton={triton.__version__}',f'vllm={bool(importlib.util.find_spec(\"vllm\"))}',f'device={torch.cuda.get_device_name()}',f'sm={torch.cuda.get_device_properties(0).multi_processor_count}']))"
    ;;
  latency)
    python bench/profile_components.py --out "$ROOT/raw/latency_components.csv"
    ;;
  ablation)
    python bench/profile_components.py --out "$ROOT/raw/ablation_components.csv" --ablations
    ;;
  ncu_full)
    for impl in rf1_fused rf2_splitk; do
      bash scripts/ice/ncu_profile.sh "$impl" 32 1 4096 4096 64 uniform fp16
    done
    cp results/ncu/*.csv "$ROOT/ncu/" 2>/dev/null || true
    ;;
  ncu_noflush)
    for impl in rf1_fused rf2_splitk; do
      OUT="$ROOT/ncu/${impl}-noflush"
      module load cuda
      ncu --profile-from-start off --set full --cache-control none --clock-control none -o "$OUT" -f python bench/ncu_driver.py "$impl" 32 1 4096 4096 64 uniform fp16
      if [ -f "${OUT}.ncu-repz" ]; then zstd -q -d -f "${OUT}.ncu-repz" -o "${OUT}.ncu-rep"; fi
      ncu --import "${OUT}.ncu-rep" --page raw --csv > "${OUT}.csv"
    done
    ;;
  *)
    echo "unknown STAGE=$STAGE" >&2
    exit 2
    ;;
esac
printf '%s\n' "[$(date -Is)] DONE $STAGE job=$SLURM_JOB_ID" >> "$ROOT/STATUS.md"
trap - EXIT
