#!/bin/bash
# Short GPU job: environment check, correctness tests, and a smoke benchmark. No real sweep.
# Command-line flags override the #SBATCH lines below (e.g. --gres=gpu:A100:1 if H100s are busy).
# Output: results/slurm-test-<jobid>.out
#SBATCH -J rankfuse-test
#SBATCH --account=gts-smussmann3
#SBATCH -q embers
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -o results/slurm-test-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load anaconda3
module load cuda
eval "$(conda shell.bash hook)"
# V100 (sm_70) needs the CUDA 12.6 torch build; A100/H100 use the main (CUDA 13) env.
if [ -z "${ENV_PREFIX:-}" ]; then
    case "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" in
        *V100*) ENV_PREFIX=$HOME/scratch/envs/rankfuse-v100 ;;
        *)      ENV_PREFIX=$HOME/scratch/envs/rankfuse ;;
    esac
fi
conda activate "$ENV_PREFIX"
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$HOME/scratch/.triton}

nvidia-smi
python -c "import torch, triton, rankfuse; print('torch', torch.__version__, 'triton', triton.__version__, torch.cuda.get_device_name()); print(sorted(rankfuse.REGISTRY))"

# Fail fast if this torch build has no kernels for the GPU (e.g. V100 / sm_70 on recent torch).
python -c "import torch; torch.ones(1, device='cuda').sum().item(); print('cuda ok')"

echo "=== RankFuse kernels ==="
python -m pytest -q -rs -k "rf1 or rf2" ${PYTEST_ARGS:-}
echo "=== full suite ==="
python -m pytest -q -rs ${PYTEST_ARGS:-}
echo "=== smoke benchmark ==="
python bench/run_sweep.py bench/configs/smoke.yaml
