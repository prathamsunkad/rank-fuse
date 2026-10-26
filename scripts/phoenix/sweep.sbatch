#!/bin/bash
# Run correctness tests on the GPU, then a benchmark sweep.
#   sbatch scripts/phoenix/sweep.sbatch                      # decode sweep
#   sbatch --export=ALL,CONFIG=bench/configs/prefill.yaml scripts/phoenix/sweep.sbatch
#
# EDIT BEFORE FIRST USE:
#   -A : your PI's charge account (`pace-quota` lists yours), e.g. gts-<pi-username>
#   -q : inferno (charged, default here) or embers (free, preemptible backfill)
#   --gres : A100, the headline GPU on Phoenix. Report headline numbers on ONE GPU type.
#            Avoid V100 for timing: Triton has no tensor-core path for sm_70, so RF1/RF2 run
#            their dots as plain FMAs there (correctness tests are still fine).
#
# The env is picked from the GPU name (V100 -> rankfuse-v100, else rankfuse); ENV_PREFIX overrides.
# Long sweeps need a longer limit, e.g. the systematic ones:
#   sbatch -t 04:00:00 --export=ALL,CONFIG=bench/configs/sys_decode.yaml scripts/phoenix/sweep.sbatch
# Free backfill instead of charged time:
#   sbatch -q embers scripts/phoenix/sweep.sbatch
#SBATCH -J rankfuse-sweep
#SBATCH -A gts-smussmann3
#SBATCH -q inferno
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o results/slurm-%j.out

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

CONFIG=${CONFIG:-bench/configs/decode.yaml}
nvidia-smi
# Fail fast if this torch build has no kernels for the GPU (e.g. CUDA 13 torch on a V100).
python -c "import torch; torch.ones(1, device='cuda').sum().item(); print('cuda ok', torch.__version__, torch.cuda.get_device_name())"
python -m pytest -q -rs
# CONFIG may list several configs separated by spaces; each writes its own CSV.
for c in $CONFIG; do
    python bench/run_sweep.py "$c"
done
