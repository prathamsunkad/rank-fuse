#!/bin/bash
# First job on a new GPU type: report the node, check versions and internet access, run tests.
#   sbatch scripts/ice/check.sbatch                          # H200 (headline GPU)
#   sbatch --gres=gpu:h100:1 scripts/ice/check.sbatch        # fallback order: h100, then a100
#
# If ICE requires a charge account or QOS for your course, add `#SBATCH -A ...` / `#SBATCH -q ...`
# here and in the other .sbatch files (`sacctmgr show assoc user=$USER format=account,qos%40`).
#SBATCH -J rankfuse-check
#SBATCH -p ice-gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --mem=64G
#SBATCH -t 00:45:00
#SBATCH -o results/slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/ice/env.sh

nvidia-smi
python -c "import torch, triton, importlib.util; v=importlib.util.find_spec('vllm'); print('torch', torch.__version__, 'cuda', torch.version.cuda, \
'triton', triton.__version__, 'vllm', 'available' if v else 'not-installed'); print(torch.cuda.get_device_name(), \
torch.cuda.get_device_properties(0).multi_processor_count, 'SMs')"
mkdir -p results/h200_ice_$(date +%F)
python -c "import torch, triton; from pathlib import Path; p=Path('results/h200_ice_' + __import__('datetime').date.today().isoformat()) / 'env.txt'; p.write_text('\\n'.join([f'torch={torch.__version__}', f'triton={triton.__version__}', f'cuda={torch.version.cuda}', f'device={torch.cuda.get_device_name()}', f'sm={torch.cuda.get_device_properties(0).multi_processor_count}']))"
module load cuda
ncu --version
python -m pytest -q tests/test_correctness.py tests/test_kernel_features.py
if curl -sI --max-time 10 https://huggingface.co >/dev/null; then
    echo "internet: compute node can reach huggingface.co"
else
    echo "internet: NO access from compute node; download models on the login node instead"
fi
