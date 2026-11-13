#!/bin/bash
# One-time environment setup on PACE Phoenix. Run from the repo root on a login node:
#   bash scripts/phoenix/env_setup.sh
# Check `module avail anaconda3 cuda` if a module name below has changed.
set -euo pipefail

module load anaconda3
module load cuda
eval "$(conda shell.bash hook)"

ENV_NAME=${ENV_NAME:-rankfuse}
if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda env create -n "$ENV_NAME" -f environment.yml
fi
conda activate "$ENV_NAME"
pip install -e ".[gpu,analysis,dev]"

python -c "import torch, triton; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'triton', triton.__version__)"
