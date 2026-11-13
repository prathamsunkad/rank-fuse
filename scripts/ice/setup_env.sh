#!/bin/bash
# One-time environment setup on PACE ICE. Clone the repo into ~/scratch/rankfuse/rank-fuse
# (see scripts/ice/env.sh for the folder layout), then run from the repo root on a login node:
#   bash scripts/ice/setup_env.sh
#
# Installs the newest stable vLLM, which brings its own PyTorch, Triton and CUDA libraries,
# then this repo on top. Afterwards, pin what was installed:
#   source scripts/ice/env.sh && pip freeze > requirements-lock.txt
# and verify on a GPU node with: sbatch scripts/ice/check.sbatch
set -euo pipefail

source scripts/ice/env.sh
mkdir -p "$HF_HOME" "$RF_ADAPTERS" "$TRITON_CACHE_DIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$(dirname "$RF_ENV")"

if [ ! -d "$RF_ENV" ]; then
    conda create -y -p "$RF_ENV" python=3.11
fi
conda activate "$RF_ENV"

pip install --upgrade pip
pip install vllm                      # newest stable release; pinned afterwards via requirements-lock.txt
pip install -e ".[analysis,dev]"      # not [gpu]: vLLM already installed a matching Triton
pip install safetensors               # usually already present with vLLM

python -c "import torch, triton, importlib.util; print('torch', torch.__version__, 'cuda', torch.version.cuda, \
'triton', triton.__version__, 'vllm_spec', bool(importlib.util.find_spec('vllm')))"
