#!/bin/bash
# One-time environment setup on PACE Phoenix. Run from the repo root on a login node:
#   bash scripts/phoenix/env_setup.sh
# Check `module avail anaconda3 cuda` if a module name below has changed.
#
# The home quota is too small for a torch + CUDA env, so the env and the conda/pip caches go
# under ENV_ROOT (default ~/scratch). Scratch is purged of files unused for 60 days; for a
# permanent env, point ENV_ROOT at project storage, e.g. ENV_ROOT=~/p-<pi>-0/<user>.
set -euo pipefail

module load anaconda3
module load cuda
eval "$(conda shell.bash hook)"

ENV_ROOT=${ENV_ROOT:-$HOME/scratch}
ENV_PREFIX=${ENV_PREFIX:-$ENV_ROOT/envs/rankfuse}
export CONDA_PKGS_DIRS=$ENV_ROOT/.conda-pkgs
export PIP_CACHE_DIR=$ENV_ROOT/.pip-cache
# Unpacking multi-GB torch wheels in /tmp or ~/.cache also hits the login-node/home quota.
export TMPDIR=$ENV_ROOT/.tmp
export XDG_CACHE_HOME=$ENV_ROOT/.cache
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR" "$XDG_CACHE_HOME"

if [ ! -d "$ENV_PREFIX" ]; then
    # conda records every env in $HOME/.conda/environments.txt; if ~/.conda is a link to a full
    # project volume that write fails with "Disk quota exceeded", so point HOME at ENV_ROOT here.
    HOME="$ENV_ROOT" conda env create -p "$ENV_PREFIX" -f environment.yml
fi
conda activate "$ENV_PREFIX"
pip install -e ".[gpu,analysis,dev]"

# Default PyPI torch is built for CUDA 13, which has no V100 (sm_70) kernels. For a V100 env:
#   ENV_PREFIX=~/scratch/envs/rankfuse-v100 TORCH_INDEX=https://download.pytorch.org/whl/cu126 \
#       bash scripts/phoenix/env_setup.sh
if [ -n "${TORCH_INDEX:-}" ]; then
    pip install --force-reinstall torch --index-url "$TORCH_INDEX"
fi

python -c "import torch, triton; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'triton', triton.__version__)"
echo "Activate with: conda activate $ENV_PREFIX"
