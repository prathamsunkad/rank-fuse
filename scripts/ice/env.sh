# Shared environment for every PACE ICE job and setup step. Source it, don't run it:
#   source scripts/ice/env.sh
#
# Everything RankFuse needs lives in one folder in scratch (home quotas on ICE are small), next
# to nothing else, so it is easy to find and to delete:
#
#   ~/scratch/rankfuse/
#     rank-fuse/      this repository (clone it here)
#     env/            conda environment
#     hf/             model downloads
#     adapters/       generated Matryoshka adapters
#     cache/          pip, conda and Triton caches
#
# Override the folder (or any single path) before sourcing, e.g. `export RF_ROOT=/path`.

export RF_ROOT=${RF_ROOT:-$HOME/scratch/rankfuse}
export RF_ENV=${RF_ENV:-$RF_ROOT/env}                        # conda env prefix
export HF_HOME=${HF_HOME:-$RF_ROOT/hf}                       # model downloads
export RF_ADAPTERS=${RF_ADAPTERS:-$RF_ROOT/adapters}         # generated Matryoshka adapters
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$RF_ROOT/cache/triton}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$RF_ROOT/cache/pip}
export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-$RF_ROOT/cache/conda-pkgs}   # default is ~/.conda/pkgs (home quota)
export PYTHONNOUSERSITE=1   # ignore packages pip-installed with --user into ~/.local for other projects

module load anaconda3
eval "$(conda shell.bash hook)"
if [ -d "$RF_ENV" ]; then
    conda activate "$RF_ENV"
fi
