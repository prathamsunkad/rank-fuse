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
#     cache/          pip, conda, Triton, vLLM and kernel-library caches
#
# Override the folder (or any single path) before sourcing, e.g. `export RF_ROOT=/path`.

export RF_ROOT=${RF_ROOT:-$HOME/scratch/rankfuse}
export RF_ENV=${RF_ENV:-$RF_ROOT/env}                        # conda env prefix
export HF_HOME=${HF_HOME:-$RF_ROOT/hf}                       # model downloads
export RF_ADAPTERS=${RF_ADAPTERS:-$RF_ROOT/adapters}         # generated Matryoshka adapters
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$RF_ROOT/cache/triton}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$RF_ROOT/cache/torch_extensions}   # JIT-built C++/CUDA extensions (B4)
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$RF_ROOT/cache/pip}
export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-$RF_ROOT/cache/conda-pkgs}   # default is ~/.conda/pkgs (home quota)
export PYTHONNOUSERSITE=1   # ignore packages pip-installed with --user into ~/.local for other projects
# vLLM and the kernel libraries it brings write to ~/.cache, ~/.config, ~/.tilelang and ~/.humming
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$RF_ROOT/cache/vllm}
export VLLM_CONFIG_ROOT=${VLLM_CONFIG_ROOT:-$RF_ROOT/cache/vllm-config}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$RF_ROOT/cache/flashinfer}   # FlashInfer appends .cache/flashinfer
export TILELANG_CACHE_DIR=${TILELANG_CACHE_DIR:-$RF_ROOT/cache/tilelang}
export HUMMING_CACHE_DIR=${HUMMING_CACHE_DIR:-$RF_ROOT/cache/humming}
export HUMMING_TMP_DIR=${HUMMING_TMP_DIR:-$RF_ROOT/cache/humming/tmp}
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1   # no usage reports from cluster nodes

module load anaconda3
eval "$(conda shell.bash hook)"
if [ -d "$RF_ENV" ]; then
    conda activate "$RF_ENV"
fi
