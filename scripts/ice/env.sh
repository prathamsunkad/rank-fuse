# Shared environment for every PACE ICE job and setup step. Source it, don't run it:
#   source scripts/ice/env.sh
#
# Large files go to scratch: home quotas on ICE are small. Override any path before sourcing,
# e.g. `export RF_SCRATCH=/path/to/scratch`.

RF_SCRATCH=${RF_SCRATCH:-$HOME/scratch}
export RF_ENV=${RF_ENV:-$RF_SCRATCH/envs/rankfuse}         # conda env prefix
export HF_HOME=${HF_HOME:-$RF_SCRATCH/hf}                    # model downloads
export RF_ADAPTERS=${RF_ADAPTERS:-$RF_SCRATCH/adapters}      # generated Matryoshka adapters
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$RF_SCRATCH/triton-cache}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$RF_SCRATCH/pip-cache}

module load anaconda3
eval "$(conda shell.bash hook)"
if [ -d "$RF_ENV" ]; then
    conda activate "$RF_ENV"
fi
