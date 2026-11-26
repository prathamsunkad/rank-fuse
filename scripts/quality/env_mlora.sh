# Environment for the quality-vs-rank study. Source it (login or compute node).
# Separate conda env (torch 2.8 / transformers 4.57.6 / lm-eval 0.4.9.1 / vLLM 0.11.0); the shared
# torch 2.13 / vLLM 0.32 env in $R/env is not touched.
R=/storage/ice1/4/7/psunkad3/rankfuse
export RQ=$R
export HF_HOME=$R/hf_cache                 # new downloads (models + datasets) for this study
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1
export PIP_CACHE_DIR=$R/cache/pip CONDA_PKGS_DIRS=$R/cache/conda-pkgs
export TRITON_CACHE_DIR=$R/cache/triton_mlora VLLM_CACHE_ROOT=$R/cache/vllm_mlora VLLM_CONFIG_ROOT=$R/cache/vllm_mlora_cfg
export TORCHINDUCTOR_CACHE_DIR=$R/cache/inductor_mlora FLASHINFER_WORKSPACE_BASE=$R/cache/flashinfer_mlora
export TOKENIZERS_PARALLELISM=false
module load anaconda3 >/dev/null 2>&1
eval "$(conda shell.bash hook)"
conda activate $R/envs/mlora
# model paths (0.5B reuses the copy an earlier stage downloaded)
export QWEN05=$(ls -d $R/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*/ | head -1)
export QWEN7=$(ls -d $R/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/ 2>/dev/null | head -1)   # read-only copy from the e2e-harness session
