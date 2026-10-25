# rank-fuse
Triton-based kernel for efficient heterogeneous-rank MatryoshkaLoRA serving.

Design, claims, baselines and milestones: **[docs/PLAN.md](docs/PLAN.md)**; the Matryoshka-LoRA
serving plan and its status: **[docs/PLAN_MATLORA.md](docs/PLAN_MATLORA.md)**; how the kernels work,
with measured results: **[docs/ALGORITHMS.md](docs/ALGORITHMS.md)**.
Every measured result so far (tables as Markdown and CSV, raw data, rebuild script):
**[results/h200_2026-10-25/](results/h200_2026-10-25/README.md)**.

## Layout

```
rankfuse/
  interface.py        shared contract: MatryoshkaLoRA, Batch, @register, REGISTRY
  reference.py        B0 reference (correctness oracle)
  baselines/          B1 padded+mask (+ graphed), B2 grouped (+ graphed), B3 vLLM's kernels
                      (vllm_ops.py); B4 stub with a plan
  kernels/            RF1 (rankfuse_fused.py: split-K rf1_fused, rf1_single) and RF2
                      (rf2_fused; rankfuse_rf2_splitk/segment/bucketed.py) Triton kernels
  workloads.py        seeded shapes and rank distributions
  roofline.py         analytic bytes/FLOPs/lower-bound per strategy
tests/                correctness vs B0 for every registered impl; kernel serving features; roofline
bench/run_sweep.py    one-layer benchmark -> CSV (eager or CUDA-graph timing); bench/configs/*.yaml
bench/model_harness.py, model_forward.py   whole Qwen2.5-0.5B with each LoRA backend: check, timing
bench/vllm_step_time.py                    LoRA share of a stock vLLM step
bench/ncu_driver.py   one profiled call for Nsight Compute
analysis/plot.py      latency vs tokens, measured vs roofline
scripts/ice/          PACE ICE: env, first-run check, sweep, ncu profiling, vLLM adapter check
scripts/make_adapters.py                   random Matryoshka adapters in PEFT format
scripts/phoenix/      older PACE Phoenix scripts
```

## Local development (CPU is fine)

```bash
pip install -e ".[dev,analysis]"
pytest -q                                          # GPU-only cases are skipped
python -m rankfuse.roofline --ranks 4 8 16 32      # predicted costs per strategy
python bench/run_sweep.py bench/configs/smoke.yaml # harness check, writes results/raw/*.csv
```

## PACE ICE

Everything lives in one folder in scratch (`~/scratch/rankfuse`: this repo, the conda env, models,
adapters, caches); `scripts/ice/env.sh` sets every path and is sourced by every job.

```bash
git clone git@github.com:prathamsunkad/rank-fuse.git ~/scratch/rankfuse/rank-fuse
cd ~/scratch/rankfuse/rank-fuse
bash scripts/ice/setup_env.sh                      # once, on a login node (vLLM brings torch/triton)
sbatch scripts/ice/check.sbatch                    # GPU, versions, internet, pytest on an H200
sbatch scripts/ice/sweep.sbatch                    # tests + Llama-3.1-8B decode sweep, graph timing
sbatch --export=ALL,CONFIG=bench/configs/llama3_8b_prefill.yaml scripts/ice/sweep.sbatch
python analysis/plot.py results/raw/*.csv --out results/decode.png
```

Inside a GPU job (`salloc -p ice-gpu --gres=gpu:h200:1 -t 1:00:00 --cpus-per-task=8 --mem=64G`,
then `source scripts/ice/env.sh`):

```bash
python bench/run_sweep.py bench/configs/decode.yaml --timing graph   # GPU time only
bash scripts/ice/ncu_profile.sh rf1_fused 32 1                       # Nsight Compute, last call only
python scripts/make_adapters.py --model qwen2.5-0.5b --r-max 64 --out $RF_ADAPTERS/qwen2.5-0.5b
python bench/model_harness.py check                                  # whole model, every backend vs fp32
python bench/model_forward.py                                        # whole-model timing
python bench/vllm_step_time.py lora64 graphs results/raw/vllm_step.jsonl   # one engine per process
```

`--timing eager` (the default without the flag) includes the Python launch overhead of each call,
about 70-110 us on ICE nodes, which hides kernel differences at decode sizes; `--timing graph`
captures each call in a CUDA graph and reports GPU time only.

## Adding an implementation

```python
from rankfuse.interface import register, apply_scale

@register("rf1_masked", requires_cuda=True)
def rf1_masked(x, lora, token_ranks):   # x [T, h_in], lora.A_t [r_max, h_in], lora.B [r_max, h_out]
    ...
    return apply_scale(y, lora, token_ranks)   # y [T, h_out], LoRA delta only
```

Import the module from its package `__init__.py`. It is then tested against B0 and benchmarked automatically.
`register` also takes `dtypes=` (supported input dtypes; tests skip the others) and `prepare=`, a
function `(x, lora, token_ranks) -> kwargs` returning the per-step metadata a server would build
once (batch maximum rank, rank order, ...); passing it removes the call's host syncs, which lets
`run_sweep.py --timing graph` capture the call.
