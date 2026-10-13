# rank-fuse
Triton-based kernel for efficient heterogeneous-rank MatryoshkaLoRA serving.

Design, claims, baselines and milestones: **[docs/PLAN.md](docs/PLAN.md)**.

## Layout

```
rankfuse/
  interface.py        shared contract: MatryoshkaLoRA, Batch, @register, REGISTRY
  reference.py        B0 reference (correctness oracle)
  baselines/          B1 padded+mask, B2 grouped (+ CUDA-graph variant); B3/B4 stubs with plans
  kernels/            RF1 fused-masked and RF2 rank-sorted Triton kernels
  workloads.py        seeded shapes and rank distributions
  roofline.py         analytic bytes/FLOPs/lower-bound per strategy
tests/                correctness vs B0 for every registered impl; roofline checks
bench/run_sweep.py    benchmark harness -> CSV;  bench/configs/*.yaml
analysis/plot.py      latency vs tokens, measured vs roofline
scripts/phoenix/      env setup, sbatch sweep, ncu profiling
```

## Local development (CPU is fine)

```bash
pip install -e ".[dev,analysis]"
pytest -q                                          # GPU-only cases are skipped
python -m rankfuse.roofline --ranks 4 8 16 32      # predicted costs per strategy
python bench/run_sweep.py bench/configs/smoke.yaml # harness check, writes results/raw/*.csv
```

## PACE Phoenix

```bash
bash scripts/phoenix/env_setup.sh                  # once, on a login node
# edit -A / -q / --gres in scripts/phoenix/sweep.sbatch first
sbatch scripts/phoenix/sweep.sbatch                # tests + decode sweep
sbatch --export=ALL,CONFIG=bench/configs/prefill.yaml scripts/phoenix/sweep.sbatch
python analysis/plot.py results/raw/decode-*.csv --out results/decode.png
bash scripts/phoenix/ncu_profile.sh b1_padded 32 1 # inside a GPU job
```

## Adding an implementation

```python
from rankfuse.interface import register, apply_scale

@register("rf1_masked", requires_cuda=True)
def rf1_masked(x, lora, token_ranks):   # x [T, h_in], lora.A_t [r_max, h_in], lora.B [r_max, h_out]
    ...
    return apply_scale(y, lora, token_ranks)   # y [T, h_out], LoRA delta only
```

Import the module from its package `__init__.py`. It is then tested against B0 and benchmarked automatically.
