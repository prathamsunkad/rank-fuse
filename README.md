# rank-fuse
Triton-based kernel for efficient heterogeneous-rank MatryoshkaLoRA serving.

The Matryoshka-LoRA serving plan and its status: **[docs/PLAN_MATLORA.md](docs/PLAN_MATLORA.md)**;
how the kernels work, with measured results: **[docs/ALGORITHMS.md](docs/ALGORITHMS.md)**.
Measured results: **[H200, 25 Oct](results/h200_2026-10-25/README.md)** (tables as Markdown and CSV,
raw data, rebuild script) and **[A100, 26 Oct](results/a100_2026-10-26/ALL_ROWS.md)**.

## Project framing

RankFuse makes two separate claims, and each has its own comparison:

1. **Use the nesting instead of registering every rank as an adapter.** vLLM treats each rank of a
   Matryoshka adapter as a separate adapter, with its own copy and LoRA slot (seven per adapter). One
   shared copy removes that. No mixed-rank serving implementation exists to compare against, since
   the paper's implementation only slices the matrix to the requested rank, so vLLM's approach is the
   naive baseline. Most of this win is already in the PyTorch baseline (B1), not in a custom kernel.
2. **Serve low-rank requests with less work.** RF2 reads fewer rank rows for blocks of low-rank
   tokens. This is the harder claim: it only pays off on some workloads.

### Implementations

| Implementation | What it is |
|---|---|
| **B3** | vLLM's kernels, with each rank as a separate padded adapter |
| **B1** | PyTorch with one shared copy, padded to the largest rank and masked; our construction, and the simplest mixed-rank design |
| **B2** | PyTorch grouped by rank: the paper's slicing extended to mixed batches |
| **RF1** (`rf1_fused`) | One shared copy in Triton, with every token block at the batch's rank width |
| **RF2** (`rf2_segment`) | RF1 plus rank skipping: each token block runs at the smallest rank bucket covering its own maximum rank |
| **RF2, sorted index** (`rf2_index`) | The same per-block widths, reading tokens through a rank-sorted index list instead of needing them physically sorted |

Split-K splits the long `h_in` sum across concurrent programs. It is a general Triton optimization
in every RF1 and RF2 variant, and it does not skip ranks.

RF1 is not a clean measure of fusion: one-kernel fusion alone lost to B1 at decode, so `rf1_fused`
also has split-K and other Triton optimizations. `rf1_single` is the one-kernel version. RF1 and RF2
use general optimizations; beyond one shared copy and reading a prefix of its rows, we have not found
another kernel advantage specific to the Matryoshka structure.

### Comparisons

| Comparison | What it shows |
|---|---|
| **B3 vs B1** | What using the Matryoshka structure gains over vLLM's approach |
| **RF2 vs RF1** | What rank skipping gains: the main low-rank result |
| **RF1 vs B1** | Whether RF2 is measured against a competitive kernel, not whether rank skipping helps; RF1 can lose here (0.80× of B1's speed on the A100 at r_max 128) |
| **RF2 vs B1** | Whether the result beats the simple approach; report it next to RF2 vs RF1 |

Quote the vLLM result with its limits: B3 is vLLM's kernels run on their own, not a vLLM server, and
RF1's gain over them on one layer (1.9–2.8×) falls to 1.1–1.3× on a whole model.

### Assumptions and future work

* **Rank-sorted batches:** we test two scheduler assumptions and leave their overhead, and any vLLM
  changes they need, to future work; neither has been checked against vLLM. Physically sorted
  tokens use `rf2_segment`, which reads whatever order it receives; a sorted index list, the weaker
  assumption and closer to vLLM's existing per-step sort, uses `rf2_index`. Output is correct in any
  order, but unsorted input without a list skips less.
* **Multi-adapter serving:** B1 is easiest to explain here, and the slot savings (one per adapter
  instead of seven) do not depend on kernel speed. Still to write: a detailed comparison of memory
  use and locality against vLLM. Whether locality gives RF2 a larger lead is an untested hypothesis.

### Sweep plan

How the RF2 kernels differ, the two scheduler assumptions, and why each sweep exists:
**[docs/RF2_TESTING.md](docs/RF2_TESTING.md)**.

Report every case, including those where RF2 loses, against the ideal bound.

| Case | Workload | Config |
|---|---|---|
| **Best** | Large batches of 25–99% low ranks with the rest at r_max, down to a single request at r_max (`low_frac_100`) forcing RF1 to full width | `rf2_ideal.yaml` |
| **Average** | Sampled rank mixes, simulated serving steps, and the four chunked-prefill step types in `rf2_mixed.yaml` (decode only, one prompt filling the budget, short prompts not filling it, many partial prefills); weight results by how often each step type occurs. These mixes are assumptions, not a real trace | `rf2_realistic.yaml`, `rf2_realistic_decode.yaml`, `rf2_shapes.yaml`, `rf2_mixed.yaml` |
| **Worst** | All tokens at one rank (RF1 is already narrow), and small decode batches where launch cost dominates | `all_min` and `all_max` in `rf2_ideal.yaml`, `parity.yaml`; `num_requests` in `rf2_realistic_decode.yaml` |

Change one variable at a time to isolate its effect:

| Variable | Config key |
|---|---|
| Rank mix | `dist` |
| Maximum rank | `r_max` (64, 128, 256; the paper's setting goes up to 256) |
| Batch size | `num_requests` |
| Prefill vs decode | `tokens_per_request`, `lengths` |
| Decode share of a mixed step (chunked prefill) | `lengths: chunked_<budget>_<chunk>` (fills the budget) or `serving_<k>_<budget>` (k sampled prompts, may not fill it), with `num_requests` decodes |
| Step types that are not a cross product | `grid` as a list of grids, each expanded on its own |
| Separate rank mix for prefill requests | `prefill_dist` (decodes use `dist`) |
| Total tokens | `batches` (`[num_requests, tokens_per_request]`) |
| Request order | `request_order` |

Run `parity.yaml` and `parity_add.yaml` before any other RF2 sweep.

## Layout

```
rankfuse/
  interface.py        shared contract: MatryoshkaLoRA, Batch, @register, REGISTRY
  reference.py        B0 reference (correctness oracle)
  baselines/          B1 padded+mask (+ graphed), B2 grouped (+ graphed), B3 vLLM's kernels
                      (vllm_ops.py); B4 stub with a plan
  kernels/            Triton kernels. rankfuse_fused.py: the shared split-K base and RF1
                      (rf1_fused); RF2 variants in rankfuse_rf2_{segment,index,bucketed,ranges}.py;
                      tuned_a100.py: per-width launch settings for one A100; earlier one-kernel
                      ablations: rf1_single, rf2_fused, rankfuse_mask.py, rankfuse_ragged.py
  workloads.py        seeded shapes and rank distributions
  roofline.py         analytic bytes/FLOPs/lower-bound per strategy
tests/                correctness vs B0 for every registered impl; kernel serving features; roofline
bench/run_sweep.py    one-layer benchmark -> CSV (eager or CUDA-graph timing); bench/configs/*.yaml
bench/model_harness.py, model_forward.py   whole Qwen2.5-0.5B with each LoRA backend: check, timing
bench/vllm_step_time.py                    LoRA share of a stock vLLM step
bench/ncu_driver.py   one profiled call for Nsight Compute
analysis/plot.py      latency vs tokens, measured vs roofline
analysis/rf2_eval.py  RF2 sweep tables, including sorted vs random order
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

## Repository conventions

* **Weight layout:** `A_t [r_max, h_in]` (A transposed) and `B [r_max, h_out]`, so a rank-r prefix is
  a contiguous block of rows in both.
* **Tokens, not requests:** decode and prefill use the same flattened format, and `Batch.seg_starts`
  keeps request boundaries.
* **Per-rank scale:** `MatryoshkaLoRA.scale_per_rank` (for example α/r).
* **Correctness first:** nothing gets benchmarked until it passes `tests/test_correctness.py` against B0.

## Adding an implementation

```python
from rankfuse.interface import register, apply_scale

@register("my_kernel", requires_cuda=True)   # a new, unused name
def my_kernel(x, lora, token_ranks):   # x [T, h_in], lora.A_t [r_max, h_in], lora.B [r_max, h_out]
    ...
    return apply_scale(y, lora, token_ranks)   # y [T, h_out], LoRA delta only
```

Import the module from its package `__init__.py`. It is then tested against B0 and benchmarked automatically.
`register` also takes `dtypes=` (supported input dtypes; tests skip the others) and `prepare=`, a
function `(x, lora, token_ranks) -> kwargs` returning the per-step metadata a server would build
once (batch maximum rank, rank order, ...); passing it removes the call's host syncs, which lets
`run_sweep.py --timing graph` capture the call.
