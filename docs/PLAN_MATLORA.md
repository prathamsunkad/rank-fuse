# RankFuse: Matryoshka-LoRA Serving Plan

This document fixes the project scope and lays out every step from the current kernels to
end-to-end serving results. The project's claims, baselines and repository conventions are in the
[README](../README.md).

Status as of 2026-10-25, verified on one NVIDIA H200 on PACE ICE (vLLM 0.32.0, PyTorch 2.13.0,
Triton 3.7.1; 1020 tests passing, 29 skipped: B3 in fp32). How the kernels work and every number below:
[ALGORITHMS.md](ALGORITHMS.md).

* Done:
  * P0.1–P0.3: H200 as the headline GPU, `requirements-lock.txt`, `scripts/ice/` (env, check,
    sweep, ncu). P0.4 in part: Qwen2.5-0.5B-Instruct as the development model, loading in stock
    vLLM with every generated adapter (`scripts/ice/vllm_lora_check.py`); the 7–8B headline
    model is not chosen yet. P0.5: `docs/VLLM_LORA_NOTES.md`.
  * P1.1: `rf2_fused`, plus two RF2 variants on the split-K design: `rf2_segment` (blocks of
    consecutive tokens) and `rf2_bucketed` (one kernel pair per rank bucket, chosen per batch by
    `use_rf2`). A third, sorted `rf2_splitk`, was removed: on rank-sorted input it forms the same
    blocks as `rf2_segment`, plus a sort and index lookups. P1.2. P1.3: `b3_vllm`. P1.5. P1.6 in part: ncu
    (`scripts/ice/ncu_profile.sh`); nsys launch counts not done. P1.7 in part: `rf1_fused`
    re-tuned by hand for the H200 (split-K shrink, multi-wave and tiled expand); no autotuning.
  * P2.1. P2.2: `bench/vllm_step_time.py` (LoRA share of a stock vLLM step) and
    `bench/model_forward.py` (whole-model forward in PyTorch with every backend, an early P5.1
    without vLLM).
  * P3.1–P3.3 (`rankfuse_lora`: add into output, output slices, CUDA-graph capture).
* Results (H200, fp16, CUDA-graph replay): one 4096 × 4096 layer, `rf1_fused` is 2.1–2.4× faster
  than B1 at 1–128 tokens, 1.3–1.7× at 256–2048, 1.07× at 4096 and even at 8192; 1.9–2.8× faster
  than B3.
  Whole Qwen2.5-0.5B forward: 1.38–1.47× faster than with B1 and 1.12–1.31× than with B3 at
  decode. LoRA is 8–52% of a stock vLLM decode step for that model. RF2 gains 10–15% only on
  large mixed-rank prompts.
* Next: a lean launch path for `rf1_fused` (about 108 µs of Python per eager call; vLLM runs
  long prompts eagerly, where RF1 is 0.80–0.85× of B1 today), sweeps on 7–8B shapes, P4, P1.4
  and the Stage 2 kernel (P3.4).

---

## 1. What we are building

RankFuse is **a rank-aware LoRA operator plus a thin integration into an existing serving
engine (vLLM)**, so that the engine can serve Matryoshka adapters natively. We do not build a
new serving system. Batching, scheduling, KV-cache management and the HTTP server all come
from vLLM.

Existing systems treat ranks 8, 16, 32 and 64 of one Matryoshka adapter as four unrelated
adapters. That costs:

1. **Memory and slots:** four stored copies (vLLM pads each to `max_lora_rank`) and four LoRA
   slots per adapter.
2. **Bandwidth:** the shared prefix is loaded once per variant present in a batch.
3. **Launches and padding:** per-adapter kernels, extra launches, padded work.

RankFuse removes all three by using the fact that every rank is a prefix of the same weights.

### Where RankFuse sits in the serving stack

| Layer | What it does | vLLM | Punica / S-LoRA | RankFuse |
|---|---|---|---|---|
| 5. API | requests arrive with a prompt, an adapter id and (for us) a rank | ✓ | ✓ | adds a `rank` field |
| 4. Scheduler | continuous batching: which requests run this step | ✓ | ✓ (S-LoRA adds adapter-aware memory) | reuse vLLM's |
| 3. Memory managers | KV-cache paging; which adapters are resident on the GPU | ✓ (LoRA slots, LRU) | S-LoRA: unified paging of KV + adapters | store each Matryoshka adapter once, not once per rank |
| 2. Batch metadata | per step: token → adapter index, built once, used by every layer | ✓ (Punica wrapper) | ✓ | add token → rank next to it |
| 1. Operators | the LoRA matmuls inside every adapted projection | Punica-derived Triton shrink/expand | SGMV/BGMV (Punica), MBGMV (S-LoRA, mixed ranks) | RF1/RF2: nested-rank-aware fused kernels |

The contribution is orthogonal to Punica and S-LoRA: it could be added to either of them or to
vLLM. We integrate with vLLM because it is the most widely deployed engine.

### Assumptions

* **One rank per request, the same in every layer.** Token → rank metadata is built once per
  engine step and reused by every LoRA layer. The kernels accept any `token_ranks` per call, so
  per-layer ranks stay possible later.
* Ranks are Matryoshka prefixes: rank r uses rows `[:r]` of `A_t` and `B`. Tiers are powers of
  two from 1 to `r_max` (1, 2, 4, ..., 64 for `r_max = 64`).
* Performance does not depend on weight values, so Levels 1–3 use random adapters with the
  correct shapes. Trained adapters are needed only for the optional quality study (Level 4).

### Out of scope

The scheduler, continuous batching, KV-cache paging, tokenization and the API server.
One optional extension touches the scheduler (section 9).

---

## 2. Two deployment stages

### Stage 1: one Matryoshka adapter, mixed ranks

* Every token uses the same `A_t` / `B`, each at its own rank.
* Kernel: RF1 / RF2 as they exist, plus the integration features in section 6.
* Stock vLLM baseline: register the 4 rank slices as 4 adapters (each padded to
  `max_lora_rank`, 4 LoRA slots) and route each request to the slice for its rank.
* Expected wins: weights read once (Claim 1), one launch with `v` on chip (fusion),
  optionally skipping padded ranks (Claim 2, RF2).
* Limit: with one adapter the memory saving is small in absolute terms. The story is
  bandwidth and launches.

### Stage 2: many Matryoshka adapters, mixed ranks

* Each token carries `(adapter_id, rank)`. Weights are shared only among tokens on the same
  adapter.
* Kernel: segmented by adapter (like Punica's SGMV), rank mask inside each segment. RF2's
  ordering becomes adapter-major, then rank-major.
* Expected wins:
  1. **Adapter capacity.** With ranks registered separately, one adapter takes 4 slots, so only
     `max_loras / 4` distinct adapters stay resident and the rest wait for loads from CPU
     memory. RankFuse uses 1 slot per adapter. This changes latency and throughput regardless
     of kernel speed, and it is directly comparable to S-LoRA's adapter-scaling results.
  2. Bandwidth and launches, as in Stage 1, per adapter segment.
* Baselines: stock vLLM with rank variants as separate adapters; S-LoRA if it builds.

Stage 1 is designed so that Stage 2 is an extension: weights stackable as
`[num_adapters, r_max, h]`, metadata as per-token arrays, the same kernel inner loop with an
adapter offset in front.

---

## 3. Reality check: how large can end-to-end gains be?

The LoRA delta is a small part of a forward pass. For a 7–8B model the base projections and
attention dominate; LoRA adds about 1–3% of the FLOPs, though at decode its weight reads and
launches can be a larger share of time. If LoRA is 10% of a decode step and RankFuse makes it
2× faster, the step gets about 5% faster.

Consequences:

1. **Measure LoRA's share of step time first** (task P2.1). It is the ceiling on end-to-end
   gains from a faster kernel, and it goes in the report.
2. **Kernel results are the primary result;** end-to-end results show how much survives.
3. **Results that escape this ceiling:**
   * adapter capacity in Stage 2 (fewer evictions, less queueing);
   * decode at small batch, where launches and weight loads matter most;
   * LoRA on every projection (q, k, v, o, gate, up, down): 7 calls × 32 layers = 224 LoRA calls
     per step, so per-call overhead adds up.

Claim 2 needs the same honesty. LoRA is memory-bound, so a rank-8 token is not much cheaper
than a rank-64 token in a mixed batch. Report where rank skipping helps and where it does not,
with the reason.

---

## 4. Claims and the evidence for each

| Claim | Proven at | Key figure |
|---|---|---|
| **1.** Existing systems duplicate Matryoshka prefixes; RankFuse loads them once | Level 1 (bytes and latency vs B3/B4); Level 3 Stage 2 (adapter capacity) | DRAM bytes vs B3; throughput vs number of adapters |
| **Fusion** removes launches and intermediate traffic | Level 1 (RF1 vs B1, B1 graphed) | latency vs T, with launch breakdown |
| **2.** Skipping unused ranks helps in some regimes | Level 1 (RF2 vs RF1 across rank distributions) | speedup vs rank distribution and T |
| **End-to-end:** gains survive in a real server | Levels 2–3 | step time with LoRA-share breakdown; TPOT and goodput vs request rate |

Claim 2 is falsifiable: the operator is memory-bound and most of its traffic is the same at every
rank, so a well-explained "no" is a valid result. Report exactly where any benefit appears.

---

## 5. Evaluation methodology

Each claim is proven at the lowest level that can prove it. Higher levels show how much of
the effect survives in a real system.

### Level 0: correctness

* Every implementation matches B0 within dtype tolerance (`tests/test_correctness.py`).
* Stage 2: multi-adapter tests where each token's `(adapter, rank)` output matches a per-token
  reference.
* vLLM: with random-weight adapters and greedy decoding, RankFuse and stock vLLM produce the
  same tokens for the same adapter and rank.

### Level 1: operator microbenchmarks (main evidence for Claims 1 and 2)

* **Shapes from real models.** Llama-3-8B (confirm for the chosen model):

  | Projection | h_in → h_out |
  |---|---|
  | q, o | 4096 → 4096 |
  | k, v (grouped-query attention) | 4096 → 1024 |
  | gate, up | 4096 → 14336 |
  | down | 14336 → 4096 |

* **Sweep:** T (decode 1–256; prefill 128–8192), rank distribution, `r_max` (64, 128), dtype,
  number of adapters (Stage 2).
* **Implementations:** B1, B1 in a CUDA graph, B2 / B2p, B3 (vLLM's ops called directly),
  B4 (S-LoRA) if available, `rf1_fused`, RF2.
* **Metrics:** latency (median, p10/p90); speedup over B1 and B3; fraction of the roofline
  minimum; from `ncu`: DRAM bytes, achieved bandwidth (% of peak), L2 traffic and hit rate;
  from `nsys`: kernel count and launch overhead; registers and spills.
* **Figures:** latency vs T; speedup vs rank distribution; measured bytes vs roofline;
  breakdown of each win into launches, bytes and FLOPs.

### Level 2: full-model forward pass (static batches)

* A real model with a Matryoshka adapter on all projections, run through vLLM's offline
  engine with fixed batches and identical inputs.
* **Measure:** decode step time at batch 1, 8, 32, 128; prefill time at several prompt
  lengths; the same with no LoRA (gives the LoRA share).
* **Variants:** base (no LoRA); stock vLLM with rank variants as separate adapters; stock vLLM
  with one full-rank adapter; vLLM + `rf1_fused`; vLLM + RF2.
* **CUDA graphs on (vLLM's default) and off.** With graphs on, launch-count wins shrink, and
  the report must show this.

### Level 3: online serving (end-to-end)

* **Server:** vLLM on one GPU with fixed `max_num_seqs`, `max_loras`, `max_lora_rank`.
* **Workload:**
  * prompts from ShareGPT (real length distributions) or a fixed synthetic distribution;
  * Poisson arrivals at increasing request rates up to saturation;
  * rank per request drawn from the rank distribution, fixed for the whole request;
  * Stage 2: adapter per request from a popularity distribution (uniform and Zipf).
* **Metrics:**
  * throughput: requests/s and output tokens/s at saturation;
  * TTFT (time to first token), p50 and p99;
  * TPOT / ITL (time per output token / inter-token latency), p50 and p99;
  * goodput: requests/s served within a latency target (e.g. p99 TPOT < 100 ms);
  * Stage 2: distinct adapters resident, adapter loads and evictions, queueing delay.
* **Figures:** p50/p99 TTFT and TPOT vs request rate per system; throughput vs number of
  adapters (Stage 2); goodput at a fixed latency target.

### Level 4: quality (optional, motivation only)

Either cite the MatryoshkaLoRA paper's accuracy-vs-rank results, or train one small Matryoshka
adapter with PEFT and report perplexity or task accuracy per rank.

### Rules for every level

* Pin GPU type, driver, CUDA, PyTorch, Triton and vLLM versions; record them in every results
  file.
* One GPU type for headline numbers; others are secondary.
* Warmup, many repetitions, report spread; run serving experiments at least 3 times.
* Change one variable at a time: same scheduler settings, CUDA-graph setting, prompts and
  seeds across systems.
* Compare against vLLM's fast path (CUDA graphs on); graphs-off is an ablation.
* Every latency win must be explained by a Level 1 measurement (bytes, launches) or a
  memory-capacity measurement.

---

## 6. Implementation work

Tasks are grouped into phases. Each task lists where the code goes and what "done" means.

### P0. Setup and decisions

| ID | Task | Done when |
|---|---|---|
| P0.1 | Pick the headline GPU on PACE ICE and record its specs in `rankfuse/roofline.py` if missing | GPU type agreed; roofline has its bandwidth and peak |
| P0.2 | Pin versions: PyTorch, Triton, vLLM. Write `requirements-lock.txt` | one environment builds on ICE and runs `pytest -q` |
| P0.3 | ICE job scripts: `scripts/ice/` (env setup, kernel sweep, ncu, model forward, serving) adapted from `scripts/phoenix/` | each script runs a smoke job on ICE |
| P0.4 | Choose models: a small development model (Qwen2.5-0.5B/1.5B or Llama-3.2-1B) and a 7–8B headline model (Llama-3.1-8B or Qwen2.5-7B); download to scratch | both load in vLLM on ICE |
| P0.5 | Read the pinned vLLM version's LoRA code end to end and list the exact files and functions to change (likely under `vllm/lora/`: layers, model manager, Punica wrapper, Triton ops, request type) | a short note in `docs/VLLM_LORA_NOTES.md` |

### P1. Finish the operator study (Level 1)

| ID | Task | Where | Done when |
|---|---|---|---|
| P1.1 | Restructure RF2 like `rf1_fused`: shrink once per program, column split, runtime T, no per-call setup; sort tokens by rank and stop each tile at its local max rank | `rankfuse/kernels/rankfuse_ragged.py` (or a new `rankfuse_ragged_v2.py`) | passes all correctness tests |
| P1.2 | Real model shapes in benchmark configs, including asymmetric k/v, gate/up, down | `bench/configs/` | sweep runs on ICE |
| P1.3 | B3: vLLM's shrink/expand ops called directly, each rank slice registered as its own padded adapter | `rankfuse/baselines/vllm_ops.py` (plan already in the file) | passes correctness tests; registered as `b3_vllm` |
| P1.4 | B4 decision: time-box the S-LoRA build to 2–3 days, else a labeled Triton "S-LoRA-style" re-implementation | `rankfuse/baselines/slora_bgmv.py` | registered as `b4_slora` |
| P1.5 | B1 in a CUDA graph, with `r_b` passed in (no host sync), as the fair decode baseline | `rankfuse/baselines/padded.py` (new registered variant) | registered and tested |
| P1.6 | Profiling scripts: `ncu` metrics (DRAM bytes, bandwidth %, L2 bytes and hit rate) and `nsys` launch counts, written to CSV | `scripts/ice/ncu_profile.sh`, `analysis/` | RF1 bytes ≈ roofline minimum; B3 bytes higher |
| P1.7 | Autotune block sizes for `rf1_fused` and RF2 per GPU | kernel files | best config recorded per GPU |
| P1.8 | Level 1 figures | `analysis/plot.py` | figures in section 5, Level 1 |

### P2. Measure the ceiling in stock vLLM (Level 2, no RankFuse code)

| ID | Task | Where | Done when |
|---|---|---|---|
| P2.1 | Generate random Matryoshka adapters in PEFT format for the chosen models (all 7 projections, tiers r_max/8…r_max), plus the 4 rank-sliced copies stock vLLM needs | `scripts/make_adapters.py` | stock vLLM loads them |
| P2.2 | Model-forward benchmark: base vs LoRA, decode batch 1/8/32/128, several prefill lengths, CUDA graphs on and off | `bench/model_forward.py` | LoRA share of step time reported per setting |

P1 and P2 are independent and should run in parallel. P2.2 decides how much effort the
end-to-end story deserves.

### P3. Kernel features needed for integration

| ID | Task | Done when |
|---|---|---|
| P3.1 | Add into the output: `y += delta` into the base layer's output tile inside the kernel | tests compare against base output + B0 delta |
| P3.2 | Output slices for fused projections (QKV, gate/up): one call, several slices with offsets | tests per slice layout used by vLLM |
| P3.3 | CUDA-graph safety: no host syncs, `r_limit` from the adapter's `r_max` or a bucket, fixed compiled variants | the kernel captures and replays in a CUDA graph inside a test |
| P3.4 | Stage 2 kernel: per-token adapter index, stacked weights `[num_adapters, r_max, h]`, adapter-major then rank-major ordering | multi-adapter correctness tests pass |

### P4. vLLM integration, Stage 1

All changes live in a fork of the pinned vLLM version, behind a flag so stock and RankFuse run
from the same build.

| ID | Task | Done when |
|---|---|---|
| P4.1 | Request API: a Matryoshka rank on each LoRA request | a request can choose any tier |
| P4.2 | Adapter manager: load one Matryoshka adapter into one slot; all its ranks share it | one slot per adapter regardless of ranks requested |
| P4.3 | Batch metadata: build per-token ranks wherever vLLM builds per-token adapter indices, once per step | ranks visible to every LoRA layer |
| P4.4 | LoRA layers: replace the shrink/expand calls with the RankFuse op | all LoRA layer types used by the model route through RankFuse |
| P4.5 | Feature flag (e.g. an env var or engine arg) selecting stock or RankFuse | both paths run from one build |
| P4.6 | Integration check: greedy outputs identical to stock vLLM for the same adapter and rank | exact token match on a fixed prompt set |

### P5. Stage 1 experiments (Levels 2 and 3)

| ID | Task | Where | Done when |
|---|---|---|---|
| P5.1 | Level 2 with RankFuse: same matrix as P2.2 plus `rf1_fused` and RF2 | `bench/model_forward.py` | step-time table and figure |
| P5.2 | Serving load generator: ShareGPT or synthetic prompts, Poisson arrivals, rank distribution per request | `bench/serving/` | runs against stock and RankFuse servers |
| P5.3 | Level 3 sweeps: request rate up to saturation, 3 runs each | `scripts/ice/serving.sbatch` | latency-vs-rate and goodput figures |

### P6. Stage 2: multi-adapter

| ID | Task | Done when |
|---|---|---|
| P6.1 | Integrate the P3.4 kernel; adapter manager handles many Matryoshka adapters, one slot each | Level 0 checks pass with several adapters |
| P6.2 | Multi-adapter workload: adapter popularity (uniform, Zipf) and rank distribution per request | load generator supports both |
| P6.3 | Capacity experiment: throughput, latency and evictions vs number of distinct adapters, fixed `max_loras` | throughput-vs-adapters figure |
| P6.4 | S-LoRA comparison if P1.4 produced a working build | comparison table or a documented reason it is absent |

### P7. Optional

| ID | Task |
|---|---|
| P7.1 | Level 4 quality: accuracy or perplexity per rank, from the paper or a small trained adapter |
| P7.2 | Rank as a load knob: under heavy load, serve requests at a lower tier to meet a latency target. Only worthwhile if Claim 2 shows lower ranks are faster |

### P8. Write-up

Claims mapped to figures as in section 4, with the memory-bound and LoRA-share explanations up
front, and every negative result explained.

---

## 7. Order of work

```
P0 setup ──┬── P1 operator study ──┐
           └── P2 LoRA share ──────┴── P3 kernel features ── P4 vLLM Stage 1 ── P5 experiments
                                                                   └── P6 Stage 2 ── P8 write-up
```

1. P0 first: nothing else produces trustworthy numbers without a pinned environment.
2. P1 and P2 in parallel. P2.2's LoRA share sets expectations for everything after it.
3. P3 before P4: the kernel must be graph-safe and write into vLLM's output layout before
   integration is possible.
4. P4.6 (identical outputs) gates every Level 2 and Level 3 result.
5. Stage 2 (P6) reuses everything from Stage 1.

Suggested split for four people: two on kernels (P1.1, P1.7, P3), one on baselines and vLLM
(P1.3–P1.5, P0.5, P4), one on harness, profiling and experiments (P0.3, P1.6, P2, P5).

---

## 8. Repository layout additions

```
rankfuse/kernels/      rankfuse_fused.py, rankfuse_rf2_*.py (done), multi-adapter
rankfuse/baselines/    vllm_ops.py (B3, done), slora_bgmv.py (B4), graphed B1 (done)
rankfuse/integration/  small adapter layer between vLLM's LoRA calls and RankFuse ops
bench/configs/         real-model shapes
bench/model_forward.py Level 2 in PyTorch (done); bench/vllm_step_time.py stock vLLM (done)
bench/serving/         Level 3 load generator and result collection
scripts/ice/           env setup, sweeps, ncu/nsys, model forward, serving jobs
scripts/make_adapters.py
third_party/vllm/      pinned vLLM fork (submodule) with the RankFuse flag
docs/VLLM_LORA_NOTES.md
```

---

## 9. Risks and open decisions

* **End-to-end ceiling.** If P2.2 shows a small LoRA share, end-to-end speedups will be
  small. The capacity result (P6.3) and the decode-at-small-batch results then carry the
  end-to-end story.
* **CUDA graphs** remove much of the launch overhead that fusion saves at decode. Report
  results with graphs on as the main setting.
* **vLLM version churn.** The LoRA code moves between releases. Pin one version early and do
  not upgrade mid-project.
* **vLLM padding behavior** (stock adapters padded to `max_lora_rank`) is assumed; confirm it in
  the pinned version with `ncu` (P1.6).
* **S-LoRA build** may not work with current PyTorch/CUDA. Time-box it (P1.4).
* **Matryoshka tiers and scaling** must be confirmed from the paper and set in
  `rankfuse/workloads.py`.
* **L2 flushing** is on by default (cold weights, as in a real forward pass). Report one run with
  it off as an ablation.
* **Gated models** (Llama) need a Hugging Face token on ICE; Qwen models are open.
* **Decided (2026-10-22):**
  * GPU: H200 on PACE ICE (`ice-gpu` partition, `--gres=gpu:h200:1`); H100, then A100 if H200
    is unavailable. Headline numbers stay on one GPU type.
  * vLLM: the newest stable release at setup time, then frozen in `requirements-lock.txt`.
  * Development model: Qwen2.5-0.5B.
  * Ranks: `r_max = 64`, tiers 1, 2, 4, 8, 16, 32, 64 (`rankfuse.workloads.default_tiers`).
* **Still open:** headline 7–8B model (Qwen2.5-7B suggested: no access token needed); target
  modules (all 7 projections recommended); whether ICE needs a charge account or QOS.
