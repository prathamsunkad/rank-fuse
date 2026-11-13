# RankFuse: Project Plan

## 1. The problem in one paragraph

MatryoshkaLoRA trains one adapter whose smaller ranks are **prefixes** of the full matrices: rank 4 uses `A[:, :4]` and `B[:4]`, rank 32 uses `A[:, :32]` and `B[:32]`. At serving time, each request can pick a rank (1/8, 1/4, 1/2 or full) to trade accuracy for compute. A batch therefore mixes ranks while sharing the same weights. No existing serving system knows that ranks are nested, and a naive kernel either pads every request to the largest rank or splits the batch by rank. RankFuse is a Triton kernel that serves the whole mixed-rank batch in one launch, loading the shared prefix once and computing each request only up to its own rank.

## 2. What we are claiming

We make two separate claims. Each is tested against different baselines.

**Claim 1: existing multi-LoRA systems waste work on Matryoshka adapters; RankFuse doesn't.**
vLLM (Punica-style kernels) and S-LoRA would need each rank registered as its own adapter, which means duplicate copies of the shared prefix and repeated weight loads. RankFuse loads the prefix once per batch.
*Expected: RankFuse wins, mostly at small batch sizes. This is the safer claim.*

**Claim 2: low-rank requests should be cheaper, even inside a mixed batch.**
The simple approach (pad to the largest rank, then mask) makes a rank-4 request do the same work as a rank-32 request. RankFuse skips unused ranks.
*Expected: uncertain. This operator is memory-bound, and the dominant traffic (reading x, reading and writing y) is the same at every rank. This is the real research question.*

Suggested phrasing for the report:

> RankFuse avoids the redundant weight storage and loading that existing multi-LoRA systems incur on Matryoshka adapters (Claim 1). We also test whether skipping unused rank computation gives further speedups over a simple padded approach, and identify when it does and doesn't (Claim 2).

| Outcome | What we can say |
|---|---|
| Win both | Strong result |
| Win Claim 1, roughly tie Claim 2 | Still good: we beat existing systems and show rank savings matter less than expected in decode, with an explanation |
| Win Claim 2 only in some regimes | Report exactly where the benefit appears |

The hypothesis is falsifiable, so a well-explained "no" on Claim 2 is a valid result.

## 3. Baselines

Every implementation computes the same thing: for each token i with rank r_i, `y_i = x_i @ A[:, :r_i] @ B[:r_i]`.

| ID | Name | How it works | Launches | Shares weights | Skips unused ranks | Tests |
|---|---|---|---|---|---|---|
| B0 | Reference | Python loop per token, fp32 | many | ✓ | ✓ | correctness only |
| B1 | Padded + mask | `v = x @ A[:, :r_batch_max]`, zero `v[i, r_i:]`, `y = v @ B[:r_batch_max]` | ~3–4 | ✓ | ✗ | Claim 2 (headline) |
| B2 | Grouped by rank | one matmul pair per distinct rank, gather/scatter tokens | 2 per group | ✓ | ✓ | Claim 2 |
| B2g | Grouped + CUDA graph | B2 captured in a CUDA graph | 1 replay | ✓ | ✓ | separates launch overhead |
| B3 | vLLM LoRA ops | each rank registered as its own adapter, padded to max rank | 2 | ✗ | ✗ | Claim 1 |
| B4 | S-LoRA bgmv (optional) | each rank its own adapter at its true rank | 2 | ✗ | ✓ | Claim 1 |
| RF1 | RankFuse v1 | fused Triton kernel, rank tiles masked per token | 1 | ✓ | partly | — |
| RF2 | RankFuse v2 | only valid (token tile, rank tile) work items launched | 1 | ✓ | ✓ | — |

Why B1 masks the **activation** and not the weights: all requests share one A and one B, so the weights can't be masked differently per request in one matmul. Zeroing column k of row i in `v` removes rank component k from request i only.

Why B1 is the hardest baseline: it's two large cuBLAS matmuls that load the shared weights once. Its only waste is FLOPs, and FLOPs aren't the bottleneck.

B3 is what you'd do with today's most deployed engine. B4 is the best published mixed-rank technique; S-LoRA itself is a 2023 research prototype that is not widely deployed.

<!-- ## 4. Back-of-envelope numbers (`python -m rankfuse.roofline`)

**Weights.** One decode request at each of ranks 4/8/16/32, h = 4096, fp16:

| Strategy | Weight rows read | Weight data |
|---|---|---|
| RankFuse / B1 (shared prefix) | 32 | 0.5 MiB |
| S-LoRA-style (separate adapters, true ranks) | 4+8+16+32 = 60 | 0.94 MiB |
| vLLM-style (separate adapters, padded to 32) | 4 × 32 = 128 | 2.0 MiB |

**Tokens.** Each token reads x and reads and writes y: 3 × 4096 × 2 B ≈ 24 KiB, whatever its rank.

- Small batch (decode, a few to tens of tokens): weights dominate, so Claim 1 should show up clearly.
- Large batch (512 tokens ≈ 12 MiB of x/y traffic): weights are a small share, so all strategies converge. The model predicts the ideal fused kernel is within about 1% of B1.

**Launch overhead is the hidden variable.** At decode sizes, the memory-bound lower bound is about 0.3 µs, roughly 10× smaller than one kernel launch (a few µs). So at small batch, the number of launches (B1: 3–4, B2: 2 per group, RankFuse: 1) will probably decide the results more than bytes or FLOPs do. B2 vs. B2g measures this directly. The report needs to say which effect each win comes from. -->

Modeling assumptions are listed at the top of `rankfuse/roofline.py`. The vLLM row assumes its kernel reads the zero-padded rows; verify with `ncu`.



## 6. Experiments

| Axis | Values |
|---|---|
| Tokens | decode: 1–256 requests × 1 token; prefill: 1–16 requests × 128–512 tokens |
| Hidden size | 4096, 8192 (and asymmetric h_in ≠ h_out) |
| r_max | 64, 128; tiers r_max/8, /4, /2, full |
| Rank distribution | all_min, all_max, uniform, skew_low (0.4/0.3/0.2/0.1), skew_high |
| Adapters | 1 (main); several Matryoshka adapters × mixed ranks (later, the setting where B3/B4 are meaningful) |
| Metrics | median latency, tokens/s, DRAM bytes and achieved bandwidth (`ncu`), each next to its roofline bound |

All headline numbers come from one GPU type on PACE Phoenix (H100 by default; record it).

Key figures for the report:
1. Latency vs. batch size, all implementations, decode, uniform mix.
2. Speedup vs. B1 and B3 across rank distributions (shows when Claim 2 holds).
3. Measured bytes vs. roofline (shows why).
4. B2 vs. B2g vs. RankFuse (launch overhead).

## 7. Milestones

| Week | Deliverable |
|---|---|
| 1–2 | ✅ Repo, interface, B0/B1/B2, workloads, roofline, tests, harness. First decode/prefill sweep of B1/B2/B2g on Phoenix. Start the S-LoRA build (time-boxed). |
| 3–4 | RankFuse v1 (masked) passes tests. B3 wrapped and verified with ncu. |
| 5–6 | RankFuse v2 (ragged). B4 working, or a labeled Triton re-implementation. First full sweep. |
| 7–8 | Multi-adapter × multi-rank experiments; full sweeps on the chosen GPU. |
| 9 | ncu profiling; ablations (masked vs. ragged, tile sizes, with and without CUDA graphs, L2 flush on/off). |
| 10 | Final analysis and report. |

Suggested split for four people: two on the RankFuse kernels, one on B3/B4 and the third-party builds, one on harness, roofline, profiling and plots (this role unblocks everyone first).

## 8. Open risks and decisions

- **S-LoRA build** may not work with current PyTorch/CUDA. Time-box it to 2–3 days, then fall back to a labeled Triton re-implementation.
- **vLLM padding behavior** is assumed (buffers allocated at `max_lora_rank`). Confirm on the pinned version with ncu.
- **Matryoshka per-rank scaling** and the exact rank tiers: confirm from the paper and set them in `workloads.py`.
- **Phoenix account/QOS:** fill in `-A` and `-q` in `scripts/phoenix/sweep.sbatch`.
- **L2 flushing** is on by default (cold weights, as in a real forward pass). Report one run with it off as an ablation.
