# RankFuse: the RF1 and RF2 kernels, explained

*CS 6222 MLSys project. How RankFuse computes Matryoshka LoRA for a batch of requests that each
want a different rank: what is stored, what each GPU program reads, how the matrix
multiplications are tiled, what is saved compared with the baselines, and how the kernels fit
the full serving story from Stage 1 (one adapter, many ranks) to Stage 2 (many adapters).*

Status: 25 Oct 2026. All numbers are from one NVIDIA H200 (132 SMs, 141 GB HBM at 4.8 TB/s,
about 50 MB L2), fp16, Triton 3.7.1, measured by CUDA-graph replay (GPU time only) unless noted.

---

## 1. The problem in one paragraph

A LoRA adapter changes a model by adding a small correction to each adapted projection: for a
token with input `x`, the projection's output gets `x · A · B` added, where `A` and `B` are thin
matrices of rank `r`. A **Matryoshka** LoRA adapter is trained so that its first `r` rows work as
a smaller adapter on their own: rank 8 is the first 8 rows of the rank-64 adapter, rank 16 the
first 16, and so on (we use ranks 1, 2, 4, 8, 16, 32, 64). A server can then let every request
choose its own rank, small for speed or large for quality. A batch therefore mixes ranks while
sharing one set of weights. Existing serving systems do not know the ranks are nested: they must
register each rank as a separate adapter, which stores and reads the shared rows several times.
RankFuse computes the whole mixed-rank batch from **one copy** of the adapter.

---

## 2. Notation

| Symbol | Meaning | Example (one projection of a 7–8B model) |
|---|---|---|
| `T` | tokens in the batch (all requests flattened) | 1–8192 |
| `h_in`, `h_out` | the projection's input and output width | 4096, 4096 |
| `r_max` | largest rank the adapter supports | 64 |
| `r_t` | rank chosen by token `t` (all tokens of a request share it) | 1 … 64, or 0 for no adapter |
| `x` | input, `[T, h_in]` | |
| `A_t` | A transposed, `[r_max, h_in]`: the rank-`r` adapter is rows `0 … r-1` | |
| `B` | `[r_max, h_out]`: the rank-`r` adapter is rows `0 … r-1` | |
| `v` | the intermediate, `[T, r]` | |
| `y` | the LoRA correction, `[T, h_out]`, usually added into the base output | |

For every token:

```
v[t] = x[t] · A_t[0:r_t]ᵀ          shrink: h_in numbers  →  r_t numbers
y[t] = v[t] · B[0:r_t]              expand: r_t numbers   →  h_out numbers
out[t] += scale(r_t) · y[t]          add into the base projection's output
```

Storing `A` transposed (`A_t`) and `B` row-major makes every rank a **contiguous block of rows** in
both matrices, so "use rank `r`" means "read the first `r` rows".

**The masking identity.** Because `y[t] = Σ_k v[t,k] · B[k]`, computing `v[t,k]` for more rank
positions than a token needs and then setting `v[t,k] = 0` for `k ≥ r_t` gives exactly the same
`y[t]`. This lets one matrix multiply serve tokens of different ranks: compute up to some common
rank `R`, then zero each token's extra positions. Every kernel below relies on it.

---

## 3. How the baselines compute it

### 3.1 B0: the reference

A Python loop over tokens, in fp32. Slow and obviously correct. Every other implementation is
checked against it.

### 3.2 B1: plain PyTorch (pad and mask)

```
r_b  = max(r_t over the batch)
v    = x · A_t[0:r_b]ᵀ                      cuBLAS matmul, v written to GPU memory
v   *= (arange(r_b) < r_t[:, None])         four small kernels, each reading and writing v
y    = v · B[0:r_b]                          cuBLAS matmul
out += y                                     one more pass over the output (add mode)
```

It reads the shared weights once (one matrix), so it has no duplication. Its costs are about
7 kernel launches per call, `v` travelling through memory between them, and a separate pass to
add into the output. cuBLAS internally splits the long `h_in` sum of the shrink across programs
(split-K), which makes B1 fast at small batches.

It masks the activation `v`, not the weights: all requests share one `A_t` and one `B`, so the
weights can't differ per request within one matmul. Zeroing column k of row t in `v` removes rank
component k from token t only.

### 3.3 B3: vLLM's kernels (the Punica design), the way stock vLLM serves Matryoshka

vLLM's LoRA kernels descend from Punica's **SGMV** (segmented gather matrix-vector
multiplication). SGMV's idea: tokens that use different adapters can still share one kernel
launch. Tokens are grouped into contiguous *segments* by adapter, the kernel gives each segment
its own programs, and each segment *gathers* its own adapter's weights:

```
SGMV shrink: for each adapter segment s (in parallel):   V_s = X_s · A_sᵀ      (split along h_in)
SGMV expand: for each adapter segment s (in parallel):   Y_s += V_s · B_s      (split along h_out)
```

Punica splits the shrink along `h_in` because `r` is tiny and `h_in` is large (the same reason we
use split-K below), and splits the expand along the output columns.

For a Matryoshka adapter this design has one blind spot: **each rank must be its own adapter.**
vLLM's `LoRARequest` has no rank field, so ranks 1, 2, 4, …, 64 are registered as 7 adapters,
each stored **padded to `max_lora_rank`** (64) in vLLM's stacked buffers (`[7, 64, h_in]` and
`[7, h_out, 64]`). A batch with all 7 ranks forms 7 segments, and each segment reads its own
padded copy: the shared first rows are read 7 times, and the padding rows are read too.

```
stored by vLLM:      copy r1  [■□□□□□□□…□]   1 real row  + 63 zero rows
                     copy r2  [■■□□□□□□…□]
                     …
                     copy r64 [■■■■■■■■…■]  64 real rows
                     = 448 rows read when all ranks are in a batch

stored by RankFuse:  one adapter [■■■■■■■■…■]  64 rows, every rank is a prefix
```

### 3.4 B4: S-LoRA (not built yet)

S-LoRA also treats each rank as a separate adapter but stores each at its true size (127 rows
for ranks 1–64 instead of 448). It duplicates less than vLLM, but still more than one copy.

---

## 4. Just enough GPU vocabulary

* **HBM**: the GPU's main memory. Huge bandwidth once data streams, but each read takes
  hundreds of nanoseconds to arrive.
* **L2 cache**: about 50 MB on chip, much closer. Data read recently by any program is often
  served from here.
* **SM**: one of the 132 independent processors. A kernel's work is split into **programs**
  (Triton's word for thread blocks); each runs on one SM. The set of programs is the **grid**.
* **Tile**: the small block of a matrix one program loads and works on at a time.
* **`tl.dot`**: a small matrix multiply on the tensor cores, e.g. `[16 × 128] · [128 × 64]`.
* **Registers**: storage inside the SM; the fastest place to keep the intermediate `v`.
* **Latency-bound vs bandwidth-bound**: a kernel is latency-bound when it spends its time
  waiting for each read before the next can start; bandwidth-bound when it moves data as fast as
  memory allows. LoRA at decode sizes is latency-bound; at long prompts it is bandwidth-bound.
* **CUDA graph**: a recorded sequence of kernels replayed with one command, which removes the
  CPU cost of launching each kernel. vLLM records decode steps this way.

---

## 5. RF1, first version: one fused kernel (`rf1_single`)

The first RankFuse kernel fused the whole B1 chain into **one launch**: shrink, mask, scale and
expand, with `v` kept in registers.

### 5.1 Work split

The grid is `(token blocks, column ranges, output slices)`:

* a **token block** is `BM` consecutive tokens (16 at decode sizes, up to 64 for prompts);
* a **column range** is a stretch of output columns, chosen so there is about one program per SM.

At 32 tokens on a 4096 → 4096 projection: 2 token blocks × 64 column ranges = 128 programs.

### 5.2 What one program does

```
tokens = my BM tokens; ranks = their r_t; R = R_PAD (batch max rank, rounded up to 16/32/64…)
v = zeros[BM, R]                                     # in registers, fp32
for h in 0, 64, 128, … , h_in−64:                    # 64 steps in a row (BLOCK_H = 64)
    X = load x[tokens, h : h+64]                     # [BM × 64]
    Ā = load A_t[0:R, h : h+64]                      # [R  × 64]   rows of the ONE adapter
    v += X · Āᵀ                                      # tl.dot: [BM×64]·[64×R] → [BM×R]
v[t, k] = 0 where k ≥ r_t                            # rank mask
v *= scale(r_t)                                      # optional per-rank scale
for each 64-column tile c in my range:
    B̄ = load B[0:R, c]                               # [R × 64]
    out[tokens, c] (+)= v · B̄                        # tl.dot: [BM×R]·[R×64] → [BM×64]
```

### 5.3 What it saves, and why it was still slow

* **Saved:** one launch instead of about 7; `v` never touches memory; the adapter is one copy.
  Of all methods it read the fewest bytes from HBM.
* **Slow at decode:** the shrink loop is a chain of 64 dependent steps; each waits for its two
  loads. With few tokens there is one small program per SM, so nothing hides the waiting. The
  profiler showed under 1% of memory bandwidth in use. Worse, all 64 column-range programs of a
  token block repeat the same shrink (25 MB of L2 traffic at 32 tokens).
* **Measured** (4096 × 4096, 32 tokens): about 48 µs for the kernel, against about 26 µs for B1
  in a CUDA graph.

---

## 6. RF1, final version: split-K (`rf1_fused`)

The fix is the one cuBLAS and Punica use for the shrink: **split the long `h_in` sum across
programs that run at the same time, then add the pieces.** RF1 now runs two kernels.

### 6.1 Decisions made on the CPU before launching

| Decision | Rule | 32 tokens | 8192 tokens |
|---|---|---|---|
| Rank width `R_PAD` | batch max rank rounded up to 16, 32, 64, 128 or 256 (the tensor cores need ≥ 16) | 64 | 64 |
| Token block `BM` | 16 up to 128 tokens; 64 above (32 or 16 for r_max 128/256, to fit registers) | 16 | 64 |
| Token blocks `m_tiles` | `cdiv(T, BM)` | 2 | 128 |
| Slices `S` of `h_in` | enough shrink programs to cover the 132 SMs about once (`SMs // m_tiles`), between 1 and 16; at least 2 above 128 tokens | 16 | 2 |
| Slice length `K_PER` | `h_in / S`, rounded up to whole 128-wide chunks | 256 | 2048 |
| Pipeline | `BH = 128` inputs per step, 3 loads in flight (reduced if shared memory is short) | | |
| Workspace `ws` | `[S, T, R_PAD]` fp32 for the partial sums | 128 KB (stays in L2) | 4 MB |

### 6.2 Kernel 1: split shrink (`_rf1_shrink_splitk_kernel`)

Grid `(m_tiles, S, slices)`: one program per (token block, slice of `h_in`).

```
tokens = block m;  h_lo = k · K_PER
v = zeros[BM, R_PAD]
for h in h_lo, h_lo+128, … , h_lo+K_PER−128:        # only K_PER/128 steps: 2 at 32 tokens
    X = load x[tokens, h : h+128]                    # [BM × 128]   contiguous rows
    Ā = load A_t[0:R_PAD, h : h+128]                 # [R  × 128]   rows ≥ batch max load as 0
    v += X · Āᵀ                                      # tensor cores, fp32 accumulate
ws[k, tokens, :] = v                                 # partial sum for slice k
```

```
            h_in = 4096 inputs
x row  [=========================================]
        slice 0   slice 1   slice 2  …  slice 15       ← 16 programs per token block,
         256       256       256          256             all running at the same time
          │         │         │            │
          ▼         ▼         ▼            ▼
        ws[0]     ws[1]     ws[2]   …   ws[15]          partial v, [BM × 64] each
```

The longest chain of dependent loads drops from 64 steps to 2–16, and the shrink is computed
once per token block instead of once per column range.

### 6.3 Kernel 2: combine, mask, expand

Two versions, chosen by batch size. Both start the same way:

```
v = ws[0, tokens] + ws[1, tokens] + … + ws[S−1, tokens]      # sum the partials (from L2)
v[t, k] = 0 where k ≥ r_t                                    # rank mask
v *= scale(r_t)                                              # optional
v16 = v cast to fp16 for the tensor cores
```

The mask and scale commute with the sum because both act on each token separately, so applying
them after combining the slices is exact.

**2a. Up to 128 tokens (`_rf1_expand_splitk_kernel`).** Grid `(m_tiles, n_split, slices)` with
about 4 programs per SM. Each program owns a strip of output columns and walks it in 128-wide
tiles:

```
for each 128-column tile c in my strip:
    B̄ = load B[0:R_PAD, c]                     # [R × 128]   the ONE adapter's rows
    acc = v16 · B̄                              # [BM×R]·[R×128] → [BM×128]
    if add: acc += load out[tokens, c]
    store out[tokens, c] = acc
```

**2b. Above 128 tokens (`_rf1_expand_tile_kernel`).** Grid `(m_tiles, h_out/256, slices)`:
**one 64 × 256 output tile per program, no column loop**, 8 warps. At 8192 tokens this is 2048
programs, so many output writes are in flight at once. (A loop that ran only once still cost
about 30% here; hence a separate kernel.)

**Aligned stores.** Output slices (for fused q/k/v or gate/up) are described by a small table of
column offsets in memory, so the compiler cannot see that they are multiples of 16 and would
write the output one value at a time. When every slice width is a multiple of 16 (true for
single projections and vLLM's fused layouts) the kernel passes that fact with
`tl.multiple_of`, which enables wide vector stores. This alone took the expand at 8192 tokens
from 67 to 45 µs.

### 6.4 The two worked examples

```
32 tokens (decode)                         8192 tokens (prompt)
  kernel 1: 2 blocks × 16 slices = 32        kernel 1: 128 blocks × 2 slices = 256 programs
            programs, 2 steps each                     16 steps each
  ws: 16 × 32 × 64 fp32 = 128 KB             ws: 2 × 8192 × 64 fp32 = 4 MB
  kernel 2a: 2 × 32 = 64 programs,           kernel 2b: 128 × 16 = 2048 programs,
             one 128-col tile each                      one 64 × 256 tile each
  time 7.5 µs (B1: 16.8)                     time 61.8 µs (B1: 61.1)
```

### 6.5 Memory traffic, where RankFuse saves

Per call, RF1 reads from HBM:

```
weights:   R × (h_in + h_out) × 2 bytes       once, shared by every token and every rank
x:         T × h_in × 2 bytes
out:       T × h_out × 2 bytes written (and read, in add mode)
ws:        S × T × R × 4 bytes written and read, normally served from L2
```

Programs that re-read weight rows or `x` get them from L2, not HBM. Measured with Nsight
Compute (4096 × 4096, uniform ranks 1–64, add mode, one call):

| | B1 | **B3 (vLLM)** | **RF1** |
|---|---|---|---|
| 32 tokens: HBM read | 2.04 MB | **7.92 MB** | **1.72 MB** |
| 32 tokens: L2 traffic | 6.2 MB | 16.4 MB | 5.4 MB |
| 2048 tokens: HBM read | 54.1 MB | 42.0 MB | 36.7 MB |

B3's 7.9 MB is the 7 padded copies (7 × about 1 MB) plus token data; RF1's 1.7 MB is one copy
plus token data.

### 6.6 CUDA-graph safety

With the batch max rank passed in (`r_limit`), RF1 makes no CPU–GPU synchronisation and its only
allocations are the output (if not supplied) and the workspace, both allowed inside a capture.
The tests capture it in a graph and replay it on new inputs.

---

## 7. RF2: RF1 plus skipping unused ranks

RF1 computes every token up to the **batch's** highest rank. RF2 asks: inside a batch, can a
group of low-rank tokens skip the rows it does not need?

### 7.1 Buckets

The tensor cores need at least 16 rows, so RF2 works in **rank buckets** of 16, 32, 64, 128
and 256 rows. A token block whose highest rank is 8 runs at bucket 16; one whose highest rank is
40 runs at bucket 64. Each bucket is a separately compiled version of the tile code; a runtime
branch picks one.

### 7.2 First version: sort the tokens (`rf2_fused`)

```
order = argsort(token ranks)              # tokens of similar rank become neighbours
block m = tokens order[m·BM : (m+1)·BM]
R = bucket(max rank in block m)
shrink and expand as in RF1, but loading only rows 0 … R−1 of A_t and B
write each token's result back to its original row
```

Measured: no real gain. Reading tokens in sorted order scatters the loads, the sort costs 6–46 µs
per batch, and most time goes to reading `x` and writing `out`, which do not depend on rank.

A split-K version, `rf2_splitk`, has been removed: on rank-sorted input it forms the same blocks as
`rf2_segment`, plus a sort and index lookups. Rank order is now the batch's own, given to
`rf2_segment`. The 25–26 Oct result tables labelled "RF2 sorted", and the "RF2" columns of the A100
tables, came from that removed kernel.

### 7.3 Second version: segments without sorting (`rf2_segment`)

In a serving batch, the tokens of one request already sit next to each other and share a rank.
Segment RF2 keeps the **original token order** and uses blocks of consecutive tokens:

```
blocks = [(first token, length)] of BM consecutive tokens            (built once per step)
for block m (in both kernels):
    tokens = start_m … start_m + len_m − 1                            # contiguous
    R      = bucket(max rank among these tokens)
    if R == 0: write zeros (or nothing, in add mode)
    else: split-K shrink over rows 0 … R−1; combine; mask; expand with rows 0 … R−1
```

```
batch (one rank per request):  [ req A, rank 4 ......... ][ req B, rank 64 ........ ][ req C, rank 4 … ]
blocks of BM tokens:           [ R=16 ][ R=16 ][ R=16 ][ R=64 ][ R=64 ][ R=64 ][ R=16 ][ R=16 ] …
                               only blocks inside request B read all 64 rows
```

It is the same idea as Punica's SGMV segments, with two differences: the segments are by
**rank** instead of by adapter, and every segment reads a **prefix of the same weights** rather
than its own copy. An "aligned" variant whose blocks never cross a request boundary was also
tested; it matched the simple layout on prompts and is very slow at decode (one-token blocks),
so the simple layout is the one to keep.

### 7.4 Third version: one kernel pair per bucket (`rf2_bucketed`)

Segment RF2 still compiles every bucket into one kernel and branches at run time, so each
program is sized (registers, token block) for the largest bucket. Bucketed RF2 splits the batch
on the CPU instead, once per step, from the request ranks the scheduler already knows:

```
plan (CPU, once per step):  bucket(rank) of every token; token indices of each bucket, in order
for each bucket R present, heaviest first, each on its own CUDA stream:
    split-K shrink over rows 0 … R−1     (kernel compiled for exactly R rows, no branch)
    combine, mask, expand with rows 0 … R−1
rank-0 rows: zero-filled (or left alone in add mode)
join the streams back into the caller's stream          (all of it capturable in a CUDA graph)
```

Low buckets get larger token blocks (128 tokens for buckets 16 and 32 against 64 in RF1), since
their `v` tile is small. Running the buckets on concurrent streams is essential: the small
buckets fill the SMs the large one leaves idle; one after another they are 1.1–2.2× slower than
RF1.

Measured: 1.10–1.15× faster than RF1 on 4096 × 4096 prompt batches mixing low ranks with a
rank-64 request, about 2% better than segment RF2. It loses on small decode batches, when the
batch's highest rank is already low, on the 896-wide layers of Qwen2.5-0.5B (whole model
0.95–1.01× of RF1: the extra launches cost what the skipping saves) and badly in eager mode (up to
2× slower: stream fork/join and per-bucket launches from Python). `use_rf2` is the CPU-side rule
that picks it only where it paid: CUDA graphs on, more than 128 tokens, bucketed rank work at most
45% of RF1's, and at least 4096 tokens or 64 tokens per request. The rule was fitted on one sweep
and still needs checking on other shapes.

### 7.5 What RF2 saves, and the ceiling

RF2 saves the rank-dependent part of the work: tensor-core time and the weight rows read, for
blocks whose rank is below the batch maximum. It cannot save reading `x` or writing `out`. The
size of the rank-dependent part, measured on RF1 at 8192 tokens: 61.5 µs when the batch's
highest rank is 64, 49.1 µs when it is 2 (computed at 16). So the most any rank skipping can save
is about 20%, and about 10–18% for realistic mixed batches.

Measured (4096 × 4096, add mode not used):

| Batch | RF1 split-K | Segment RF2 | Gain |
|---|---|---|---|
| 15 requests × 512 at rank 4 + 1 × 512 at rank 64 | 61.4 µs | 55.9 | 1.10× |
| 32 × 256, mostly low ranks (max 64) | 61.3 | 55.2 | 1.11× |
| 16 × 512, mostly low ranks (max 8) | 50.1 | 52.0 | 0.96× |
| decode, 32 tokens | 7.5 | 7.4 | about equal |

RF2 helps (about 10%; bucketed RF2 10–15%) when a batch mixes many low-rank requests with a
high-rank one, and is neutral or slightly slower otherwise; the per-batch choice on the CPU
(`use_rf2`, section 7.4) avoids the loss.

---

## 8. What RankFuse saves, all in one table

| | B1 (PyTorch) | B3 (vLLM / Punica style) | RankFuse (RF1 / RF2) |
|---|---|---|---|
| Adapter copies stored | 1 | one per rank, padded (7 × 64 rows) | **1** |
| Weight rows read per batch, all 7 ranks present | 64 | 448 | **64** (RF2: fewer for low-rank blocks) |
| Kernels per call | 6–7 | 3 | **2** |
| Intermediate `v` | GPU memory, several round trips | fp32 buffer, zeroed then atomically added | registers + small fp32 workspace in L2 |
| Rank mask | 4 separate kernels | none needed (zeros in padded rows) | in registers |
| Add into base output | separate pass | inside the expand | inside the expand |
| Adapter slots per Matryoshka adapter (serving) | n/a | 7 | **1** |

Measured results so far:

| Experiment | Result |
|---|---|
| One layer, 4096 × 4096, GPU time | RF1 split-K 2.1–2.4× faster than B1 at 1–128 tokens, 1.3–1.7× at 256–2048, tie at 8192; 1.4–1.8× in add mode |
| One layer vs vLLM's kernels (B3), add mode | RF1 1.9–2.8× faster; 1.2–1.5× even when only one rank copy is used |
| Whole Qwen2.5-0.5B, CUDA graphs | whole forward pass 1.4× faster than with B1 and 1.1–1.3× faster than with vLLM's kernels; LoRA's share of a decode step 25–29% (B1 48–51%, B3 36–45%) |
| Whole model, eager | RF1 0.80–0.85× of B1, because its Python launch path costs about 108 µs per call |

### 8.1 Roofline worked example (`python -m rankfuse.roofline`)

Weights, for one decode request at each of ranks 4, 8, 16 and 32, h = 4096, fp16
(`tests/test_roofline.py` checks these numbers):

| Strategy | Weight rows read | Weight data |
|---|---|---|
| RankFuse / B1 (shared prefix) | 32 | 0.5 MiB |
| S-LoRA-style (separate adapters, true ranks) | 4+8+16+32 = 60 | 0.94 MiB |
| vLLM-style (separate adapters, padded to 32) | 4 × 32 = 128 | 2.0 MiB |

Each token also reads x and reads and writes y: 3 × 4096 × 2 B ≈ 24 KiB, whatever its rank.

* Small batches (a few to tens of tokens): weights dominate, so sharing one copy should show up clearly.
* Large batches (512 tokens ≈ 12 MiB of x and y traffic): weights are a small share and all
  strategies converge. The model puts the ideal fused kernel within about 1% of B1.
* At decode sizes the memory-bound lower bound is about 0.3 µs, roughly 10× less than one kernel
  launch, so the number of launches can decide results more than bytes or FLOPs do. B2 vs B2g
  measures this directly.

The modeling assumptions are at the top of `rankfuse/roofline.py`. The vLLM row assumes its kernel
reads the zero-padded rows.

---

## 9. Where this sits in Matryoshka LoRA serving

### 9.1 The serving stack

```
5  API              request: prompt + adapter (+ rank, added by RankFuse)
4  Scheduler        which requests run each step         ← reused from vLLM unchanged
3  Memory managers  KV cache; which adapters are on GPU  ← one slot per Matryoshka adapter
2  Batch metadata   per token: adapter id (+ rank)       ← RankFuse adds the rank array
1  Operators        the LoRA matmuls in every layer      ← RF1 / RF2   (this document)
```

RankFuse is the operator (layer 1) plus a thin integration into an existing engine (vLLM). It
does not replace the scheduler or the memory manager.

### 9.2 Stage 1: one Matryoshka adapter, mixed ranks (current)

* Every token uses the same `A_t` and `B`, each at its own rank: exactly the setting of RF1 and
  RF2 above.
* Integration into vLLM (next): a `rank` field on `LoRARequest`; every rank of an adapter shares
  one adapter id and therefore one slot; a per-token rank buffer filled each step next to vLLM's
  per-token adapter indices (and in the CUDA-graph capture hook); the rank in the prefix-cache
  key; LoRA layers calling RankFuse behind a flag; outputs checked token for token against stock
  vLLM.
* Wins: one copy read per batch (Claim 1), fewer kernels with the mask on chip (fusion),
  about 10% from rank skipping on mixed prompts (Claim 2).

### 9.3 Stage 2: many Matryoshka adapters, mixed ranks (later)

Each token carries `(adapter id, rank)`. The kernel becomes a two-level version of the same
design, which is where it meets Punica's SGMV directly:

```
weights stacked:  A_t[num_adapters, r_max, h_in],  B[num_adapters, r_max, h_out]
segments:         tokens grouped by adapter (as SGMV does), then by rank inside each adapter
per block:        adapter index a, rank bucket R
shrink:           split-K over h_in, reading rows 0 … R−1 of A_t[a]
expand:           rows 0 … R−1 of B[a], one tile per program for long prompts
```

The kernel inner loops stay as described here; only an adapter offset is added in front. The
biggest Stage 2 win does not depend on kernel speed at all: stock vLLM needs one slot per
(adapter, rank) pair, 7 per Matryoshka adapter, so only `max_loras / 7` adapters fit on the GPU
at once and the rest wait for loads from CPU memory. RankFuse needs one slot per adapter, so
seven times as many stay resident, with fewer evictions and less waiting. This is directly
comparable to S-LoRA's adapter-scaling results.

### 9.4 How the claims map to evidence

| Claim | Evidence | Still to do |
|---|---|---|
| 1. Existing systems duplicate the shared prefix; RankFuse loads it once | B3 vs RF1: 4.6× fewer HBM bytes at 32 tokens, 1.9–2.8× faster; whole model 1.1–1.3× | inside vLLM; Stage 2 capacity; B4 (S-LoRA) |
| Fusion removes launches and traffic | RF1 vs B1: up to 2.4× (one layer), 1.4× (whole model) | other model sizes |
| 2. Skipping unused ranks helps in some regimes | segment RF2 about 10% on mixed-rank prompts, neutral elsewhere | realistic serving mixes |
| End-to-end gains survive in a real server | whole-model 1.1–1.5× with graphs; LoRA is 8–63% of a vLLM step for Qwen2.5-0.5B | vLLM integration, serving runs, 7–8B model |

---

## 10. Limits and open items

* All kernel tuning is on 4096 × 4096; the model's other shapes (k/v 4096 → 1024, gate/up
  4096 → 14336, down 14336 → 4096, and Qwen2.5-0.5B's 896-wide ones) need a sweep.
* RF1's Python launch path (about 108 µs per call) must be slimmed before vLLM integration,
  because vLLM runs long prompts outside CUDA graphs.
* At decode the floor is about 5.8 µs per call, mostly the fixed cost of two kernel launches; a
  single-launch split-K (the last shrink program does the expand) could save about 2 µs.
* At 8192 tokens RF1's expand writes at about 38% of memory speed against cuBLAS's 60%; Hopper
  features (TMA, persistent programs) could close that gap.
* RF2's gain shows only on large shapes (4096-wide layers, long prompts); RF1 stays the default.

---

## Appendix: where the code is

| What | File |
|---|---|
| RF1 split-K (`rf1_fused`), RF1 single kernel (`rf1_single`), RF2 sorted single kernel (`rf2_fused`) | `rankfuse/kernels/rankfuse_fused.py` |
| Segment RF2 (`rf2_segment`, `rf2_segment_fw`, `rf2_probe_read`, `rf2_probe_branch`) | `rankfuse/kernels/rankfuse_rf2_segment.py` |
| Sorted-index RF2 (`rf2_index`, `rf2_index_fw`) | `rankfuse/kernels/rankfuse_rf2_index.py` |
| Bucketed RF2 (`rf2_bucketed`, `rf2_bucketed_tuned`, `rf2_bucketed_seq`, `use_rf2`) | `rankfuse/kernels/rankfuse_rf2_bucketed.py` |
| Per-bucket row-range RF2 (`rf2_ranges`, `rf2_ranges_seq`) | `rankfuse/kernels/rankfuse_rf2_ranges.py` |
| Per-width launch settings (`_config`) and the A100 table | `rankfuse/kernels/rankfuse_fused.py`, `rankfuse/kernels/tuned_a100.py`, `bench/tune_blocks.py` |
| First RF1 and RF2 kernels (`rf1_masked`, `rf2_ragged`), kept as ablations | `rankfuse/kernels/rankfuse_mask.py`, `rankfuse/kernels/rankfuse_ragged.py` |
| B0 reference, B1 padded (+ graphed) | `rankfuse/reference.py`, `rankfuse/baselines/padded.py` |
| B3 (vLLM kernels, padded rank copies, `b3_vllm`) | `rankfuse/baselines/vllm_ops.py` |
| One-layer sweep (eager or CUDA-graph timing) | `bench/run_sweep.py` |
| Whole-model harness and timing | `bench/model_harness.py`, `bench/model_forward.py` |
| LoRA share of a vLLM step | `bench/vllm_step_time.py` |
| Nsight Compute profiling | `scripts/ice/ncu_profile.sh`, `bench/ncu_driver.py` |
| vLLM 0.32 LoRA internals | `docs/VLLM_LORA_NOTES.md` |
