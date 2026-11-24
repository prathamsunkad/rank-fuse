"""RF1 and RF2 (fused): LoRA kernels that compute the shrink once per program.

    rf1_fused   pad every token to the batch maximum rank and mask per token (B1's math), on
                the split-K path of section 8 (two kernels)
    rf1_single  the same math in ONE kernel (sections 3-4), kept as the fusion ablation
    rf2_fused   rf1_single's one kernel, but tokens are sorted by rank and each token block stops
                at its own maximum rank, so low-rank blocks load and multiply fewer rank rows.
                Kept as an ablation; the current RF2 variants (rf2_segment, rf2_index,
                rf2_bucketed, rf2_ranges) run on the split-K path through `splitk_pass`

All are thin wrappers around `rankfuse_lora`, which also takes the options a serving engine
needs: adding into an existing output, several output slices in one launch (fused QKV or
gate/up projections), and a caller-supplied r_limit so the call is CUDA-graph capturable.

=====================================================================================
1. What this operator computes
=====================================================================================

For every token t with rank r_t (rankfuse/interface.py):

    y[t] = scale(r_t) * x[t] @ A_t[:r_t].T @ B[:r_t]

    x           [T, h_in]       one row per token (requests are already flattened)
    A_t         [r_max, h_in]   A transposed: the rank-r prefix is rows [:r], contiguous
    B           [r_max, h_out]  the rank-r prefix is rows [:r], contiguous
    token_ranks [T]             rank of each token; 0 means "no adapter", so y[t] = 0
    y           [T, h_out]      the LoRA delta (or out += delta with add=True)

The two matmuls are called the shrink (x @ A_t.T: h_in -> r numbers per token) and the
expand (v @ B: r -> h_out numbers per token). v = x @ A_t.T is the intermediate.

=====================================================================================
2. Strategy: the same math as B1, in one kernel
=====================================================================================

B1 (rankfuse/baselines/padded.py) runs this as a chain of PyTorch kernels:

    r_b = int(token_ranks.max())          host sync
    v   = x @ A_t[:r_b].T                 cuBLAS, v written to HBM
    v  *= arange(r_b) < ranks[:, None]    ~4 small kernels, v read and written each time
    y   = v @ B[:r_b]                     cuBLAS, v read back from HBM
    y  *= scale                           one more pass over y

rf1_fused keeps B1's math exactly: pad every token to the batch maximum rank, compute all of
those rank components, then zero the ones each token does not use (the "rank mask"). Zeroing
v[t, k] for k >= r_t removes rank component k from token t only, because
y[t, n] = sum_k v[t, k] * B[k, n]. So B1 vs rf1_fused measures fusion alone: one launch
instead of ~6-7, and v never leaves the chip.

RF2 adds rank skipping. In rf2_fused the host sorts token indices by rank, so each token block
holds tokens of similar rank. Each program reads its block's maximum rank and runs a version of
the tile compiled for the smallest rank bucket (16/32/64/128/256) that covers it. A block of
rank-8 tokens loads 16 rank rows of A_t and B instead of r_limit. The split-K RF2 variants
(section 8) skip the same way; RF1 vs RF2 measures the benefit of skipping unused ranks
(Claim 2).

=====================================================================================
3. Work split: why the shrink is computed once per program
=====================================================================================

The grid is (m_tiles, n_split, n_slices):
    * axis 0 picks a block of BM tokens,
    * axis 1 picks a contiguous range of cols_per_prog output columns,
    * axis 2 picks an output slice (1 for a plain projection, 3 for fused QKV).

Every output column of a token needs that token's whole v row, which needs a reduction over
all of h_in. So each program must compute v for its tokens before it can produce any output.
rankfuse_mask.py gives each program a single BN-wide column tile, which makes all
h_out / BN programs of a token block recompute the same v (64x at h_out = 4096).

Here each program computes v once, keeps it in registers, and then loops over every column
tile in its range, reusing v for each. v is therefore computed n_split times per token block,
and n_split is chosen as the smallest split that still gives about one program per SM:

    n_split = min(n_tiles, max(1, NUM_SMS // (m_tiles * n_slices)))

    decode,  T = 5,    BM = 16: m_tiles = 1,  n_split = 64 -> grid (1, 64, 1)   on 132 SMs
    prefill, T = 4096, BM = 64: m_tiles = 64, n_split = 2  -> grid (64, 2, 1)

Decode keeps a large split because a single token block would otherwise run on one SM; the
repeated shrinks there read x and A_t from L2, not HBM. Prefill already has enough token
blocks to fill the GPU, so v is computed only twice.

=====================================================================================
4. What one program does (_lora_tile)
=====================================================================================

    K1  read its program ids; skip if its column range starts past its slice's width
    K2  tok    = its BM token indices (rf2: read from the sorted order); row_ok masks padding
    K3  ranks  = token_ranks[tok] (0 for padding rows)
    K4  k      = 0..R-1 rank positions; k_ok masks k >= r_limit
            rf1: R = R_PAD for every block. rf2: R = smallest bucket >= the block's max rank,
            and a block whose ranks are all 0 only writes zeros (or nothing, with add=True)
    K5  shrink: v[BM, R] += x[tok, h chunk] @ A_t[k, h chunk].T over all of h_in (fp32)
    K6  rank mask: v[t, k] = 0 where k >= ranks[t]
    K7  scale: v[t, :] *= scale_table[ranks[t]]   ((s * v) @ B == s * (v @ B))
    K8  cast v to the input dtype for the tensor-core expand (B1's v is also fp16/bf16)
    K9  expand: for each BN-wide column tile in its range,
            acc = v @ B[k, cols]; with add=True acc += out[tok, cols]; store to out

HBM traffic is x, A_t[:r_limit], B[:r_limit] read and the output written (and read, with
add=True); every other read of x, A_t and B by a second program is usually served from L2.

=====================================================================================
5. Serving options (rankfuse_lora)
=====================================================================================

    add=True        out += delta in the same pass. In a model the delta is added to the base
                    projection's output; without this, a separate add kernel would read and
                    write the whole [T, h_out] output again.
    slice_sizes     A_t [S, r_max, h_in] and B [S, r_max, n_max] hold S adapters that share x
                    and write consecutive column ranges of out (vLLM fuses q/k/v and gate/up
                    into one projection). Slice s writes slice_sizes[s] columns starting at
                    sum(slice_sizes[:s]); B[s] rows are padded to n_max. Every slice is
                    handled in the same launch.
    out             may be a column view of a larger buffer: only out.stride(1) == 1 is needed.
    r_limit         batch maximum rank. Leave it None in benchmarks so the call pays the same
                    single host sync as B1. A serving engine passes it (the adapter's r_max
                    works), which removes every host sync.
    rank_sort       True: read tokens through a rank-sorted order. Alone, the one-kernel
                    rf2_fused; with rank_skip, rf2_index.
    order           with rank_sort: a precomputed rank-sorted token order (the scheduler can build
                    it once per step and reuse it across layers). Default: torch.argsort.
    rank_skip       RF2 on the split-K path: each token block runs at its own rank bucket, chosen
                    in the kernel (rf2_segment; rf2_index with rank_sort).
    full_width      with rank_skip: every block takes the batch's bucket through the same binary,
                    the control that isolates skipping (rf2_segment_fw, rf2_index_fw).
    rank_probe      1 or 2, RF1's split-K path only: add the rank read, or the read and one
                    branch, at one compiled width (rf2_probe_read, rf2_probe_branch; diagnostics).
    split_k         without rank_sort or rank_skip. None (default): the split-K path of section 8
                    with an automatic slice count; 0: the single-kernel path (rf1_single);
                    n >= 1: split-K with about n slices.

CUDA graphs: with r_limit passed, rankfuse_lora makes no host syncs and allocates nothing but
`out` (when not given) and rf2's sort. Call it once before capture so the kernels compile and
the cached scale and slice tables exist; capture and replay then work.

Host cost per eager call (split-K path). _splitk (every rankfuse_lora call on the split-K path:
RF1, rf2_segment, rf2_index, the probes, and rf2_ranges' per-run calls) caches its launch
settings per call shape (_SplitKPlan: the same _config / _splitk_shrink_tiles / _split_count rules
and the same arguments as splitk_pass) and the compiled kernels per Triton specialization, so a
repeated shape launches the two kernels directly instead of through Triton's Python launcher.
RF1Launcher is the serving form: checks at setup, r_limit from the caller, a shared workspace and
direct launches. Both give output bitwise identical to splitk_pass (same compiled kernels, same
arguments); bench/vllm_rf1_launch.py measures the host time.

=====================================================================================
6. Important variables
=====================================================================================

    r_limit        batch maximum rank (B1's r_b). Rank rows >= r_limit are never needed.
    R_PAD          r_limit rounded up to 16/32/64/128/256. tl.dot needs every dim >= 16, and
                   rounding keeps the number of compiled variants small. k_ok makes the
                   extra rows load as zero, so they contribute nothing.
    BM, BH, BN     tokens per program, h_in chunk per shrink step, columns per expand step.
    m_tiles        cdiv(T, BM) token blocks.
    n_split        column ranges per (token block, slice) = times each v is computed.
    cols_per_prog  columns owned by one program, a multiple of BN so no tile is shared.

Only block sizes, widths (R, R_PAD) and feature flags (HAS_SCALE, ADD_OUT, ZERO_FILL, ALIGNED,
LOOP, INDEXED, RANK_SKIP, RANK_PROBE) are tl.constexpr. T, h_in, n_max, r_limit, force_rank and
cols_per_prog are runtime arguments, so a new batch size does not recompile.

=====================================================================================
7. Assumptions
=====================================================================================

* One adapter per call (per slice); every token reads the same A_t and B.
* x, A_t, B contiguous and sharing one dtype (fp16, bf16, fp32) with out; token_ranks
  int32/int64.
* Every token rank is <= r_limit. When the caller passes r_limit this is trusted, not
  checked. A larger rank is silently truncated to r_limit in RF1; in the rank-skipping kernels
  (rf2_fused, RANK_SKIP) a block whose maximum rank is above R_PAD matches no width and writes
  nothing, so its rows keep whatever out held.
* Offsets are int32, so T * h and r_max * h must stay below 2**31.

=====================================================================================
8. Split-K path (rf1_fused)
=====================================================================================

Measured on an H200 (4096 x 4096, r_max 64), the single kernel loses to B1 in a CUDA graph
for two reasons:

    decode        each program reduces all of h_in alone, one BH chunk after another, with
                  about one program per SM: the kernel waits on a long chain of loads
    long prompts  each program expands a BM x (h_out / n_split) strip in BN-wide steps, again
                  with one program per SM: the output writes reach about a tenth of HBM bandwidth

The split-K path runs two kernels:

    _splitk_shrink_kernel  grid (m_tiles, S, n_slices). Program (m, k, s) reduces slice k
        of h_in (K_PER inputs, a whole number of BH chunks) for token block m and stores its
        partial v [BM, R] in fp32 to ws[s, k, tok, :]. The slices run in parallel, so the
        longest chain of dependent loads is K_PER / BH steps instead of h_in / BH. S is chosen
        so that m_tiles * S * n_slices is about SPLITK_WAVES x the SM count (at most
        MAX_SPLIT_K); cuBLAS splits B1's shrink the same way.
    _splitk_expand_kernel  LOOP: grid (m_tiles, n_split, n_slices). Each program sums the S
        partials of its tokens, applies the rank mask and the scale (both linear per token, so
        they commute with the sum), casts v and runs the expand over its column range, adding
        into out with add=True as before. n_split aims at EXPAND_WAVES programs per SM, so
        several programs share an SM and overlap their loads and stores.
        Not LOOP, above EXPAND_TILED_MIN_TOKENS tokens: grid (m_tiles, cdiv(h_out, BN),
        n_slices), every program owns ONE BM x EXPAND_TILED_BLOCK_N output tile and has no
        column loop (thousands of programs at long prompts, which keeps many output writes in
        flight). The shrink then uses at least MIN_SPLIT_K_TILED slices (on an H200: 4 at 2048
        tokens, 2 at 4096 and 8192).

The expand takes ALIGNED: when every slice width (hence every offset) is a multiple of 16, the
kernel is told so with tl.multiple_of, which lets the output stores be vectorized. Slice
offsets come from a table in memory, so the compiler cannot infer this; without the hint the
stores go out one element at a time (the expand at 8192 tokens took 67 instead of 45 us).
Single projections and vLLM's fused q/k/v and gate/up layouts are all aligned.

Every RF1 and RF2 variant on this path launches these same kernels through splitk_pass, so
slices, the hint and tile rules cannot differ between them. Launch settings (block size, warps,
BH, stages, tiled-expand BN) come from _config for the width a launch runs at: the default rules,
overridden by the tuned table (tuned_a100.py) on its GPU. RF1, rf2_segment and rf2_index look
them up at the batch's width; rf2_ranges and rf2_bucketed_tuned at each bucket's. Two settings
separate the variants: INDEXED (rf2_index reads tokens through a sorted index, rf2_bucketed each
bucket's tokens through a table) and the rank width (R_PAD for RF1; a fixed bucket per launch
for rf2_bucketed and rf2_ranges, which runs RF1 on each bucket's contiguous rows; with
RANK_SKIP, each block's bucket at run time, branching to a part compiled for that width).
rf2_segment_fw passes force_rank = r_limit, so each block takes the batch's bucket through the
same binary: rf2_segment vs rf2_segment_fw is the effect of skipping alone, and rf2_segment_fw
vs rf1_fused is the cost of the rank read and branch. rf2_probe_read and rf2_probe_branch split
that cost into steps at one compiled width: rf1_fused -> read -> one branch -> rf2_segment_fw, whose
last step is the extra compiled widths.

ws is [n_slices, S, T, R_PAD] fp32, allocated per call (inside a CUDA-graph capture it comes
from the graph's memory pool). At T = 32 with S = 16 it is 128 KiB and stays in L2. Rank-0
rows need no special case: their v is masked to zero. The shrink's pipeline depth, then BH,
shrink when one stage of x and A_t tiles would not fit in shared memory (fp32 inputs,
r_max 128-256, GPUs with less shared memory).

=====================================================================================
9. Multi-adapter mode (MULTI)
=====================================================================================

With MULTI=True the same two kernels serve many adapters (rankfuse/multi.py, RF1-MA and RF2-MA in
rankfuse/kernels/rankfuse_multi.py). splitk_pass(..., multi=MultiTiles) launches them:

    grid axis 0  one program row per entry of a per-step tile table (adapter, first position,
                 length, group rank limit, first token), sized to its fixed capacity so one
                 captured CUDA graph serves every step (MultiMeta.load_); empty entries exit
    tokens       position p of a tile is token order[p] (index_ptr), or first + p when the tile's
                 tokens are consecutive in the batch; partials go to ws row p as before, outputs to
                 each token's own row
    weights      A_t and B are page pools: rank row k of adapter a, slice s is at
                 frame(a, k // 16) * FRAME + off[s] + (k % 16) * LD (int32 arithmetic unless OFF64),
                 with per-slice offsets from a [S, 5] table, so merged q/k/v keep a separate A each
    ranks        rows >= the tile's group limit are masked (r_limit is unused); RANK_SKIP branches on
                 max(limit, force_rank) to the 16/32/64/128/256-wide part
    scale        scale_table is [adapters, SCALE_LD] (scale of adapter a at rank r)

The MULTI=False path is Tony's code unchanged: the two kernels call _splitk_shrink_single /
_splitk_expand_single (his kernel bodies, inlined by Triton), and the multi-only parameters default
to None / False, so the single-adapter specializations compile to the same code as before.
"""

import functools
import os
from typing import NamedTuple

import torch
import triton
import triton.language as tl
from triton import knobs
from triton.runtime import driver

from rankfuse.interface import MatryoshkaLoRA, register
from rankfuse.kernels._common import SUPPORTED_DTYPES, check_layout, prepare_r_limit, prepare_rank_order

BLOCK_H = 64
BLOCK_N = 64
MAX_R_PAD = 256

# Split-K path (section 8). WAVES, EXPAND_BLOCK_N and EXPAND_WARPS: best geometric-mean
# GPU time against B1 over T = 1..8192 in a graph-replay sweep on an H200 (4096 x 4096, r_max 64).
SPLITK_BLOCK_H = 128  # h_in chunk per shrink step
SPLITK_STAGES = 3  # loads in flight in the shrink loop
SPLITK_WAVES = 1  # shrink programs per SM to aim for when choosing the slice count
MAX_SPLIT_K = 16
EXPAND_BLOCK_N = 128
EXPAND_WAVES = 4  # expand programs per SM to aim for
EXPAND_WARPS = 4
# Above EXPAND_TILED_MIN_TOKENS the expand runs one output tile per program (no column loop),
# with wider tiles and more warps: about 2x faster than the strip layout at 1024-8192 tokens.
EXPAND_TILED_MIN_TOKENS = 128
EXPAND_TILED_BLOCK_N = 256
EXPAND_TILED_WARPS = 8
MIN_SPLIT_K_TILED = 2  # at least this many shrink slices above EXPAND_TILED_MIN_TOKENS
USE_ALIGNED_HINT = True  # bench/rf1_ablation.py turns it off to measure it


@triton.jit
def _lora_tile(
    x_ptr,
    a_ptr,
    b_ptr,
    scale_tab_ptr,
    out_ptr,
    tok,
    row_ok,
    ranks,
    H_IN,
    N_MAX,
    OUT_STRIDE,
    out_col0,
    n_cols,
    r_limit,
    col_base,
    cols_per_prog,
    BM: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
    R: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    ADD_OUT: tl.constexpr,
):
    # K4: rank positions up to this tile's bucket; rows >= r_limit load as zero.
    k = tl.arange(0, R)
    k_ok = k < r_limit

    # K5: shrink, once per program. Each step loads x[tok, h] [BM, BH] and A_t[k, h] [R, BH];
    # both are contiguous along h, so the loads are coalesced.
    v = tl.zeros((BM, R), dtype=tl.float32)
    for h_start in range(0, H_IN, BH):
        h = h_start + tl.arange(0, BH)
        h_ok = h < H_IN
        x = tl.load(
            x_ptr + tok[:, None] * H_IN + h[None, :],
            mask=row_ok[:, None] & h_ok[None, :],
            other=0.0,
        )
        a = tl.load(
            a_ptr + k[:, None] * H_IN + h[None, :],
            mask=k_ok[:, None] & h_ok[None, :],
            other=0.0,
        )
        v += tl.dot(x, tl.trans(a), input_precision="ieee")

    # K6: B1's rank mask, applied in registers instead of ~4 kernels over v in HBM.
    v = tl.where(k[None, :] < ranks[:, None], v, 0.0)

    # K7: per-token scale on v (one small tile) instead of on every output tile.
    if HAS_SCALE:
        scale = tl.load(scale_tab_ptr + ranks, mask=row_ok, other=0.0)
        v = v * scale[:, None]

    # K8: tensor-core inputs must share B's dtype; B1 rounds v to this dtype too.
    v_in = v.to(out_ptr.dtype.element_ty)

    # K9: expand. cols_per_prog is a multiple of BN, so a tile never crosses into the next
    # program's range and cols < n_cols is the only bound to check.
    for col_start in range(0, cols_per_prog, BN):
        cols = col_base + col_start + tl.arange(0, BN)
        col_ok = cols < n_cols
        b = tl.load(
            b_ptr + k[:, None] * N_MAX + cols[None, :],
            mask=k_ok[:, None] & col_ok[None, :],
            other=0.0,
        )
        acc = tl.dot(v_in, b, input_precision="ieee")
        ptrs = out_ptr + tok[:, None] * OUT_STRIDE + out_col0 + cols[None, :]
        mask = row_ok[:, None] & col_ok[None, :]
        if ADD_OUT:
            acc += tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        tl.store(ptrs, acc, mask=mask)


@triton.jit
def _zero_tile(
    out_ptr,
    tok,
    row_ok,
    OUT_STRIDE,
    out_col0,
    n_cols,
    col_base,
    cols_per_prog,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    # A block whose tokens all have rank 0 has an all-zero delta.
    zeros = tl.zeros((BM, BN), dtype=out_ptr.dtype.element_ty)
    for col_start in range(0, cols_per_prog, BN):
        cols = col_base + col_start + tl.arange(0, BN)
        tl.store(
            out_ptr + tok[:, None] * OUT_STRIDE + out_col0 + cols[None, :],
            zeros,
            mask=row_ok[:, None] & (cols[None, :] < n_cols),
        )


@triton.jit
def _rf1_fused_kernel(
    x_ptr,
    a_ptr,
    b_ptr,
    ranks_ptr,
    scale_tab_ptr,
    out_ptr,
    slice_off_ptr,
    slice_size_ptr,
    T,
    H_IN,
    N_MAX,
    OUT_STRIDE,
    r_limit,
    cols_per_prog,
    A_SLICE,
    B_SLICE,
    BM: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
    R_PAD: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    ADD_OUT: tl.constexpr,
):
    # K1: which slice and column range; programs past a narrow slice's width have no work.
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(2)
    out_col0 = tl.load(slice_off_ptr + pid_s)
    n_cols = tl.load(slice_size_ptr + pid_s)
    col_base = tl.program_id(1) * cols_per_prog
    if col_base < n_cols:
        # K2-K3: this program's tokens, in batch order, and their ranks.
        rows = pid_m * BM + tl.arange(0, BM)
        row_ok = rows < T
        ranks = tl.load(ranks_ptr + rows, mask=row_ok, other=0)
        _lora_tile(
            x_ptr, a_ptr + pid_s * A_SLICE, b_ptr + pid_s * B_SLICE, scale_tab_ptr, out_ptr,
            rows, row_ok, ranks, H_IN, N_MAX, OUT_STRIDE, out_col0, n_cols, r_limit,
            col_base, cols_per_prog,
            BM=BM, BH=BH, BN=BN, R=R_PAD, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT,
        )


@triton.jit
def _rf2_fused_kernel(
    x_ptr,
    a_ptr,
    b_ptr,
    ranks_ptr,
    order_ptr,
    scale_tab_ptr,
    out_ptr,
    slice_off_ptr,
    slice_size_ptr,
    T,
    H_IN,
    N_MAX,
    OUT_STRIDE,
    r_limit,
    cols_per_prog,
    A_SLICE,
    B_SLICE,
    BM: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
    R_PAD: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    ADD_OUT: tl.constexpr,
    ZERO_FILL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(2)
    out_col0 = tl.load(slice_off_ptr + pid_s)
    n_cols = tl.load(slice_size_ptr + pid_s)
    col_base = tl.program_id(1) * cols_per_prog
    if col_base < n_cols:
        # K2-K3: this program's tokens come from the rank-sorted order, so the block holds
        # tokens of similar rank; outputs are written back to each token's own row.
        pos = pid_m * BM + tl.arange(0, BM)
        row_ok = pos < T
        tok = tl.load(order_ptr + pos, mask=row_ok, other=0)
        ranks = tl.load(ranks_ptr + tok, mask=row_ok, other=0)
        tile_rank = tl.max(ranks, axis=0)
        a_s = a_ptr + pid_s * A_SLICE
        b_s = b_ptr + pid_s * B_SLICE

        # K4: run the tile compiled for the smallest rank bucket covering this block. The
        # R_PAD checks are compile-time, so buckets above the batch's bucket are not built.
        if tile_rank == 0:
            if ZERO_FILL:
                _zero_tile(out_ptr, tok, row_ok, OUT_STRIDE, out_col0, n_cols, col_base,
                           cols_per_prog, BM=BM, BN=BN)
        elif tile_rank <= 16:
            _lora_tile(x_ptr, a_s, b_s, scale_tab_ptr, out_ptr, tok, row_ok, ranks, H_IN, N_MAX,
                       OUT_STRIDE, out_col0, n_cols, r_limit, col_base, cols_per_prog,
                       BM=BM, BH=BH, BN=BN, R=16, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT)
        elif tile_rank <= 32:
            if R_PAD >= 32:
                _lora_tile(x_ptr, a_s, b_s, scale_tab_ptr, out_ptr, tok, row_ok, ranks, H_IN,
                           N_MAX, OUT_STRIDE, out_col0, n_cols, r_limit, col_base, cols_per_prog,
                           BM=BM, BH=BH, BN=BN, R=32, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT)
        elif tile_rank <= 64:
            if R_PAD >= 64:
                _lora_tile(x_ptr, a_s, b_s, scale_tab_ptr, out_ptr, tok, row_ok, ranks, H_IN,
                           N_MAX, OUT_STRIDE, out_col0, n_cols, r_limit, col_base, cols_per_prog,
                           BM=BM, BH=BH, BN=BN, R=64, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT)
        elif tile_rank <= 128:
            if R_PAD >= 128:
                _lora_tile(x_ptr, a_s, b_s, scale_tab_ptr, out_ptr, tok, row_ok, ranks, H_IN,
                           N_MAX, OUT_STRIDE, out_col0, n_cols, r_limit, col_base, cols_per_prog,
                           BM=BM, BH=BH, BN=BN, R=128, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT)
        else:
            if R_PAD >= 256:
                _lora_tile(x_ptr, a_s, b_s, scale_tab_ptr, out_ptr, tok, row_ok, ranks, H_IN,
                           N_MAX, OUT_STRIDE, out_col0, n_cols, r_limit, col_base, cols_per_prog,
                           BM=BM, BH=BH, BN=BN, R=256, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT)


@triton.jit
def _splitk_shrink_part(x_ptr, a_s, ws_k, tok, pos, row_ok, H_IN, r_limit, h_lo, K_PER,
                        BM: tl.constexpr, BH: tl.constexpr, R: tl.constexpr, R_PAD: tl.constexpr):
    # Reduce one h_in slice over R rank rows; the partial v goes to the workspace.
    k = tl.arange(0, R)
    k_ok = k < r_limit
    v = tl.zeros((BM, R), dtype=tl.float32)
    for h_start in range(0, K_PER, BH):
        h = h_lo + h_start + tl.arange(0, BH)
        h_ok = h < H_IN
        x = tl.load(x_ptr + tok[:, None] * H_IN + h[None, :], mask=row_ok[:, None] & h_ok[None, :], other=0.0)
        a = tl.load(a_s + k[:, None] * H_IN + h[None, :], mask=k_ok[:, None] & h_ok[None, :], other=0.0)
        v += tl.dot(x, tl.trans(a), input_precision="ieee")
    tl.store(ws_k + pos[:, None] * R_PAD + k[None, :], v, mask=row_ok[:, None])


# ---- Multi-adapter mode (MULTI=True; see section 9) -------------------------------------------------
# Tokens come from a per-step tile table and weights from a page pool. Every helper below is used only
# under MULTI, so the single-adapter specializations (MULTI=False) compile exactly as before.


@triton.jit
def _frame_rows(pt_ptr, adapter, PT_LD, k, k_ok, FRAME, off, LD, ALIGNED: tl.constexpr, OFF64: tl.constexpr):
    # Element offset of rank row k of this adapter's matrix: frame(k // 16) * FRAME + off + (k % 16) * LD.
    frame = tl.load(pt_ptr + adapter * PT_LD + k // 16, mask=k_ok, other=0)
    if OFF64:
        rows = frame.to(tl.int64) * FRAME + off + (k % 16) * LD
    else:  # every offset fits in int32 (pools below 2**31 elements): cheaper address arithmetic
        rows = frame * FRAME + off.to(tl.int32) + (k % 16) * LD.to(tl.int32)
    if ALIGNED:
        # FRAME, off and LD are multiples of 16 (checked on the host): lets a row's loads be vectorized.
        rows = tl.multiple_of(rows, 16)
    return rows


@triton.jit
def _tile_tokens(tiles_ptr, tile, order_ptr, pos, row_ok, BM: tl.constexpr):
    # Token indices of a tile. A tile whose tokens are consecutive in the batch carries its first
    # token (column 4) and skips the dependent load of `order`.
    first = tl.load(tiles_ptr + tile * 5 + 4)
    if first >= 0:
        tok = first + tl.arange(0, BM)
    else:
        tok = tl.load(order_ptr + pos, mask=row_ok, other=0)
    return tok


@triton.jit
def _splitk_shrink_part_multi(x_ptr, a_pool, pt_ptr, adapter, PT_LD, FRAME, a_off, ws_k, tok, pos, row_ok, H_IN, limit,
                              h_lo, K_PER, BM: tl.constexpr, BH: tl.constexpr, R: tl.constexpr, R_PAD: tl.constexpr,
                              ALIGNED: tl.constexpr, OFF64: tl.constexpr):
    # _splitk_shrink_part over this tile's adapter rows (through the page table), up to its group limit.
    k = tl.arange(0, R)
    k_ok = k < limit
    a_rows = _frame_rows(pt_ptr, adapter, PT_LD, k, k_ok, FRAME, a_off, H_IN, ALIGNED, OFF64)
    v = tl.zeros((BM, R), dtype=tl.float32)
    for h_start in range(0, K_PER, BH):
        h = h_lo + h_start + tl.arange(0, BH)
        h_ok = h < H_IN
        x = tl.load(x_ptr + tok[:, None] * H_IN + h[None, :], mask=row_ok[:, None] & h_ok[None, :], other=0.0)
        a = tl.load(a_pool + a_rows[:, None] + h[None, :], mask=k_ok[:, None] & h_ok[None, :], other=0.0)
        v += tl.dot(x, tl.trans(a), input_precision="ieee")
    tl.store(ws_k + pos[:, None] * R_PAD + k[None, :], v, mask=row_ok[:, None])


@triton.jit
def _splitk_store_tile_multi(v_in, b_pool, b_rows, k_ok, out_ptr, tok, row_ok, col_start, n_cols, out_col0, OUT_STRIDE,
                             BN: tl.constexpr, ADD_OUT: tl.constexpr):
    cols = col_start + tl.arange(0, BN)
    col_ok = cols < n_cols
    b = tl.load(b_pool + b_rows[:, None] + cols[None, :], mask=k_ok[:, None] & col_ok[None, :], other=0.0)
    acc = tl.dot(v_in, b, input_precision="ieee")
    ptrs = out_ptr + tok[:, None] * OUT_STRIDE + out_col0 + cols[None, :]
    mask = row_ok[:, None] & col_ok[None, :]
    if ADD_OUT:
        acc += tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    tl.store(ptrs, acc, mask=mask)


@triton.jit
def _splitk_expand_part_multi(ws_s, scale_ptr, b_pool, pt_ptr, adapter, PT_LD, FRAME, b_off, b_ld, SCALE_LD, out_ptr, tok,
                              pos, row_ok, ranks, limit, T, OUT_STRIDE, SPLIT_K, out_col0, n_cols, col_base, cols_per_prog,
                              BM: tl.constexpr, BN: tl.constexpr, R: tl.constexpr, R_PAD: tl.constexpr,
                              HAS_SCALE: tl.constexpr, ADD_OUT: tl.constexpr, LOOP: tl.constexpr, ALIGNED: tl.constexpr,
                              OFF64: tl.constexpr):
    # _splitk_expand_part with the tile's group limit, a per-adapter scale table and paged B rows.
    k = tl.arange(0, R)
    k_ok = k < limit
    v = tl.zeros((BM, R), dtype=tl.float32)
    for split in range(0, SPLIT_K):
        v += tl.load(ws_s + split * T * R_PAD + pos[:, None] * R_PAD + k[None, :], mask=row_ok[:, None], other=0.0)
    v = tl.where(k[None, :] < ranks[:, None], v, 0.0)
    if HAS_SCALE:
        v = v * tl.load(scale_ptr + adapter * SCALE_LD + ranks, mask=row_ok, other=0.0)[:, None]
    v_in = v.to(out_ptr.dtype.element_ty)
    b_rows = _frame_rows(pt_ptr, adapter, PT_LD, k, k_ok, FRAME, b_off, b_ld, ALIGNED, OFF64)
    if LOOP:
        for col_start in range(0, cols_per_prog, BN):
            _splitk_store_tile_multi(v_in, b_pool, b_rows, k_ok, out_ptr, tok, row_ok, col_base + col_start, n_cols,
                                     out_col0, OUT_STRIDE, BN=BN, ADD_OUT=ADD_OUT)
    else:
        _splitk_store_tile_multi(v_in, b_pool, b_rows, k_ok, out_ptr, tok, row_ok, col_base, n_cols, out_col0,
                                 OUT_STRIDE, BN=BN, ADD_OUT=ADD_OUT)


@triton.jit
def _splitk_shrink_multi(x_ptr, a_ptr, order_ptr, ws_ptr, ws_row0, T, H_IN, force_rank, K_PER, WS_SLICE, tiles_ptr, pt_ptr,
                         mslice_ptr, PT_LD, FRAME, BM: tl.constexpr, BH: tl.constexpr, R: tl.constexpr,
                         R_PAD: tl.constexpr, RANK_SKIP: tl.constexpr, ALIGNED: tl.constexpr, OFF64: tl.constexpr):
    # Program (tile, k, s): tile = (adapter, first position, length, group limit, first token).
    tile = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_s = tl.program_id(2)
    limit = tl.load(tiles_ptr + tile * 5 + 3)
    if limit > 0:  # empty tiles and the no-LoRA group have limit 0
        adapter = tl.load(tiles_ptr + tile * 5)
        start = tl.load(tiles_ptr + tile * 5 + 1)
        length = tl.load(tiles_ptr + tile * 5 + 2)
        pos = start + tl.arange(0, BM)
        row_ok = tl.arange(0, BM) < length
        tok = _tile_tokens(tiles_ptr, tile, order_ptr, pos, row_ok, BM=BM)
        a_off = tl.load(mslice_ptr + pid_s * 5 + 2)
        h_lo = pid_k * K_PER
        ws_k = ws_ptr + pid_s * WS_SLICE + (pid_k * T + ws_row0) * R_PAD
        if RANK_SKIP:
            tile_rank = tl.maximum(limit, force_rank)
            if tile_rank <= 16:
                _splitk_shrink_part_multi(x_ptr, a_ptr, pt_ptr, adapter, PT_LD, FRAME, a_off, ws_k, tok, pos, row_ok, H_IN, limit, h_lo, K_PER, BM=BM, BH=BH, R=16, R_PAD=R_PAD, ALIGNED=ALIGNED, OFF64=OFF64)
            elif tile_rank <= 32:
                if R_PAD >= 32:
                    _splitk_shrink_part_multi(x_ptr, a_ptr, pt_ptr, adapter, PT_LD, FRAME, a_off, ws_k, tok, pos, row_ok, H_IN, limit, h_lo, K_PER, BM=BM, BH=BH, R=32, R_PAD=R_PAD, ALIGNED=ALIGNED, OFF64=OFF64)
            elif tile_rank <= 64:
                if R_PAD >= 64:
                    _splitk_shrink_part_multi(x_ptr, a_ptr, pt_ptr, adapter, PT_LD, FRAME, a_off, ws_k, tok, pos, row_ok, H_IN, limit, h_lo, K_PER, BM=BM, BH=BH, R=64, R_PAD=R_PAD, ALIGNED=ALIGNED, OFF64=OFF64)
            elif tile_rank <= 128:
                if R_PAD >= 128:
                    _splitk_shrink_part_multi(x_ptr, a_ptr, pt_ptr, adapter, PT_LD, FRAME, a_off, ws_k, tok, pos, row_ok, H_IN, limit, h_lo, K_PER, BM=BM, BH=BH, R=128, R_PAD=R_PAD, ALIGNED=ALIGNED, OFF64=OFF64)
            else:
                if R_PAD >= 256:
                    _splitk_shrink_part_multi(x_ptr, a_ptr, pt_ptr, adapter, PT_LD, FRAME, a_off, ws_k, tok, pos, row_ok, H_IN, limit, h_lo, K_PER, BM=BM, BH=BH, R=256, R_PAD=R_PAD, ALIGNED=ALIGNED, OFF64=OFF64)
        else:
            _splitk_shrink_part_multi(x_ptr, a_ptr, pt_ptr, adapter, PT_LD, FRAME, a_off, ws_k, tok, pos, row_ok, H_IN, limit, h_lo, K_PER, BM=BM, BH=BH, R=R, R_PAD=R_PAD, ALIGNED=ALIGNED, OFF64=OFF64)


@triton.jit
def _splitk_expand_multi(ws_ptr, ranks_ptr, order_ptr, scale_ptr, b_ptr, out_ptr, slice_off_ptr, slice_size_ptr, ws_row0, T,
                         OUT_STRIDE, force_rank, cols_per_prog, SPLIT_K, WS_SLICE, tiles_ptr, pt_ptr, mslice_ptr, PT_LD,
                         FRAME, SCALE_LD, BM: tl.constexpr, BN: tl.constexpr, R: tl.constexpr, R_PAD: tl.constexpr,
                         HAS_SCALE: tl.constexpr, ADD_OUT: tl.constexpr, ALIGNED: tl.constexpr, LOOP: tl.constexpr,
                         RANK_SKIP: tl.constexpr, OFF64: tl.constexpr):
    tile = tl.program_id(0)
    pid_s = tl.program_id(2)
    out_col0 = tl.load(slice_off_ptr + pid_s)
    n_cols = tl.load(slice_size_ptr + pid_s)
    if ALIGNED:
        out_col0 = tl.multiple_of(out_col0, 16)
        n_cols = tl.multiple_of(n_cols, 16)
    if LOOP:
        col_base = tl.program_id(1) * cols_per_prog
    else:
        col_base = tl.program_id(1) * BN
    length = tl.load(tiles_ptr + tile * 5 + 2)
    if length > 0:  # rows past the step's tile count are empty
        if col_base < n_cols:
            adapter = tl.load(tiles_ptr + tile * 5)
            start = tl.load(tiles_ptr + tile * 5 + 1)
            limit = tl.load(tiles_ptr + tile * 5 + 3)
            pos = start + tl.arange(0, BM)
            row_ok = tl.arange(0, BM) < length
            tok = _tile_tokens(tiles_ptr, tile, order_ptr, pos, row_ok, BM=BM)
            if limit == 0:
                if not ADD_OUT:
                    _splitk_zero_part(out_ptr, tok, row_ok, n_cols, out_col0, OUT_STRIDE, col_base, cols_per_prog, BM=BM, BN=BN, LOOP=LOOP)
            else:
                ranks = tl.load(ranks_ptr + tok, mask=row_ok, other=0)
                b_off = tl.load(mslice_ptr + pid_s * 5 + 3)
                b_ld = tl.load(mslice_ptr + pid_s * 5 + 4)
                ws_s = ws_ptr + pid_s * WS_SLICE + ws_row0 * R_PAD
                if RANK_SKIP:
                    tile_rank = tl.maximum(limit, force_rank)
                    if tile_rank <= 16:
                        _splitk_expand_part_multi(ws_s, scale_ptr, b_ptr, pt_ptr, adapter, PT_LD, FRAME, b_off, b_ld, SCALE_LD, out_ptr, tok, pos, row_ok, ranks, limit, T, OUT_STRIDE, SPLIT_K, out_col0, n_cols, col_base, cols_per_prog,
                                                  BM=BM, BN=BN, R=16, R_PAD=R_PAD, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT, LOOP=LOOP, ALIGNED=ALIGNED, OFF64=OFF64)
                    elif tile_rank <= 32:
                        if R_PAD >= 32:
                            _splitk_expand_part_multi(ws_s, scale_ptr, b_ptr, pt_ptr, adapter, PT_LD, FRAME, b_off, b_ld, SCALE_LD, out_ptr, tok, pos, row_ok, ranks, limit, T, OUT_STRIDE, SPLIT_K, out_col0, n_cols, col_base, cols_per_prog,
                                                      BM=BM, BN=BN, R=32, R_PAD=R_PAD, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT, LOOP=LOOP, ALIGNED=ALIGNED, OFF64=OFF64)
                    elif tile_rank <= 64:
                        if R_PAD >= 64:
                            _splitk_expand_part_multi(ws_s, scale_ptr, b_ptr, pt_ptr, adapter, PT_LD, FRAME, b_off, b_ld, SCALE_LD, out_ptr, tok, pos, row_ok, ranks, limit, T, OUT_STRIDE, SPLIT_K, out_col0, n_cols, col_base, cols_per_prog,
                                                      BM=BM, BN=BN, R=64, R_PAD=R_PAD, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT, LOOP=LOOP, ALIGNED=ALIGNED, OFF64=OFF64)
                    elif tile_rank <= 128:
                        if R_PAD >= 128:
                            _splitk_expand_part_multi(ws_s, scale_ptr, b_ptr, pt_ptr, adapter, PT_LD, FRAME, b_off, b_ld, SCALE_LD, out_ptr, tok, pos, row_ok, ranks, limit, T, OUT_STRIDE, SPLIT_K, out_col0, n_cols, col_base, cols_per_prog,
                                                      BM=BM, BN=BN, R=128, R_PAD=R_PAD, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT, LOOP=LOOP, ALIGNED=ALIGNED, OFF64=OFF64)
                    else:
                        if R_PAD >= 256:
                            _splitk_expand_part_multi(ws_s, scale_ptr, b_ptr, pt_ptr, adapter, PT_LD, FRAME, b_off, b_ld, SCALE_LD, out_ptr, tok, pos, row_ok, ranks, limit, T, OUT_STRIDE, SPLIT_K, out_col0, n_cols, col_base, cols_per_prog,
                                                      BM=BM, BN=BN, R=256, R_PAD=R_PAD, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT, LOOP=LOOP, ALIGNED=ALIGNED, OFF64=OFF64)
                else:
                    _splitk_expand_part_multi(ws_s, scale_ptr, b_ptr, pt_ptr, adapter, PT_LD, FRAME, b_off, b_ld, SCALE_LD, out_ptr, tok, pos, row_ok, ranks, limit, T, OUT_STRIDE, SPLIT_K, out_col0, n_cols, col_base, cols_per_prog,
                                              BM=BM, BN=BN, R=R, R_PAD=R_PAD, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT, LOOP=LOOP, ALIGNED=ALIGNED, OFF64=OFF64)



# force_rank is left out of specialization: rank skipping and its full-width control run the
# same compiled binary, so the rank width taken at run time is their only difference.
@triton.jit(do_not_specialize=["force_rank"])
def _splitk_shrink_kernel(
    x_ptr,
    a_ptr,
    ranks_ptr,
    index_ptr,
    ws_ptr,
    count,
    ws_row0,
    T,
    H_IN,
    r_limit,
    force_rank,
    K_PER,
    A_SLICE,
    WS_SLICE,
    BM: tl.constexpr,
    BH: tl.constexpr,
    R: tl.constexpr,
    R_PAD: tl.constexpr,
    INDEXED: tl.constexpr,
    RANK_SKIP: tl.constexpr,
    RANK_PROBE: tl.constexpr,
    tiles_ptr=None,
    pt_ptr=None,
    mslice_ptr=None,
    PT_LD=None,
    FRAME=None,
    MULTI: tl.constexpr = False,
    ALIGNED: tl.constexpr = False,
    OFF64: tl.constexpr = False,
):
    if MULTI:
        # Section 9: tiles from the tile table (index_ptr is the token order), paged A rows.
        _splitk_shrink_multi(x_ptr, a_ptr, index_ptr, ws_ptr, ws_row0, T, H_IN, force_rank, K_PER, WS_SLICE, tiles_ptr, pt_ptr,
                             mslice_ptr, PT_LD, FRAME, BM=BM, BH=BH, R=R, R_PAD=R_PAD, RANK_SKIP=RANK_SKIP,
                             ALIGNED=ALIGNED, OFF64=OFF64)
    else:
        _splitk_shrink_single(x_ptr, a_ptr, ranks_ptr, index_ptr, ws_ptr, count, ws_row0, T, H_IN, r_limit, force_rank, K_PER,
                              A_SLICE, WS_SLICE, BM=BM, BH=BH, R=R, R_PAD=R_PAD, INDEXED=INDEXED, RANK_SKIP=RANK_SKIP,
                              RANK_PROBE=RANK_PROBE)


@triton.jit
def _splitk_shrink_single(x_ptr, a_ptr, ranks_ptr, index_ptr, ws_ptr, count, ws_row0, T, H_IN, r_limit, force_rank, K_PER,
                          A_SLICE, WS_SLICE, BM: tl.constexpr, BH: tl.constexpr, R: tl.constexpr, R_PAD: tl.constexpr,
                          INDEXED: tl.constexpr, RANK_SKIP: tl.constexpr, RANK_PROBE: tl.constexpr):
    # Section 8, phase 1: token block m, h_in slice k, output slice s.
    pid_k = tl.program_id(1)
    pid_s = tl.program_id(2)
    pos = tl.program_id(0) * BM + tl.arange(0, BM)
    row_ok = pos < count
    if INDEXED:
        tok = tl.load(index_ptr + pos, mask=row_ok, other=0)
    else:
        tok = pos
    a_s = a_ptr + pid_s * A_SLICE
    h_lo = pid_k * K_PER
    ws_k = ws_ptr + pid_s * WS_SLICE + (pid_k * T + ws_row0) * R_PAD
    if RANK_SKIP:
        ranks = tl.load(ranks_ptr + tok, mask=row_ok, other=0)
        tile_rank = tl.maximum(tl.max(ranks, axis=0).to(tl.int32), force_rank)
        if tile_rank > 0:
            if tile_rank <= 16:
                _splitk_shrink_part(x_ptr, a_s, ws_k, tok, pos, row_ok, H_IN, r_limit, h_lo, K_PER, BM=BM, BH=BH, R=16, R_PAD=R_PAD)
            elif tile_rank <= 32:
                if R_PAD >= 32:
                    _splitk_shrink_part(x_ptr, a_s, ws_k, tok, pos, row_ok, H_IN, r_limit, h_lo, K_PER, BM=BM, BH=BH, R=32, R_PAD=R_PAD)
            elif tile_rank <= 64:
                if R_PAD >= 64:
                    _splitk_shrink_part(x_ptr, a_s, ws_k, tok, pos, row_ok, H_IN, r_limit, h_lo, K_PER, BM=BM, BH=BH, R=64, R_PAD=R_PAD)
            elif tile_rank <= 128:
                if R_PAD >= 128:
                    _splitk_shrink_part(x_ptr, a_s, ws_k, tok, pos, row_ok, H_IN, r_limit, h_lo, K_PER, BM=BM, BH=BH, R=128, R_PAD=R_PAD)
            else:
                if R_PAD >= 256:
                    _splitk_shrink_part(x_ptr, a_s, ws_k, tok, pos, row_ok, H_IN, r_limit, h_lo, K_PER, BM=BM, BH=BH, R=256, R_PAD=R_PAD)
    elif RANK_PROBE == 1:
        ranks = tl.load(ranks_ptr + tok, mask=row_ok, other=0)
        kept = row_ok & (tl.max(ranks, axis=0) >= 0)  # always true; uses the read without a branch
        _splitk_shrink_part(x_ptr, a_s, ws_k, tok, pos, kept, H_IN, r_limit, h_lo, K_PER, BM=BM, BH=BH, R=R, R_PAD=R_PAD)
    elif RANK_PROBE == 2:
        ranks = tl.load(ranks_ptr + tok, mask=row_ok, other=0)
        if tl.maximum(tl.max(ranks, axis=0).to(tl.int32), force_rank) > 0:
            _splitk_shrink_part(x_ptr, a_s, ws_k, tok, pos, row_ok, H_IN, r_limit, h_lo, K_PER, BM=BM, BH=BH, R=R, R_PAD=R_PAD)
    else:
        _splitk_shrink_part(x_ptr, a_s, ws_k, tok, pos, row_ok, H_IN, r_limit, h_lo, K_PER, BM=BM, BH=BH, R=R, R_PAD=R_PAD)


@triton.jit
def _splitk_store_tile(v_in, b_s, out_ptr, tok, row_ok, k, k_ok, col_start, n_cols, out_col0, N_MAX, OUT_STRIDE,
                       BN: tl.constexpr, ADD_OUT: tl.constexpr):
    cols = col_start + tl.arange(0, BN)
    col_ok = cols < n_cols
    b = tl.load(b_s + k[:, None] * N_MAX + cols[None, :], mask=k_ok[:, None] & col_ok[None, :], other=0.0)
    acc = tl.dot(v_in, b, input_precision="ieee")
    ptrs = out_ptr + tok[:, None] * OUT_STRIDE + out_col0 + cols[None, :]
    mask = row_ok[:, None] & col_ok[None, :]
    if ADD_OUT:
        acc += tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    tl.store(ptrs, acc, mask=mask)


@triton.jit
def _splitk_expand_part(ws_s, scale_tab_ptr, b_s, out_ptr, tok, pos, row_ok, ranks, T, N_MAX, OUT_STRIDE, r_limit, SPLIT_K,
                        out_col0, n_cols, col_base, cols_per_prog, BM: tl.constexpr, BN: tl.constexpr, R: tl.constexpr,
                        R_PAD: tl.constexpr, HAS_SCALE: tl.constexpr, ADD_OUT: tl.constexpr, LOOP: tl.constexpr):
    # Sum the partials, mask, scale, expand over R rank rows.
    k = tl.arange(0, R)
    k_ok = k < r_limit
    v = tl.zeros((BM, R), dtype=tl.float32)
    for split in range(0, SPLIT_K):
        v += tl.load(ws_s + split * T * R_PAD + pos[:, None] * R_PAD + k[None, :], mask=row_ok[:, None], other=0.0)
    v = tl.where(k[None, :] < ranks[:, None], v, 0.0)
    if HAS_SCALE:
        scale = tl.load(scale_tab_ptr + ranks, mask=row_ok, other=0.0)
        v = v * scale[:, None]
    v_in = v.to(out_ptr.dtype.element_ty)
    if LOOP:
        for col_start in range(0, cols_per_prog, BN):
            _splitk_store_tile(v_in, b_s, out_ptr, tok, row_ok, k, k_ok, col_base + col_start, n_cols, out_col0, N_MAX,
                               OUT_STRIDE, BN=BN, ADD_OUT=ADD_OUT)
    else:
        # A loop that runs once still cost ~30% of the expand at 8192 tokens.
        _splitk_store_tile(v_in, b_s, out_ptr, tok, row_ok, k, k_ok, col_base, n_cols, out_col0, N_MAX,
                           OUT_STRIDE, BN=BN, ADD_OUT=ADD_OUT)


@triton.jit
def _splitk_zero_part(out_ptr, tok, row_ok, n_cols, out_col0, OUT_STRIDE, col_base, cols_per_prog,
                      BM: tl.constexpr, BN: tl.constexpr, LOOP: tl.constexpr):
    # A block whose ranks are all 0, when writing rather than adding.
    zeros = tl.zeros((BM, BN), dtype=out_ptr.dtype.element_ty)
    if LOOP:
        for col_start in range(0, cols_per_prog, BN):
            cols = col_base + col_start + tl.arange(0, BN)
            tl.store(out_ptr + tok[:, None] * OUT_STRIDE + out_col0 + cols[None, :], zeros, mask=row_ok[:, None] & (cols[None, :] < n_cols))
    else:
        cols = col_base + tl.arange(0, BN)
        tl.store(out_ptr + tok[:, None] * OUT_STRIDE + out_col0 + cols[None, :], zeros, mask=row_ok[:, None] & (cols[None, :] < n_cols))


@triton.jit(do_not_specialize=["force_rank"])
def _splitk_expand_kernel(
    ws_ptr,
    ranks_ptr,
    index_ptr,
    scale_tab_ptr,
    b_ptr,
    out_ptr,
    slice_off_ptr,
    slice_size_ptr,
    count,
    ws_row0,
    T,
    N_MAX,
    OUT_STRIDE,
    r_limit,
    force_rank,
    cols_per_prog,
    SPLIT_K,
    B_SLICE,
    WS_SLICE,
    BM: tl.constexpr,
    BN: tl.constexpr,
    R: tl.constexpr,
    R_PAD: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    ADD_OUT: tl.constexpr,
    ALIGNED: tl.constexpr,
    LOOP: tl.constexpr,
    INDEXED: tl.constexpr,
    RANK_SKIP: tl.constexpr,
    RANK_PROBE: tl.constexpr,
    tiles_ptr=None,
    pt_ptr=None,
    mslice_ptr=None,
    PT_LD=None,
    FRAME=None,
    SCALE_LD=None,
    MULTI: tl.constexpr = False,
    OFF64: tl.constexpr = False,
):
    if MULTI:
        # Section 9: tiles from the tile table (index_ptr is the token order), paged B rows,
        # per-adapter scales (scale_tab_ptr is [adapters, SCALE_LD]).
        _splitk_expand_multi(ws_ptr, ranks_ptr, index_ptr, scale_tab_ptr, b_ptr, out_ptr, slice_off_ptr, slice_size_ptr, ws_row0, T,
                             OUT_STRIDE, force_rank, cols_per_prog, SPLIT_K, WS_SLICE, tiles_ptr, pt_ptr, mslice_ptr, PT_LD,
                             FRAME, SCALE_LD, BM=BM, BN=BN, R=R, R_PAD=R_PAD, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT,
                             ALIGNED=ALIGNED, LOOP=LOOP, RANK_SKIP=RANK_SKIP, OFF64=OFF64)
    else:
        _splitk_expand_single(ws_ptr, ranks_ptr, index_ptr, scale_tab_ptr, b_ptr, out_ptr, slice_off_ptr, slice_size_ptr, count,
                              ws_row0, T, N_MAX, OUT_STRIDE, r_limit, force_rank, cols_per_prog, SPLIT_K, B_SLICE, WS_SLICE,
                              BM=BM, BN=BN, R=R, R_PAD=R_PAD, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT, ALIGNED=ALIGNED,
                              LOOP=LOOP, INDEXED=INDEXED, RANK_SKIP=RANK_SKIP, RANK_PROBE=RANK_PROBE)


@triton.jit
def _splitk_expand_single(ws_ptr, ranks_ptr, index_ptr, scale_tab_ptr, b_ptr, out_ptr, slice_off_ptr, slice_size_ptr, count,
                          ws_row0, T, N_MAX, OUT_STRIDE, r_limit, force_rank, cols_per_prog, SPLIT_K, B_SLICE, WS_SLICE,
                          BM: tl.constexpr, BN: tl.constexpr, R: tl.constexpr, R_PAD: tl.constexpr, HAS_SCALE: tl.constexpr,
                          ADD_OUT: tl.constexpr, ALIGNED: tl.constexpr, LOOP: tl.constexpr, INDEXED: tl.constexpr,
                          RANK_SKIP: tl.constexpr, RANK_PROBE: tl.constexpr):
    # Section 8, phase 2. With LOOP each program walks a column strip; without it (above
    # EXPAND_TILED_MIN_TOKENS) each program owns one BM x BN output tile.
    pid_s = tl.program_id(2)
    out_col0 = tl.load(slice_off_ptr + pid_s)
    n_cols = tl.load(slice_size_ptr + pid_s)
    if ALIGNED:
        # Offsets and widths are multiples of 16 (checked on the host): tell the compiler, so
        # the output stores can be vectorized. Without it they are issued one element at a time.
        out_col0 = tl.multiple_of(out_col0, 16)
        n_cols = tl.multiple_of(n_cols, 16)
    if LOOP:
        col_base = tl.program_id(1) * cols_per_prog
    else:
        col_base = tl.program_id(1) * BN
    if col_base < n_cols:
        pos = tl.program_id(0) * BM + tl.arange(0, BM)
        row_ok = pos < count
        if INDEXED:
            tok = tl.load(index_ptr + pos, mask=row_ok, other=0)
        else:
            tok = pos
        ranks = tl.load(ranks_ptr + tok, mask=row_ok, other=0)
        ws_s = ws_ptr + pid_s * WS_SLICE + ws_row0 * R_PAD
        b_s = b_ptr + pid_s * B_SLICE
        if RANK_SKIP:
            tile_rank = tl.maximum(tl.max(ranks, axis=0).to(tl.int32), force_rank)
            if tile_rank == 0:
                if not ADD_OUT:
                    _splitk_zero_part(out_ptr, tok, row_ok, n_cols, out_col0, OUT_STRIDE, col_base, cols_per_prog, BM=BM, BN=BN, LOOP=LOOP)
            elif tile_rank <= 16:
                _splitk_expand_part(ws_s, scale_tab_ptr, b_s, out_ptr, tok, pos, row_ok, ranks, T, N_MAX, OUT_STRIDE, r_limit, SPLIT_K, out_col0, n_cols,
                                    col_base, cols_per_prog, BM=BM, BN=BN, R=16, R_PAD=R_PAD, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT, LOOP=LOOP)
            elif tile_rank <= 32:
                if R_PAD >= 32:
                    _splitk_expand_part(ws_s, scale_tab_ptr, b_s, out_ptr, tok, pos, row_ok, ranks, T, N_MAX, OUT_STRIDE, r_limit, SPLIT_K, out_col0, n_cols,
                                        col_base, cols_per_prog, BM=BM, BN=BN, R=32, R_PAD=R_PAD, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT, LOOP=LOOP)
            elif tile_rank <= 64:
                if R_PAD >= 64:
                    _splitk_expand_part(ws_s, scale_tab_ptr, b_s, out_ptr, tok, pos, row_ok, ranks, T, N_MAX, OUT_STRIDE, r_limit, SPLIT_K, out_col0, n_cols,
                                        col_base, cols_per_prog, BM=BM, BN=BN, R=64, R_PAD=R_PAD, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT, LOOP=LOOP)
            elif tile_rank <= 128:
                if R_PAD >= 128:
                    _splitk_expand_part(ws_s, scale_tab_ptr, b_s, out_ptr, tok, pos, row_ok, ranks, T, N_MAX, OUT_STRIDE, r_limit, SPLIT_K, out_col0, n_cols,
                                        col_base, cols_per_prog, BM=BM, BN=BN, R=128, R_PAD=R_PAD, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT, LOOP=LOOP)
            else:
                if R_PAD >= 256:
                    _splitk_expand_part(ws_s, scale_tab_ptr, b_s, out_ptr, tok, pos, row_ok, ranks, T, N_MAX, OUT_STRIDE, r_limit, SPLIT_K, out_col0, n_cols,
                                        col_base, cols_per_prog, BM=BM, BN=BN, R=256, R_PAD=R_PAD, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT, LOOP=LOOP)
        elif RANK_PROBE == 1:
            kept = row_ok & (tl.max(ranks, axis=0) >= 0)  # always true; uses the maximum without a branch
            _splitk_expand_part(ws_s, scale_tab_ptr, b_s, out_ptr, tok, pos, kept, ranks, T, N_MAX, OUT_STRIDE, r_limit, SPLIT_K, out_col0, n_cols,
                                col_base, cols_per_prog, BM=BM, BN=BN, R=R, R_PAD=R_PAD, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT, LOOP=LOOP)
        elif RANK_PROBE == 2:
            if tl.maximum(tl.max(ranks, axis=0).to(tl.int32), force_rank) == 0:
                if not ADD_OUT:
                    _splitk_zero_part(out_ptr, tok, row_ok, n_cols, out_col0, OUT_STRIDE, col_base, cols_per_prog, BM=BM, BN=BN, LOOP=LOOP)
            else:
                _splitk_expand_part(ws_s, scale_tab_ptr, b_s, out_ptr, tok, pos, row_ok, ranks, T, N_MAX, OUT_STRIDE, r_limit, SPLIT_K, out_col0, n_cols,
                                    col_base, cols_per_prog, BM=BM, BN=BN, R=R, R_PAD=R_PAD, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT, LOOP=LOOP)
        else:
            _splitk_expand_part(ws_s, scale_tab_ptr, b_s, out_ptr, tok, pos, row_ok, ranks, T, N_MAX, OUT_STRIDE, r_limit, SPLIT_K, out_col0, n_cols,
                                col_base, cols_per_prog, BM=BM, BN=BN, R=R, R_PAD=R_PAD, HAS_SCALE=HAS_SCALE, ADD_OUT=ADD_OUT, LOOP=LOOP)


@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


_SLICE_TABLES: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _slice_tables(sizes: tuple[int, ...], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Column offset and width of each output slice, built once per layout and device."""
    key = (sizes, device)
    if key not in _SLICE_TABLES:
        offsets = [sum(sizes[:s]) for s in range(len(sizes))]
        _SLICE_TABLES[key] = (
            torch.tensor(offsets, dtype=torch.int32, device=device),
            torch.tensor(sizes, dtype=torch.int32, device=device),
        )
    return _SLICE_TABLES[key]


def _scale_table(lora: MatryoshkaLoRA, device: torch.device) -> torch.Tensor | None:
    """scale_table[r] = scale for rank r, built once per adapter and device, then reused.

    Serving code would build this when the adapter loads. Rebuilding it per call (as
    token_scales does) costs several tiny launches.
    """
    if lora.scale_per_rank is None:
        return None
    key = (tuple(sorted(lora.scale_per_rank.items())), device)
    cached = getattr(lora, "_rf1_scale_table", None)
    if cached is None or cached[0] != key:
        table = torch.zeros(lora.r_max + 1, dtype=torch.float32)
        for r, s in lora.scale_per_rank.items():
            table[r] = s
        cached = (key, table.to(device))
        lora._rf1_scale_table = cached
    return cached[1]


# Per-width launch settings (bench/tune_blocks.py). TUNED maps (width, token range) to settings
# that override the defaults below; it is filled from rankfuse/kernels/tuned_a100.py only on the
# GPU it was tuned on, so other GPUs keep the default rules. Every split-K kernel looks settings up
# here: RF1, rf2_segment and rf2_index at the batch's width, per-bucket kernels at each bucket's.
TOKEN_RANGES = (128, 1024, 4096)  # upper bounds of ranges 0..2; range 3 is everything above
TUNED: dict[tuple[int, int], dict] = {}
TUNED_GPU = ""
if os.environ.get("RANKFUSE_TUNED", "1") != "0":  # RANKFUSE_TUNED=0: default rules (before/after runs)
    try:
        from rankfuse.kernels.tuned_a100 import GPU as TUNED_GPU, TUNED  # noqa: F811
    except ImportError:
        pass


def _token_range(num_tokens: int) -> int:
    for i, hi in enumerate(TOKEN_RANGES):
        if num_tokens <= hi:
            return i
    return len(TOKEN_RANGES)


@functools.lru_cache(maxsize=None)
def _gpu_name(device_index: int) -> str:
    return torch.cuda.get_device_name(device_index)


def _config(width: int, num_tokens: int, device_index: int = 0, bm: int | None = None, bn: int | None = None) -> dict:
    """Launch settings for one group of num_tokens tokens at `width` rank rows.

    Keys: bm (tokens per program), shrink_warps, bh and stages (None: the shrink-tile rule),
    bn and expand_warps for the tiled expand (None: EXPAND_TILED_BLOCK_N / _WARPS), loop_expand
    (True: the column-loop expand even above EXPAND_TILED_MIN_TOKENS). bm and bn are the caller's
    defaults when there is no tuned entry (None: _block_m and the module rule). Derived key: tiled,
    whether the expand runs one tile per program (above EXPAND_TILED_MIN_TOKENS, unless loop_expand)."""
    cfg = dict(bm=_block_m(num_tokens, width) if bm is None else bm, shrink_warps=4, bh=None, stages=None,
               bn=bn, expand_warps=None, loop_expand=False)
    if TUNED and _gpu_name(device_index) == TUNED_GPU:
        cfg.update(TUNED.get((width, _token_range(num_tokens)), {}))
    cfg["tiled"] = num_tokens > EXPAND_TILED_MIN_TOKENS and not cfg["loop_expand"]
    return cfg


def _r_pad(r_limit: int) -> int:
    """R_PAD: r_limit rounded up to a power of two, at least 16 (triton.next_power_of_2 is a
    constexpr_function in Triton 3.7 and costs ~2 us per host call)."""
    return 16 if r_limit <= 16 else 1 << (r_limit - 1).bit_length()


def _block_m(num_tokens: int, r_pad: int) -> int:
    # Small token blocks for decode; larger ones for prefill, but keep v [BM, R_PAD] in fp32
    # registers at <= 32 KiB per program.
    if num_tokens <= 128:
        return 16
    if r_pad <= 64:
        return 64
    return 32 if r_pad <= 128 else 16


@functools.lru_cache(maxsize=None)
def _max_shared_mem(device_index: int) -> int:
    return triton.runtime.driver.active.utils.get_device_properties(device_index)["max_shared_mem"]


def _splitk_shrink_tiles(block_m: int, r_pad: int, elem_size: int, device_index: int,
                         bh: int | None = None, stages: int | None = None) -> tuple[int, int]:
    """(BH, num_stages) for the split-K shrink: SPLITK_BLOCK_H and SPLITK_STAGES (or the given
    values), reduced while the x and A_t tiles of all pipeline stages would exceed ~70% of
    shared memory."""
    budget = int(0.7 * _max_shared_mem(device_index))
    bh = SPLITK_BLOCK_H if bh is None else bh
    stages = SPLITK_STAGES if stages is None else stages
    while stages > 2 and (block_m + r_pad) * bh * elem_size * stages > budget:
        stages -= 1
    while bh > 32 and (block_m + r_pad) * bh * elem_size * stages > budget:
        bh //= 2
    return bh, stages


def _split_count(h_in: int, tiled: bool, grid_tiles: int, n_slices: int, num_sms: int, bh: int,
                 split_k: int | None = None) -> tuple[int, int]:
    """(S, K_PER): shrink slices of h_in in whole BH chunks, for grid_tiles token blocks."""
    if split_k is None:
        split_k = max(MIN_SPLIT_K_TILED if tiled else 1, min(MAX_SPLIT_K, SPLITK_WAVES * num_sms // (grid_tiles * n_slices)))
    k_per = triton.cdiv(triton.cdiv(h_in, split_k), bh) * bh  # whole BH chunks per slice
    return triton.cdiv(h_in, k_per), k_per  # no empty slices


def splitk_pass(x, A_t, B, token_ranks, out, add, sizes, scale_table, ws, r_limit, r_pad, split_k, k_per, block_m, bh, stages,
                *, index=None, count=None, ws_row0=0, grid_tiles=None, rank=None, rank_skip=False, rank_probe=0, force_rank=0, block_n=None,
                shrink_warps=4, expand_warps=None, tiled=None, multi=None):
    """Launch the two split-K kernels for one group of tokens; every RF1 and RF2 variant comes here.

    multi: a MultiTiles (section 9) for the multi-adapter mode; A_t and B are then the weight pools.

    Tokens: the batch in order (index None), or index[:count] through a lookup, with partials in
    workspace rows ws_row0 + position. Rank width: R_PAD; or `rank` rows for this group; or, with
    rank_skip, each block's own bucket at run time, never below force_rank. rank_probe keeps one
    width and adds part of the rank-skipping path, to measure it: 1 the rank read, 2 also one
    branch. grid_tiles is the token
    block count across all groups launched together; block_n and expand_warps override the tiled
    expand's tile and warps, shrink_warps the shrink's warps, tiled the expand mode (None: tiled
    above EXPAND_TILED_MIN_TOKENS; all from _config).
    """
    if multi is not None:
        return _splitk_pass_multi(x, A_t, B, token_ranks, out, add, sizes, scale_table, ws, r_limit, r_pad, split_k, k_per,
                                  block_m, bh, stages, multi, ws_row0=ws_row0, grid_tiles=grid_tiles, rank=rank,
                                  rank_skip=rank_skip, force_rank=force_rank, block_n=block_n, shrink_warps=shrink_warps,
                                  expand_warps=expand_warps, tiled=tiled)
    num_tokens, h_in = x.shape
    n_slices, r_max, _ = A_t.shape
    n_max = B.shape[2]
    device_index = x.device.index or 0
    count = num_tokens if count is None else count
    m_tiles = triton.cdiv(count, block_m)
    ws_slice = split_k * num_tokens * r_pad
    r = r_pad if rank is None else rank
    # When a table goes unused, another pointer stands in for it.
    index_ptr = token_ranks if index is None else index
    _splitk_shrink_kernel[(m_tiles, split_k, n_slices)](
        x, A_t, token_ranks, index_ptr, ws, count, ws_row0, num_tokens, h_in, r_limit, force_rank, k_per, r_max * h_in, ws_slice,
        BM=block_m, BH=bh, R=r, R_PAD=r_pad, INDEXED=index is not None, RANK_SKIP=rank_skip, RANK_PROBE=rank_probe, num_warps=shrink_warps, num_stages=stages,
    )

    widest = max(sizes)
    offsets, widths = _slice_tables(sizes, x.device)
    # When there is no scale, out stands in for the unused table pointer.
    scales = out if scale_table is None else scale_table
    aligned = USE_ALIGNED_HINT and all(s % 16 == 0 for s in sizes)
    if tiled is None:
        tiled = num_tokens > EXPAND_TILED_MIN_TOKENS
    if tiled:
        # One tile per program; narrow the tile while B's [R_PAD, BN] tile would take more
        # than ~35% of shared memory (fp32, r_max 128-256, smaller GPUs).
        guard_r = r_pad if block_n is None else r
        block_n = EXPAND_TILED_BLOCK_N if block_n is None else block_n
        while block_n > 64 and guard_r * block_n * x.element_size() > 0.35 * _max_shared_mem(device_index):
            block_n //= 2
        cols_per_prog, warps = block_n, EXPAND_TILED_WARPS if expand_warps is None else expand_warps
    else:
        block_n, warps = EXPAND_BLOCK_N, EXPAND_WARPS
        n_tiles = triton.cdiv(widest, block_n)
        tiles = m_tiles if grid_tiles is None else grid_tiles
        n_split = min(n_tiles, max(1, EXPAND_WAVES * _num_sms(device_index) // (tiles * n_slices)))
        cols_per_prog = triton.cdiv(n_tiles, n_split) * block_n
    _splitk_expand_kernel[(m_tiles, triton.cdiv(widest, cols_per_prog), n_slices)](
        ws, token_ranks, index_ptr, scales, B, out, offsets, widths, count, ws_row0,
        num_tokens, n_max, out.stride(0), r_limit, force_rank, cols_per_prog, split_k, r_max * n_max, ws_slice,
        BM=block_m, BN=block_n, R=r, R_PAD=r_pad, HAS_SCALE=scale_table is not None, ADD_OUT=add,
        ALIGNED=aligned, LOOP=not tiled, INDEXED=index is not None, RANK_SKIP=rank_skip, RANK_PROBE=rank_probe, num_warps=warps, num_stages=2,
    )
    return out


class MultiTiles(NamedTuple):
    """What splitk_pass needs for the multi-adapter mode (section 9); rankfuse.kernels.rankfuse_multi
    builds it from a MultiMeta and a PagedWeights.

        tiles        [capacity, 5] int32: adapter, first position, length, group limit, first token
        order        [T] int32: position -> token
        page_table   [adapters, pt_ld] int32 frames
        slice_table  [S, 5] int64: out_col0, n_cols, a_off, b_off, b_ld per output slice
        frame_a/_b   elements between frames of the A / B pool
        scale_ld     row length of the per-adapter scale table (r_max + 1)
        aligned      every pool offset and slice width is a multiple of 16 (PagedWeights.aligned)
        off64        64-bit offsets (a pool of 2**31 elements or more)
    """

    tiles: torch.Tensor
    order: torch.Tensor
    page_table: torch.Tensor
    slice_table: torch.Tensor
    pt_ld: int
    frame_a: int
    frame_b: int
    scale_ld: int
    n_slices: int
    aligned: bool
    off64: bool

    @property
    def capacity(self) -> int:
        return self.tiles.shape[0]


def _splitk_pass_multi(x, A_pool, B_pool, token_ranks, out, add, sizes, scale_table, ws, r_limit, r_pad, split_k, k_per,
                       block_m, bh, stages, multi: MultiTiles, *, ws_row0=0, grid_tiles=None, rank=None, rank_skip=False,
                       force_rank=0, block_n=None, shrink_warps=4, expand_warps=None, tiled=None):
    """splitk_pass's multi-adapter mode: one program row per tile-table entry (the table's capacity, so
    the grid is the same every step) and the expand tile rules of splitk_pass, with the tile count as
    the token-block count. ws is [n_slices, split_k, T, r_pad] fp32 as in the single-adapter mode."""
    num_tokens, h_in = x.shape
    n_slices = multi.n_slices
    device_index = x.device.index or 0
    cap = multi.capacity
    ws_slice = split_k * num_tokens * r_pad
    r = r_pad if rank is None else rank
    aligned = USE_ALIGNED_HINT and multi.aligned
    tables = dict(tiles_ptr=multi.tiles, pt_ptr=multi.page_table, mslice_ptr=multi.slice_table, PT_LD=multi.pt_ld)
    _splitk_shrink_kernel[(cap, split_k, n_slices)](
        x, A_pool, token_ranks, multi.order, ws, num_tokens, ws_row0, num_tokens, h_in, r_limit, force_rank, k_per, 0, ws_slice,
        BM=block_m, BH=bh, R=r, R_PAD=r_pad, INDEXED=False, RANK_SKIP=rank_skip, RANK_PROBE=0, **tables,
        FRAME=multi.frame_a, MULTI=True, ALIGNED=aligned, OFF64=multi.off64, num_warps=shrink_warps, num_stages=stages,
    )
    widest = max(sizes)
    offsets, widths = _slice_tables(sizes, x.device)
    scales = out if scale_table is None else scale_table
    if tiled is None:
        tiled = num_tokens > EXPAND_TILED_MIN_TOKENS
    if tiled:
        guard_r = r_pad if block_n is None else r
        block_n = EXPAND_TILED_BLOCK_N if block_n is None else block_n
        while block_n > 64 and guard_r * block_n * x.element_size() > 0.35 * _max_shared_mem(device_index):
            block_n //= 2
        cols_per_prog, warps = block_n, EXPAND_TILED_WARPS if expand_warps is None else expand_warps
    else:
        block_n, warps = EXPAND_BLOCK_N, EXPAND_WARPS
        n_tiles = triton.cdiv(widest, block_n)
        tiles = cap if grid_tiles is None else grid_tiles
        n_split = min(n_tiles, max(1, EXPAND_WAVES * _num_sms(device_index) // (tiles * n_slices)))
        cols_per_prog = triton.cdiv(n_tiles, n_split) * block_n
    _splitk_expand_kernel[(cap, triton.cdiv(widest, cols_per_prog), n_slices)](
        ws, token_ranks, multi.order, scales, B_pool, out, offsets, widths, num_tokens, ws_row0,
        num_tokens, 0, out.stride(0), r_limit, force_rank, cols_per_prog, split_k, 0, ws_slice,
        BM=block_m, BN=block_n, R=r, R_PAD=r_pad, HAS_SCALE=scale_table is not None, ADD_OUT=add,
        ALIGNED=aligned, LOOP=not tiled, INDEXED=False, RANK_SKIP=rank_skip, RANK_PROBE=0, **tables,
        FRAME=multi.frame_b, SCALE_LD=multi.scale_ld, MULTI=True, OFF64=multi.off64, num_warps=warps, num_stages=2,
    )
    return out


def _settings_key() -> tuple:
    """The module settings a launch plan depends on. bench/rf1_ablation.py and bench/tune_blocks.py
    change them at run time and tests monkeypatch TUNED, so they are part of every plan key. TUNED
    and _block_m enter by identity; each plan keeps a reference to both (_SplitKPlan.refs), so an id
    cannot be reused by a later table while a plan keyed on it exists."""
    return (TUNED_GPU, id(TUNED), len(TUNED), id(_block_m), USE_ALIGNED_HINT, SPLITK_BLOCK_H, SPLITK_STAGES, SPLITK_WAVES,
            MAX_SPLIT_K, EXPAND_BLOCK_N, EXPAND_WAVES, EXPAND_WARPS, EXPAND_TILED_MIN_TOKENS, EXPAND_TILED_BLOCK_N,
            EXPAND_TILED_WARPS, MIN_SPLIT_K_TILED)


# Values of the kernels' multi-adapter parameters when MULTI is off (their defaults). Triton's launcher
# binds defaults as arguments, so a direct launch of a compiled kernel passes them as well.
_SHRINK_SINGLE_DEFAULTS = (None, None, None, None, None, False, False, False)
_EXPAND_SINGLE_DEFAULTS = (None, None, None, None, None, None, False, False)


class _SplitKPlan:
    """Launch settings of one _splitk call shape: what _splitk + splitk_pass compute from the shape.

    Built with exactly their rules (_config at r_pad, _splitk_shrink_tiles, _split_count and the
    expand-tile rules of splitk_pass), for the batch in order or through an index, one width (R =
    R_PAD) or RANK_SKIP, and RANK_PROBE. The arguments are pre-assembled in the kernels' signature
    order, constexprs included, as Triton's launcher passes them. The compiled kernels are cached
    per plan (`kernels`), keyed by Triton's run-time specialization (see _launch).
    """

    __slots__ = ("ws_numel", "shrink_grid", "expand_grid", "shrink_mid", "shrink_tail", "expand_mid", "expand_tail",
                 "shrink_opts", "expand_opts", "kernels", "refs")

    def __init__(self, num_tokens, h_in, n_slices, r_max, n_max, sizes, r_pad, elem_size, add, has_scale,
                 out_stride, device_index, split_k, rank_skip=False, rank_probe=0, indexed=False):
        num_sms = _num_sms(device_index)
        # _splitk
        cfg = _config(r_pad, num_tokens, device_index)
        block_m = cfg["bm"]
        m_tiles = triton.cdiv(num_tokens, block_m)
        bh, stages = _splitk_shrink_tiles(block_m, r_pad, elem_size, device_index, cfg["bh"], cfg["stages"])
        tiled = cfg["tiled"]
        split_k, k_per = _split_count(h_in, tiled, m_tiles, n_slices, num_sms, bh, split_k)
        # splitk_pass (count = num_tokens, ws_row0 = 0, rank = None so R = R_PAD, grid_tiles = None)
        ws_slice = split_k * num_tokens * r_pad
        self.ws_numel = n_slices * ws_slice
        self.shrink_grid = (m_tiles, split_k, n_slices)
        self.shrink_mid = (num_tokens, 0, num_tokens, h_in)
        self.shrink_tail = ((k_per, r_max * h_in, ws_slice, block_m, bh, r_pad, r_pad, indexed, rank_skip, rank_probe)
                            + _SHRINK_SINGLE_DEFAULTS)
        self.shrink_opts = (cfg["shrink_warps"], stages)
        widest = max(sizes)
        aligned = USE_ALIGNED_HINT and all(s % 16 == 0 for s in sizes)
        if tiled:
            block_n = EXPAND_TILED_BLOCK_N if cfg["bn"] is None else cfg["bn"]
            while block_n > 64 and r_pad * block_n * elem_size > 0.35 * _max_shared_mem(device_index):
                block_n //= 2
            cols_per_prog = block_n
            warps = EXPAND_TILED_WARPS if cfg["expand_warps"] is None else cfg["expand_warps"]
        else:
            block_n, warps = EXPAND_BLOCK_N, EXPAND_WARPS
            n_tiles = triton.cdiv(widest, block_n)
            n_split = min(n_tiles, max(1, EXPAND_WAVES * num_sms // (m_tiles * n_slices)))
            cols_per_prog = triton.cdiv(n_tiles, n_split) * block_n
        self.expand_grid = (m_tiles, triton.cdiv(widest, cols_per_prog), n_slices)
        self.expand_mid = (num_tokens, 0, num_tokens, n_max, out_stride)
        self.expand_tail = (cols_per_prog, split_k, r_max * n_max, ws_slice, block_m, block_n, r_pad, r_pad, has_scale,
                            add, aligned, not tiled, indexed, rank_skip, rank_probe) + _EXPAND_SINGLE_DEFAULTS
        self.expand_opts = (warps, 2)
        self.kernels = {}
        self.refs = (TUNED, _block_m)


_SPLITK_PLANS: dict[tuple, _SplitKPlan] = {}


def _splitk_plan(num_tokens, h_in, n_slices, r_max, n_max, sizes, r_pad, elem_size, add, has_scale, out_stride,
                 device_index, split_k, rank_skip=False, rank_probe=0, indexed=False) -> _SplitKPlan:
    key = (num_tokens, h_in, n_slices, r_max, n_max, sizes, r_pad, elem_size, add, has_scale, out_stride, device_index,
           split_k, rank_skip, rank_probe, indexed, _settings_key())
    plan = _SPLITK_PLANS.get(key)
    if plan is None:
        plan = _SPLITK_PLANS[key] = _SplitKPlan(*key[:-1])
    return plan


def _ptr_bits(*tensors) -> int:
    """One bit per tensor: is its address 16-byte aligned (Triton's pointer specialization)."""
    bits = 0
    for i, t in enumerate(tensors):
        if t.data_ptr() & 15 == 0:
            bits |= 1 << i
    return bits


def _launch_hooks_active() -> bool:
    """True when a Triton launch hook (a profiler such as proton) is installed. Triton 3.7 keeps
    the hooks in a HookChain that is never None, so an empty chain means no hook."""
    for hook in (knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook):
        if hook is not None and getattr(hook, "calls", True):
            return True
    return False


def _launch(plan, x, A_t, B, token_ranks, index, out, ws, r_limit, force_rank, scales, offsets, widths):
    """Launch the split-K pair of `plan`: _splitk_shrink_kernel, then _splitk_expand_kernel.

    The first call for a given specialization goes through Triton's normal launcher (which
    compiles or loads the kernels and launches them) and keeps the returned compiled kernels.
    Later calls launch those directly: the arguments are exactly the ones splitk_pass binds (same
    order, same values), so the kernels, their grids and their results are identical; what is
    skipped is Triton's per-call argument binding and cache-key computation.

    Triton specializes a compiled kernel on each non-constexpr argument: an int on == 1, % 16 == 0
    and its 32/64-bit range (force_rank is do_not_specialize: its range only), a tensor on its dtype
    and on whether its address is 16-byte aligned. Every int except r_limit and force_rank is fixed
    by the plan key and below 2**31, as are r_limit and force_rank (<= 256). The plan key fixes the
    element size but not the dtype. So the cache key below is r_limit's specialization, the dtypes
    of x (A_t, B and out share it), token_ranks, index and the scale table, and the alignment bit of
    every pointer.
    """
    key = (r_limit == 1, r_limit & 15 == 0, x.dtype, token_ranks.dtype, index.dtype, scales.dtype,
           _ptr_bits(x, A_t, token_ranks, index, ws, scales, B, out, offsets, widths))
    kernels = plan.kernels.get(key)
    shrink_args = (x, A_t, token_ranks, index, ws) + plan.shrink_mid + (r_limit, force_rank) + plan.shrink_tail
    expand_args = ((ws, token_ranks, index, scales, B, out, offsets, widths) + plan.expand_mid + (r_limit, force_rank)
                   + plan.expand_tail)
    if kernels is None or _launch_hooks_active():
        # Slow path: Triton's own launcher (also used whenever a profiler hook is installed).
        warps, stages = plan.shrink_opts
        shrink = _splitk_shrink_kernel[plan.shrink_grid](*shrink_args, num_warps=warps, num_stages=stages)
        warps, stages = plan.expand_opts
        expand = _splitk_expand_kernel[plan.expand_grid](*expand_args, num_warps=warps, num_stages=stages)
        if kernels is None:
            plan.kernels[key] = (shrink, expand)
        return
    shrink, expand = kernels
    stream = driver.active.get_current_stream(driver.active.get_current_device())
    g = plan.shrink_grid
    shrink.run(g[0], g[1], g[2], stream, shrink.function, shrink.packed_metadata, None, None, None, *shrink_args)
    g = plan.expand_grid
    expand.run(g[0], g[1], g[2], stream, expand.function, expand.packed_metadata, None, None, None, *expand_args)


def _splitk(x, A_t, B, token_ranks, out, add, sizes, r_limit, r_pad, scale_table, split_k, rank_skip, force_rank, rank_probe,
            index=None):
    """Section 8: split-K shrink into an fp32 workspace, then sum + mask + scale + expand. index: read
    the tokens through this order (rf2_index) instead of the batch order.

    The same launches as splitk_pass(..., index=index, rank_skip=rank_skip, rank_probe=rank_probe,
    force_rank=force_rank) with _config's settings at r_pad, through a cached plan (_SplitKPlan)
    and direct launches of the compiled kernels (_launch): bitwise the same output."""
    num_tokens, h_in = x.shape
    n_slices, r_max, _ = A_t.shape
    plan = _splitk_plan(num_tokens, h_in, n_slices, r_max, B.shape[2], sizes, r_pad, x.element_size(), add,
                        scale_table is not None, out.stride(0), x.device.index or 0, split_k, rank_skip, rank_probe,
                        index is not None)
    ws = torch.empty(plan.ws_numel, dtype=torch.float32, device=x.device)
    offsets, widths = _slice_tables(sizes, x.device)
    # When a table goes unused, another tensor stands in for its pointer (as in splitk_pass).
    scales = out if scale_table is None else scale_table
    _launch(plan, x, A_t, B, token_ranks, token_ranks if index is None else index, out, ws, r_limit, force_rank,
            scales, offsets, widths)
    return out


# ---------------------------------------------------------------------------------------------
# Serving fast path
# ---------------------------------------------------------------------------------------------

_WORKSPACES: dict[int, torch.Tensor] = {}


def reserve_workspace(device: torch.device, numel: int) -> torch.Tensor:
    """The fp32 split-K workspace shared by every RF1Launcher on `device`, grown to `numel`.

    Launchers on one device run one after another on one stream, so they can share it. It
    only grows here, at setup: a launcher never reallocates it during a call, so its address
    stays fixed once CUDA graphs have been captured.
    """
    index = device.index if device.index is not None else torch.cuda.current_device()
    ws = _WORKSPACES.get(index)
    if ws is None or ws.numel() < numel:
        ws = torch.empty(numel, dtype=torch.float32, device=torch.device("cuda", index))
        _WORKSPACES[index] = ws
    return ws


@functools.lru_cache(maxsize=None)
def _max_ws_numel(max_tokens, h_in, n_slices, r_max, n_max, sizes, r_pad_max, elem_size, device_index, settings) -> int:
    """Largest split-K workspace any RF1 call with 1..max_tokens tokens and any R_PAD up to r_pad_max
    needs (not monotonic in T, and the tuned table may pick a larger slice count at a lower width).
    `settings` (_settings_key()) only keys the cache."""
    r_pads = [r for r in (16, 32, 64, 128, 256) if r <= r_pad_max]
    return max(
        _SplitKPlan(t, h_in, n_slices, r_max, n_max, sizes, r, elem_size, True, False, 0, device_index, None).ws_numel
        for t in range(1, max_tokens + 1) for r in r_pads
    )


class RF1Launcher:
    """rankfuse_lora's RF1 split-K path for one adapter's weights, with the per-call host work removed.

        launcher = RF1Launcher(A_t, B, slice_sizes, max_tokens=8192)       # once, at load time
        launcher(x, token_ranks, out, r_limit=r)                            # every layer call

    Same kernels, launch settings and arguments as
    `rankfuse_lora(x, A_t, B, token_ranks, out=out, add=add, slice_sizes=slice_sizes,
    r_limit=r_limit, scale_table=scale_table)` (splitk_pass's RF1 launch: R = R_PAD, no index, no
    rank skipping), so the result is bitwise identical. What changes is the host side of a call:

        checks          A_t, B, slice_sizes and the scale table are checked here, once. A call
                        trusts its arguments: x [T, h_in] contiguous, out [T, sum(sizes)] with
                        unit column stride, both of A_t's dtype and on its device; token_ranks
                        int32/int64 with at least T entries (only the first T are read, so a
                        server can pass a static per-step buffer without slicing it).
        r_limit         required (no host sync). It may also be a value read from a CPU tensor
                        that the server fills per step.
        launch settings cached per (T, r_pad, add, out row stride, module settings); the slice
                        tables are built here.
        workspace       one shared fp32 buffer per device (reserve_workspace), sized here for
                        max_tokens, instead of torch.empty per call.
        Triton launch   the compiled kernels are launched directly (_launch).

    Weights are referenced, not copied: a server may update A_t and B in place.
    r_limit == 0 (no token uses the adapter) launches nothing with add=True and zeroes out
    with add=False, which is what the kernels would produce.
    """

    def __init__(self, A_t, B, slice_sizes=None, *, max_tokens: int, scale_table: torch.Tensor | None = None):
        if A_t.dim() == 2:
            A_t = A_t.unsqueeze(0)
        if B.dim() == 2:
            B = B.unsqueeze(0)
        n_slices, r_max, h_in = A_t.shape
        if B.dim() != 3 or B.shape[0] != n_slices or B.shape[1] != r_max:
            raise ValueError(f"B must be [{n_slices}, {r_max}, n] to match A_t {tuple(A_t.shape)}")
        n_max = B.shape[2]
        sizes = tuple(int(s) for s in slice_sizes) if slice_sizes is not None else (n_max,) * n_slices
        if len(sizes) != n_slices or not all(0 < s <= n_max for s in sizes):
            raise ValueError(f"slice_sizes must give {n_slices} widths in [1, {n_max}], got {sizes}")
        if not A_t.is_cuda or A_t.dtype not in SUPPORTED_DTYPES or B.dtype != A_t.dtype or B.device != A_t.device:
            raise ValueError("A_t and B must be CUDA tensors of one supported floating dtype")
        if not (A_t.is_contiguous() and B.is_contiguous()):
            raise ValueError("A_t and B must be contiguous")
        if scale_table is not None and (
            scale_table.dtype != torch.float32 or scale_table.device != A_t.device or scale_table.numel() < r_max + 1
        ):
            raise ValueError(f"scale_table must be fp32 [{r_max + 1}] on {A_t.device}")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        self.A_t, self.B, self.sizes, self.scale_table = A_t, B, sizes, scale_table
        self.n_slices, self.r_max, self.h_in, self.n_max = n_slices, r_max, h_in, n_max
        self.max_tokens = max_tokens
        self.device_index = A_t.device.index or 0
        self.elem_size = A_t.element_size()
        self.r_pad_max = _r_pad(r_max)
        if self.r_pad_max > MAX_R_PAD:
            raise ValueError(f"ranks above {MAX_R_PAD} are not supported")
        self.offsets, self.widths = _slice_tables(sizes, A_t.device)
        self.ws = reserve_workspace(A_t.device, _max_ws_numel(
            max_tokens, h_in, n_slices, r_max, n_max, sizes, self.r_pad_max, self.elem_size, self.device_index,
            _settings_key()))
        self._plans: dict[tuple, _SplitKPlan] = {}

    def __call__(self, x, token_ranks, out, *, r_limit: int, add: bool = True) -> torch.Tensor:
        num_tokens = x.shape[0]
        if num_tokens == 0:
            return out
        if r_limit <= 0:
            if not add:
                out.zero_()
            return out
        out_stride = out.stride(0)
        r_pad = _r_pad(r_limit)
        pkey = (num_tokens, r_pad, add, out_stride, _settings_key())
        plan = self._plans.get(pkey)
        if plan is None:
            if num_tokens > self.max_tokens or r_limit > self.r_max:
                raise ValueError(f"T={num_tokens} / r_limit={r_limit} above the launcher's max_tokens / r_max")
            plan = self._plans[pkey] = _splitk_plan(
                num_tokens, self.h_in, self.n_slices, self.r_max, self.n_max, self.sizes, r_pad, self.elem_size,
                add, self.scale_table is not None, out_stride, self.device_index, None)
            if plan.ws_numel > self.ws.numel():
                raise RuntimeError("split-K workspace too small (reserve_workspace was not sized for this call)")
        scales = out if self.scale_table is None else self.scale_table
        _launch(plan, x, self.A_t, self.B, token_ranks, token_ranks, out, self.ws, r_limit, 0, scales, self.offsets,
                self.widths)
        return out


def rankfuse_lora(
    x: torch.Tensor,
    A_t: torch.Tensor,
    B: torch.Tensor,
    token_ranks: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    add: bool = False,
    slice_sizes: list[int] | tuple[int, ...] | None = None,
    r_limit: int | None = None,
    scale_table: torch.Tensor | None = None,
    rank_sort: bool = False,
    order: torch.Tensor | None = None,
    split_k: int | None = None,
    rank_skip: bool = False,
    full_width: bool = False,
    rank_probe: int = 0,
) -> torch.Tensor:
    """Fused Matryoshka LoRA: out[:, slice s] (+)= scale * x @ A_t[s, :r].T @ B[s, :r, :size_s].

    x [T, h_in]; A_t [r_max, h_in] or [S, r_max, h_in]; B [r_max, h_out] or [S, r_max, n_max];
    token_ranks [T]; scale_table [r_max + 1] fp32 or None. Returns out [T, sum(slice_sizes)].
    rank_sort=False is RF1 (pad to r_limit; split_k picks its path, section 8); rank_sort=True
    is the one-kernel rf2_fused. rank_skip=True is RF2 on RF1's split-K path: each block runs at
    its own rank bucket, over the batch's order (a caller wanting rank order passes sorted inputs),
    or with rank_sort too, over the rank-sorted order (rf2_index; order, or argsort by default).
    full_width=True keeps that same kernel but makes every block take the batch's bucket, the
    control that isolates skipping. rank_probe=1 or 2 is RF1's full width plus the rank read, or
    the read and one branch (diagnostics). See the module docstring, section 5, for every option.
    """
    if split_k is not None and split_k < 0:
        raise ValueError(f"split_k must be None, 0 or positive, got {split_k}")
    if rank_skip and split_k == 0:
        raise ValueError("rank_skip runs on the split-K path; it cannot take split_k=0")
    if rank_probe not in (0, 1, 2) or (rank_probe and (rank_skip or rank_sort or split_k == 0)):
        raise ValueError("rank_probe is 1 or 2, on RF1's split-K path only")
    if full_width and not rank_skip:
        raise ValueError("full_width is the control for rank_skip; it needs rank_skip=True")
    if A_t.dim() == 2:
        A_t = A_t.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(0)
    n_slices, r_max, h_in = A_t.shape
    if B.dim() != 3 or B.shape[0] != n_slices or B.shape[1] != r_max:
        raise ValueError(f"B must be [{n_slices}, {r_max}, n] to match A_t {tuple(A_t.shape)}")
    n_max = B.shape[2]
    sizes = tuple(int(s) for s in slice_sizes) if slice_sizes is not None else (n_max,) * n_slices
    if len(sizes) != n_slices or not all(0 < s <= n_max for s in sizes):
        raise ValueError(f"slice_sizes must give {n_slices} widths in [1, {n_max}], got {sizes}")

    if not x.is_cuda or x.ndim != 2 or x.shape[1] != h_in:
        raise ValueError(f"x must be a CUDA tensor of shape [T, {h_in}]")
    if x.dtype not in SUPPORTED_DTYPES or A_t.dtype != x.dtype or B.dtype != x.dtype:
        raise ValueError("x, A_t, and B must share a supported floating dtype")
    if token_ranks.ndim != 1 or token_ranks.numel() != x.shape[0]:
        raise ValueError("token_ranks must be [T]")
    if token_ranks.dtype not in (torch.int32, torch.int64):
        raise ValueError("token_ranks must be int32 or int64")
    if any(t.device != x.device for t in (A_t, B, token_ranks)):
        raise ValueError("all inputs must be on the same CUDA device")
    if not all(t.is_contiguous() for t in (x, A_t, B, token_ranks)):
        raise ValueError("x, A_t, B, and token_ranks must be contiguous")

    num_tokens = x.shape[0]
    total = sum(sizes)
    if out is None:
        if add:
            raise ValueError("add=True needs an existing out to add into")
        out = torch.empty((num_tokens, total), dtype=x.dtype, device=x.device)
    elif (
        tuple(out.shape) != (num_tokens, total)
        or out.dtype != x.dtype
        or out.device != x.device
        or out.stride(1) != 1
    ):
        raise ValueError(f"out must be [{num_tokens}, {total}] {x.dtype} with unit column stride")
    if num_tokens == 0:
        return out

    # Batch maximum rank. min and max come back in one transfer (one host sync).
    if r_limit is None:
        lo, r_limit = torch.stack(torch.aminmax(token_ranks)).tolist()
        if lo < 0 or r_limit > r_max:
            raise ValueError(f"token ranks must be in [0, {r_max}]")
    elif not 0 <= r_limit <= r_max:
        raise ValueError(f"r_limit must be in [0, {r_max}]")
    r_pad = _r_pad(r_limit)
    if r_pad > MAX_R_PAD:
        raise ValueError(f"ranks above {MAX_R_PAD} are not supported")
    if rank_skip or (not rank_sort and split_k != 0):
        index = None
        if rank_skip and rank_sort:
            index = (order if order is not None else torch.argsort(token_ranks, stable=True)).to(torch.int32)
        return _splitk(x, A_t, B, token_ranks, out, add, sizes, r_limit, r_pad, scale_table, split_k,
                       rank_skip, r_limit if full_width or rank_probe == 2 else 0, rank_probe, index=index)

    # Token blocks, column split, and grid.
    block_m = _block_m(num_tokens, r_pad)
    m_tiles = triton.cdiv(num_tokens, block_m)
    widest = max(sizes)
    n_tiles = triton.cdiv(widest, BLOCK_N)
    n_split = min(n_tiles, max(1, _num_sms(x.device.index or 0) // (m_tiles * n_slices)))
    cols_per_prog = triton.cdiv(n_tiles, n_split) * BLOCK_N
    grid = (m_tiles, triton.cdiv(widest, cols_per_prog), n_slices)

    offsets, widths = _slice_tables(sizes, x.device)
    # When there is no scale, out stands in for the unused table pointer.
    scales = out if scale_table is None else scale_table
    shared = (
        num_tokens, h_in, n_max, out.stride(0), r_limit, cols_per_prog, r_max * h_in, r_max * n_max,
    )
    meta = dict(
        BM=block_m, BH=BLOCK_H, BN=BLOCK_N, R_PAD=r_pad, HAS_SCALE=scale_table is not None,
        ADD_OUT=add, num_warps=4, num_stages=2,
    )
    if rank_sort:
        if order is None:
            order = torch.argsort(token_ranks, stable=True)
        _rf2_fused_kernel[grid](
            x, A_t, B, token_ranks, order, scales, out, offsets, widths, *shared,
            ZERO_FILL=not add, **meta,
        )
    else:
        _rf1_fused_kernel[grid](
            x, A_t, B, token_ranks, scales, out, offsets, widths, *shared, **meta,
        )
    return out


@register("rf1_fused", requires_cuda=True, prepare=prepare_r_limit)
def rankfuse_fused(
    x: torch.Tensor,
    lora: MatryoshkaLoRA,
    token_ranks: torch.Tensor,
    r_limit: int | None = None,
    out: torch.Tensor | None = None,
    add: bool = False,
) -> torch.Tensor:
    """RF1: pad to the batch maximum rank, mask per token; split-K path (section 8).

    r_limit: batch maximum rank. Leave it None in benchmarks so this pays the same single
    host sync as B1; a serving engine that knows request ranks should pass it in, which
    removes every host sync and makes the call CUDA-graph capturable.
    """
    check_layout(x, lora, token_ranks)
    return rankfuse_lora(
        x, lora.A_t, lora.B, token_ranks, out=out, add=add, r_limit=r_limit, scale_table=_scale_table(lora, x.device)
    )


@register("rf1_single", requires_cuda=True, prepare=prepare_r_limit)
def rankfuse_single(
    x: torch.Tensor,
    lora: MatryoshkaLoRA,
    token_ranks: torch.Tensor,
    r_limit: int | None = None,
) -> torch.Tensor:
    """RF1 in one launch (sections 3-4): the fusion ablation for rf1_fused's split-K path."""
    check_layout(x, lora, token_ranks)
    return rankfuse_lora(
        x, lora.A_t, lora.B, token_ranks, r_limit=r_limit, scale_table=_scale_table(lora, x.device),
        split_k=0,
    )


@register("rf2_fused", requires_cuda=True, prepare=prepare_rank_order)
def rankfuse_fused_ragged(
    x: torch.Tensor,
    lora: MatryoshkaLoRA,
    token_ranks: torch.Tensor,
    r_limit: int | None = None,
    order: torch.Tensor | None = None,
) -> torch.Tensor:
    """RF2: sort tokens by rank; each token block runs only up to its own rank bucket.

    order: a precomputed rank-sorted token order to reuse across layers (default: argsort).
    """
    check_layout(x, lora, token_ranks)
    return rankfuse_lora(
        x, lora.A_t, lora.B, token_ranks, r_limit=r_limit, scale_table=_scale_table(lora, x.device),
        rank_sort=True, order=order,
    )
