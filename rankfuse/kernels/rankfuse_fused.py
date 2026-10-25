"""RF1 and RF2 (fused): LoRA kernels that compute the shrink once per program.

    rf1_fused   pad every token to the batch maximum rank and mask per token (B1's math), on
                the split-K path of section 8 (two kernels)
    rf1_single  the same math in ONE kernel (sections 3-4), kept as the fusion ablation
    rf2_fused   the same, but tokens are sorted by rank and each token block stops at its own
                maximum rank, so low-rank blocks load and multiply fewer rank rows

Both are thin wrappers around `rankfuse_lora`, which also takes the options a serving engine
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

rf2_fused adds rank skipping. The host sorts token indices by rank, so each token block holds
tokens of similar rank. Each program reads its block's maximum rank and runs a version of the
tile compiled for the smallest rank bucket (16/32/64/128/256) that covers it. A block of
rank-8 tokens loads 16 rank rows of A_t and B instead of r_limit. rf1_fused vs rf2_fused
measures the benefit of skipping unused ranks (Claim 2).

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
    order           rf2 only: a precomputed rank-sorted token order (the scheduler can build it
                    once per step and reuse it across layers). Default: torch.argsort.
    split_k         rf1 only. None (default): the split-K path of section 8 with an automatic
                    slice count; 0: the single-kernel path; n >= 1: split-K with about n slices.

CUDA graphs: with r_limit passed, rankfuse_lora makes no host syncs and allocates nothing but
`out` (when not given) and rf2's sort. Call it once before capture so the kernels compile and
the cached scale and slice tables exist; capture and replay then work.

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

Only BM, BH, BN, R_PAD, HAS_SCALE, ADD_OUT and ZERO_FILL are tl.constexpr. T, h_in, n_max,
r_limit and cols_per_prog are runtime arguments, so a new batch size does not recompile.

=====================================================================================
7. Assumptions
=====================================================================================

* One adapter per call (per slice); every token reads the same A_t and B.
* x, A_t, B contiguous and sharing one dtype (fp16, bf16, fp32) with out; token_ranks
  int32/int64.
* Every token rank is <= r_limit. When the caller passes r_limit this is trusted, not
  checked: a larger rank would be silently truncated to r_limit.
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

    _rf1_shrink_splitk_kernel  grid (m_tiles, S, n_slices). Program (m, k, s) reduces slice k
        of h_in (K_PER inputs, a whole number of BH chunks) for token block m and stores its
        partial v [BM, R_PAD] in fp32 to ws[s, k, tok, :]. The slices run in parallel, so the
        longest chain of dependent loads is K_PER / BH steps instead of h_in / BH. S is chosen
        so that m_tiles * S * n_slices is about SPLITK_WAVES x the SM count (at most
        MAX_SPLIT_K); cuBLAS splits B1's shrink the same way.
    _rf1_expand_splitk_kernel  grid (m_tiles, n_split, n_slices). Each program sums the S
        partials of its tokens, applies the rank mask and the scale (both linear per token, so
        they commute with the sum), casts v and runs the expand over its column range, adding
        into out with add=True as before. n_split aims at EXPAND_WAVES programs per SM, so
        several programs share an SM and overlap their loads and stores.
    _rf1_expand_tile_kernel  above EXPAND_TILED_MIN_TOKENS tokens, instead of the kernel above:
        grid (m_tiles, cdiv(h_out, BN), n_slices), every program owns ONE BM x
        EXPAND_TILED_BLOCK_N output tile and has no column loop (thousands of programs at long
        prompts, which keeps many output writes in flight). The shrink then uses at least
        MIN_SPLIT_K_TILED slices (on an H200: 4 at 2048 tokens, 2 at 4096 and 8192).

Both expand kernels take ALIGNED: when every slice width (hence every offset) is a multiple
of 16, the kernel is told so with tl.multiple_of, which lets the output stores be vectorized.
Slice offsets come from a table in memory, so the compiler cannot infer this; without the
hint the stores go out one element at a time (the expand at 8192 tokens took 67 instead of 45
us). Single projections and vLLM's fused q/k/v and gate/up layouts are all aligned.

ws is [n_slices, S, T, R_PAD] fp32, allocated per call (inside a CUDA-graph capture it comes
from the graph's memory pool). At T = 32 with S = 16 it is 128 KiB and stays in L2. Rank-0
rows need no special case: their v is masked to zero. The shrink's pipeline depth, then BH,
shrink when one stage of x and A_t tiles would not fit in shared memory (fp32 inputs,
r_max 128-256, GPUs with less shared memory).
"""

import functools

import torch
import triton
import triton.language as tl

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
def _rf1_shrink_splitk_kernel(
    x_ptr,
    a_ptr,
    ws_ptr,
    T,
    H_IN,
    r_limit,
    K_PER,
    A_SLICE,
    WS_SLICE,
    BM: tl.constexpr,
    BH: tl.constexpr,
    R_PAD: tl.constexpr,
):
    # Section 8, phase 1: token block m, h_in slice k, output slice s.
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_s = tl.program_id(2)
    rows = pid_m * BM + tl.arange(0, BM)
    row_ok = rows < T
    k = tl.arange(0, R_PAD)
    k_ok = k < r_limit
    a_s = a_ptr + pid_s * A_SLICE
    h_lo = pid_k * K_PER
    v = tl.zeros((BM, R_PAD), dtype=tl.float32)
    for h_start in range(0, K_PER, BH):
        h = h_lo + h_start + tl.arange(0, BH)
        h_ok = h < H_IN
        x = tl.load(x_ptr + rows[:, None] * H_IN + h[None, :], mask=row_ok[:, None] & h_ok[None, :], other=0.0)
        a = tl.load(a_s + k[:, None] * H_IN + h[None, :], mask=k_ok[:, None] & h_ok[None, :], other=0.0)
        v += tl.dot(x, tl.trans(a), input_precision="ieee")
    ws = ws_ptr + pid_s * WS_SLICE + pid_k * T * R_PAD + rows[:, None] * R_PAD + k[None, :]
    tl.store(ws, v, mask=row_ok[:, None])


@triton.jit
def _rf1_expand_splitk_kernel(
    ws_ptr,
    ranks_ptr,
    scale_tab_ptr,
    b_ptr,
    out_ptr,
    slice_off_ptr,
    slice_size_ptr,
    T,
    N_MAX,
    OUT_STRIDE,
    r_limit,
    cols_per_prog,
    SPLIT_K,
    B_SLICE,
    WS_SLICE,
    BM: tl.constexpr,
    BN: tl.constexpr,
    R_PAD: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    ADD_OUT: tl.constexpr,
    ALIGNED: tl.constexpr,
):
    # Section 8, phase 2: sum the partials, mask, scale, expand one column range.
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(2)
    out_col0 = tl.load(slice_off_ptr + pid_s)
    n_cols = tl.load(slice_size_ptr + pid_s)
    if ALIGNED:
        # Offsets and widths are multiples of 16 (checked on the host): tell the compiler, so
        # the output stores can be vectorized. Without it they are issued one element at a time.
        out_col0 = tl.multiple_of(out_col0, 16)
        n_cols = tl.multiple_of(n_cols, 16)
    col_base = tl.program_id(1) * cols_per_prog
    if col_base < n_cols:
        rows = pid_m * BM + tl.arange(0, BM)
        row_ok = rows < T
        ranks = tl.load(ranks_ptr + rows, mask=row_ok, other=0)
        k = tl.arange(0, R_PAD)
        k_ok = k < r_limit
        ws_s = ws_ptr + pid_s * WS_SLICE
        v = tl.zeros((BM, R_PAD), dtype=tl.float32)
        for split in range(0, SPLIT_K):
            v += tl.load(ws_s + split * T * R_PAD + rows[:, None] * R_PAD + k[None, :], mask=row_ok[:, None], other=0.0)
        v = tl.where(k[None, :] < ranks[:, None], v, 0.0)
        if HAS_SCALE:
            scale = tl.load(scale_tab_ptr + ranks, mask=row_ok, other=0.0)
            v = v * scale[:, None]
        v_in = v.to(out_ptr.dtype.element_ty)
        b_s = b_ptr + pid_s * B_SLICE
        for col_start in range(0, cols_per_prog, BN):
            cols = col_base + col_start + tl.arange(0, BN)
            col_ok = cols < n_cols
            b = tl.load(b_s + k[:, None] * N_MAX + cols[None, :], mask=k_ok[:, None] & col_ok[None, :], other=0.0)
            acc = tl.dot(v_in, b, input_precision="ieee")
            ptrs = out_ptr + rows[:, None] * OUT_STRIDE + out_col0 + cols[None, :]
            mask = row_ok[:, None] & col_ok[None, :]
            if ADD_OUT:
                acc += tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
            tl.store(ptrs, acc, mask=mask)


@triton.jit
def _rf1_expand_tile_kernel(
    ws_ptr,
    ranks_ptr,
    scale_tab_ptr,
    b_ptr,
    out_ptr,
    slice_off_ptr,
    slice_size_ptr,
    T,
    N_MAX,
    OUT_STRIDE,
    r_limit,
    SPLIT_K,
    B_SLICE,
    WS_SLICE,
    BM: tl.constexpr,
    BN: tl.constexpr,
    R_PAD: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    ADD_OUT: tl.constexpr,
    ALIGNED: tl.constexpr,
):
    # Section 8, phase 2 above EXPAND_TILED_MIN_TOKENS: one BM x BN output tile per program and
    # no column loop (a loop that runs once still cost ~30% of the expand at 8192 tokens).
    pid_s = tl.program_id(2)
    out_col0 = tl.load(slice_off_ptr + pid_s)
    n_cols = tl.load(slice_size_ptr + pid_s)
    if ALIGNED:
        out_col0 = tl.multiple_of(out_col0, 16)
        n_cols = tl.multiple_of(n_cols, 16)
    col_base = tl.program_id(1) * BN
    if col_base < n_cols:
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        row_ok = rows < T
        ranks = tl.load(ranks_ptr + rows, mask=row_ok, other=0)
        k = tl.arange(0, R_PAD)
        k_ok = k < r_limit
        ws_s = ws_ptr + pid_s * WS_SLICE
        v = tl.zeros((BM, R_PAD), dtype=tl.float32)
        for split in range(0, SPLIT_K):
            v += tl.load(ws_s + split * T * R_PAD + rows[:, None] * R_PAD + k[None, :], mask=row_ok[:, None], other=0.0)
        v = tl.where(k[None, :] < ranks[:, None], v, 0.0)
        if HAS_SCALE:
            scale = tl.load(scale_tab_ptr + ranks, mask=row_ok, other=0.0)
            v = v * scale[:, None]
        cols = col_base + tl.arange(0, BN)
        col_ok = cols < n_cols
        b = tl.load(b_ptr + pid_s * B_SLICE + k[:, None] * N_MAX + cols[None, :],
                    mask=k_ok[:, None] & col_ok[None, :], other=0.0)
        acc = tl.dot(v.to(out_ptr.dtype.element_ty), b, input_precision="ieee")
        ptrs = out_ptr + rows[:, None] * OUT_STRIDE + out_col0 + cols[None, :]
        mask = row_ok[:, None] & col_ok[None, :]
        if ADD_OUT:
            acc += tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        tl.store(ptrs, acc, mask=mask)


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


def _splitk_shrink_tiles(block_m: int, r_pad: int, elem_size: int, device_index: int) -> tuple[int, int]:
    """(BH, num_stages) for the split-K shrink: SPLITK_BLOCK_H and SPLITK_STAGES, reduced
    while the x and A_t tiles of all pipeline stages would exceed ~70% of shared memory."""
    budget = int(0.7 * _max_shared_mem(device_index))
    bh, stages = SPLITK_BLOCK_H, SPLITK_STAGES
    while stages > 2 and (block_m + r_pad) * bh * elem_size * stages > budget:
        stages -= 1
    while bh > 32 and (block_m + r_pad) * bh * elem_size * stages > budget:
        bh //= 2
    return bh, stages


def _rf1_splitk(x, A_t, B, token_ranks, out, add, sizes, r_limit, r_pad, scale_table, split_k):
    """Section 8: split-K shrink into an fp32 workspace, then sum + mask + scale + expand."""
    num_tokens, h_in = x.shape
    n_slices, r_max, _ = A_t.shape
    n_max = B.shape[2]
    device_index = x.device.index or 0
    num_sms = _num_sms(device_index)
    block_m = _block_m(num_tokens, r_pad)
    m_tiles = triton.cdiv(num_tokens, block_m)
    bh, stages = _splitk_shrink_tiles(block_m, r_pad, x.element_size(), device_index)
    tiled = num_tokens > EXPAND_TILED_MIN_TOKENS
    if split_k is None:
        split_k = max(MIN_SPLIT_K_TILED if tiled else 1, min(MAX_SPLIT_K, SPLITK_WAVES * num_sms // (m_tiles * n_slices)))
    k_per = triton.cdiv(triton.cdiv(h_in, split_k), bh) * bh  # whole BH chunks per slice
    split_k = triton.cdiv(h_in, k_per)  # no empty slices
    ws = torch.empty((n_slices, split_k, num_tokens, r_pad), dtype=torch.float32, device=x.device)
    ws_slice = split_k * num_tokens * r_pad
    _rf1_shrink_splitk_kernel[(m_tiles, split_k, n_slices)](
        x, A_t, ws, num_tokens, h_in, r_limit, k_per, r_max * h_in, ws_slice,
        BM=block_m, BH=bh, R_PAD=r_pad, num_warps=4, num_stages=stages,
    )

    widest = max(sizes)
    offsets, widths = _slice_tables(sizes, x.device)
    # When there is no scale, out stands in for the unused table pointer.
    scales = out if scale_table is None else scale_table
    aligned = all(s % 16 == 0 for s in sizes)
    if tiled:
        # One tile per program; narrow the tile while B's [R_PAD, BN] tile would take more
        # than ~35% of shared memory (fp32, r_max 128-256, smaller GPUs).
        block_n = EXPAND_TILED_BLOCK_N
        while block_n > 64 and r_pad * block_n * x.element_size() > 0.35 * _max_shared_mem(device_index):
            block_n //= 2
        _rf1_expand_tile_kernel[(m_tiles, triton.cdiv(widest, block_n), n_slices)](
            ws, token_ranks, scales, B, out, offsets, widths,
            num_tokens, n_max, out.stride(0), r_limit, split_k, r_max * n_max, ws_slice,
            BM=block_m, BN=block_n, R_PAD=r_pad, HAS_SCALE=scale_table is not None, ADD_OUT=add,
            ALIGNED=aligned, num_warps=EXPAND_TILED_WARPS, num_stages=2,
        )
        return out

    n_tiles = triton.cdiv(widest, EXPAND_BLOCK_N)
    n_split = min(n_tiles, max(1, EXPAND_WAVES * num_sms // (m_tiles * n_slices)))
    cols_per_prog = triton.cdiv(n_tiles, n_split) * EXPAND_BLOCK_N
    _rf1_expand_splitk_kernel[(m_tiles, triton.cdiv(widest, cols_per_prog), n_slices)](
        ws, token_ranks, scales, B, out, offsets, widths,
        num_tokens, n_max, out.stride(0), r_limit, cols_per_prog, split_k, r_max * n_max, ws_slice,
        BM=block_m, BN=EXPAND_BLOCK_N, R_PAD=r_pad, HAS_SCALE=scale_table is not None, ADD_OUT=add,
        ALIGNED=aligned, num_warps=EXPAND_WARPS, num_stages=2,
    )
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
) -> torch.Tensor:
    """Fused Matryoshka LoRA: out[:, slice s] (+)= scale * x @ A_t[s, :r].T @ B[s, :r, :size_s].

    x [T, h_in]; A_t [r_max, h_in] or [S, r_max, h_in]; B [r_max, h_out] or [S, r_max, n_max];
    token_ranks [T]; scale_table [r_max + 1] fp32 or None. Returns out [T, sum(slice_sizes)].
    rank_sort=False is RF1 (pad to r_limit; split_k picks its path, section 8); rank_sort=True
    is RF2 (per-block rank buckets). See the module docstring, section 5, for every option.
    """
    if split_k is not None and split_k < 0:
        raise ValueError(f"split_k must be None, 0 or positive, got {split_k}")
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
    r_pad = max(16, triton.next_power_of_2(max(r_limit, 1)))
    if r_pad > MAX_R_PAD:
        raise ValueError(f"ranks above {MAX_R_PAD} are not supported")
    if not rank_sort and split_k != 0:
        return _rf1_splitk(x, A_t, B, token_ranks, out, add, sizes, r_limit, r_pad, scale_table, split_k)

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
) -> torch.Tensor:
    """RF1: pad to the batch maximum rank, mask per token; split-K path (section 8).

    r_limit: batch maximum rank. Leave it None in benchmarks so this pays the same single
    host sync as B1; a serving engine that knows request ranks should pass it in, which
    removes every host sync and makes the call CUDA-graph capturable.
    """
    check_layout(x, lora, token_ranks)
    return rankfuse_lora(
        x, lora.A_t, lora.B, token_ranks, r_limit=r_limit, scale_table=_scale_table(lora, x.device)
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
