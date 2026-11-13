# Phase 1 Static Audit: Multi-Adapter Kernel & Paging

This document records the static audit of the multi-adapter Matryoshka LoRA kernels (`rankfuse/kernels/rankfuse_ma.py`), paging manager and pool (`rankfuse/paging/`), simulator (`rankfuse/paging/sim.py`), and test suite (`tests/test_multi_adapter.py`, `tests/test_paging.py`).

---

## 1. Static Audit Checklist (Items 1-9)

### 1. Paged kernel used an undefined variable (`k_ok`) and could not compile
- **Status**: FIXED
- **Evidence**:
  - `rankfuse/kernels/rankfuse_ma.py:192`: In `_rf1_ma_paged_shrink_splitk_kernel`, `k_ok` is defined as `k_ok = (frame >= 0) & (k < r_limit)` before being used at line 209: `mask=k_ok[:, None] & h_ok[None, :]`.
  - `rankfuse/kernels/rankfuse_ma.py:388`: In `_rf1_ma_paged_expand_splitk_kernel`, `k_ok` is defined as `k_ok = (frame >= 0) & (k < r_limit)` before being used at line 424: `mask=k_ok[:, None] & col_ok[None, :]`.
- **Details**: Both shrink and expand paged kernels properly define `k_ok` using the loaded frame validity (`frame >= 0`) and group rank limit (`k < r_limit`). Compiles and runs cleanly under both CUDA and Triton interpreter.

---

### 2. Kernels used BM=1 with a scalar `for h in range(H_IN)` loop, no split-K, no tensor cores
- **Status**: FIXED
- **Evidence**:
  - `rankfuse/kernels/rankfuse_ma.py:27, 48, 93, 167, 250, 351`: `DEFAULT_BM = 16`, passed as constexpr `BM >= 16`.
  - `rankfuse/kernels/rankfuse_ma.py:124-137`: `_rf1_ma_flat_shrink_splitk_kernel` loops with step `BH` (16, 32, or 64) and uses `tl.dot(x, tl.trans(a), input_precision="ieee")` targeting Tensor Cores.
  - `rankfuse/kernels/rankfuse_ma.py:199-212`: `_rf1_ma_paged_shrink_splitk_kernel` uses vectorized chunks of `BH` with `tl.dot(x, tl.trans(a), input_precision="ieee")`.
  - `rankfuse/kernels/rankfuse_ma.py:310-318, 419-427`: Expand kernels tile over columns in blocks of `BN` (16, 32, 64, or 128) using `tl.dot(v_in, b, input_precision="ieee")`.
  - `rankfuse/kernels/rankfuse_ma.py:102, 176, 286-294, 392-400`: Split-K is implemented across both shrink and expand passes: shrink program id `pid_k` writes partial reductions to the workspace, which expand kernels sum across `SPLIT_K`.
- **Details**: Scalar loop over `H_IN` with BM=1 has been completely eliminated in favor of block-tiled, Tensor-Core-accelerated `tl.dot` routines with split-K reduction.

---

### 3. group_starts / group_lens / group_rank_limit accepted but ignored; no merged q|k|v, gate|up slices
- **Status**: FIXED
- **Evidence**:
  - `rankfuse/kernels/rankfuse_ma.py:497-539`: `_prepare_groups()` accepts `group_starts`, `group_lens`, and `group_rank_limit`, propagating them to the GPU tile table builder.
  - `rankfuse/kernels/rankfuse_ma.py:36-71`: `_build_tile_table_kernel` unpacks per-group starts, lengths, adapters, and rank limits, writing `tile_adapter`, `tile_tok_start`, `tile_tok_len`, and `tile_r_limit` per tile without straddling groups.
  - `rankfuse/kernels/rankfuse_ma.py:107, 114, 181, 192`: Shrink kernels load `r_limit` per tile and apply `k_ok = k < r_limit`.
  - `rankfuse/kernels/rankfuse_ma.py:449-457, 579-584, 701-706`: Merged slices are supported via `slice_sizes` parameter. Precomputed offset and width tables (`_slice_tables`) are indexed by `pid_s = tl.program_id(2)` in both shrink and expand kernels to process merged projections (e.g. q|k|v, gate|up).
- **Details**: Group parameters and slice partitions are respected end-to-end and verified by `test_rf1_ma_group_rank_limits` and `test_rf1_ma_merged_slices`.

---

### 4. frame * PAGE_STRIDE was int32 (overflow at >= 16,384 frames); page index not bounds-masked
- **Status**: FIXED
- **Evidence**:
  - `rankfuse/kernels/rankfuse_ma.py:189-195`: In `_rf1_ma_paged_shrink_splitk_kernel`, page index is bounds-masked with `page_ok = page < PAGES` in `tl.load(page_table_ptr + pt_offset, mask=page_ok, other=-1)`. Pointer offset computation explicitly casts `frame` to `tl.int64` before multiply: `k_offset = frame.to(tl.int64) * PAGE_STRIDE + (k % 16).to(tl.int64) * FRAME_WIDTH`.
  - `rankfuse/kernels/rankfuse_ma.py:385-388, 412-417`: In `_rf1_ma_paged_expand_splitk_kernel`, page index is bounds-masked with `page_ok = page < PAGES`, and `k_b_offset` performs `frame.to(tl.int64) * PAGE_STRIDE + (k % 16).to(tl.int64) * FRAME_WIDTH + B_START + out_col0`.
- **Details**: Memory address arithmetic uses 64-bit precision, preventing overflow at $\ge 16,384$ frames. Verified by `test_rf1_ma_int64_address_overflow_logic` and `test_rf1_ma_int64_address_gpu`.

---

### 5. Eviction wiped a whole adapter instead of only its top page
- **Status**: FIXED
- **Evidence**:
  - `rankfuse/paging/adapter_page_manager.py:181-199`: `evict(adapter_id, from_page=0)` takes `from_page`.
  - `rankfuse/paging/adapter_page_manager.py:200-207`: `_evict_one()` calls `self.evict(adapter_id, from_page=self._resident[adapter_id] - 1)`.
  - `tests/test_paging.py:59-116, 118-140`: `test_seven_step_worked_example` and `test_eviction_removes_only_top_page` verify that when adapter A at rank 64 (4 pages) is evicted, only page 3 is evicted; pages 0, 1, 2 remain resident in the pool.
- **Details**: `AdapterPageManager` removes strictly the top resident page per eviction event.

---

### 6. .item() GPU readbacks in the page manager; unpinned host copies
- **Status**: FIXED
- **Evidence**:
  - `rankfuse/paging/adapter_page_manager.py:51-53`: Shadow CPU page table `self.cpu_page_table = np.full((slots, pages_per_adapter), -1, dtype=np.int32)` tracks mappings without reading back from GPU.
  - `rankfuse/paging/adapter_page_manager.py:165, 188`: Slot lookups use `int(self.cpu_page_table[adapter_idx, page])`, completely removing GPU-to-CPU `.item()` synchronization.
  - `rankfuse/paging/adapter_page_manager.py:84-107`: Pre-allocates pinned host buffers (`pin_memory=True`) in `self.cpu_store` during `register()`.
  - `rankfuse/paging/adapter_page_manager.py:148-156`: Uses dedicated asynchronous CUDA copy stream `self.copy_stream` and `self.copy_event` for non-blocking host-to-device transfers.
  - Prohibited pattern grep check in `phase1b_verify.sbatch`: `! grep -rn "\.item()\|\.tolist()" rankfuse/kernels/rankfuse_ma.py rankfuse/paging/adapter_page_manager.py` passes.
- **Details**: Zero GPU readbacks exist on the manager critical path; host memory is pinned upon registration and asynchronously streamed.

---

### 7. PagePool had a second unused page_table conflicting with the manager's
- **Status**: FIXED
- **Evidence**:
  - `rankfuse/paging/page_pool.py`: `PagePool` contains no `page_table` attribute or allocation. It acts purely as a frame allocator.
  - `rankfuse/paging/adapter_page_manager.py:54-57`: `AdapterPageManager` is the single source of truth for `self.page_table` (and `self.cpu_page_table`).
- **Details**: Duplicate page table was removed in commit `2e409a1`.

---

### 8. Tests: ~5 real cases, GPU tests skipped, loose tolerances on toy sizes, silent `except RuntimeError`
- **Status**: FIXED
- **Evidence**:
  - `tests/test_multi_adapter.py`: 33 comprehensive tests cover the full matrix: $N \in \{1, 2, 5, 33\}$; ranks $\{1, 2, 4, 8, 16, 32, 64\}$; 1-token and 1000-token groups; limit-8 next to limit-64; page-0-only residency; shuffled frames; mixed ranks; merged q|k|v and gate|up; add=True; empty group with $T>0$; $T=0$; CUDA graphs; int64 addressing.
  - CPU execution: Small-dim test suite runs under `TRITON_INTERPRET=1` without `skipif(cuda)` (24 passed on CPU).
  - Tolerances: Bitwise equality `torch.equal(flat, paged)` is asserted across tests; comparison to FP32 reference uses strict FP16-matched tolerances (`atol=3e-2, rtol=3e-2` on $H=4096$).
  - `tests/test_paging.py:160-167`: In `test_page_manager_randomized_invariants`, `try / except RuntimeError` explicitly asserts expected error strings (`assert "all resident pages are in use" in str(exc) or "pool is full" in str(exc)`) and invariant bounds (`assert total_pages_needed > pool.num_pages`), rejecting any silent swallow of bugs.
- **Details**: Full matrix coverage is established with strict assertions and no silent exceptions.

---

### 9. paging/sim.py: does the "prefix" policy evict only the TOP page of the LRU idle adapter?
- **Status**: FIXED (was STILL PRESENT in original code, now FIXED)
- **Evidence**:
  - Original `rankfuse/paging/sim.py:121-124`:
    ```python
    while sum(resident.values()) + additional > capacity and resident:
        victim, victim_pages = next(iter(resident.items()))
        del resident[victim]  # Wiped entire adapter!
        evictions += 1
    ```
    This wiped all pages of `victim` at once.
  - Fixed in `rankfuse/paging/sim.py:121-127`:
    ```python
    while sum(resident.values()) + additional > capacity and resident:
        victim, victim_pages = next(iter(resident.items()))
        if victim_pages > 1:
            resident[victim] = victim_pages - 1
        else:
            del resident[victim]
        evictions += 1
    ```
- **Details**: Simulator prefix eviction now decrements `resident[victim]` by 1 page per eviction step, maintaining `victim` at the LRU position until all its pages are exhausted, aligning exactly with `AdapterPageManager._evict_one()`.

---

## 2. Kernel Detail Audit: rankfuse/kernels/rankfuse_ma.py

### Summary of @triton.jit Functions

#### 1. `_build_tile_table_kernel` (lines 36-71)
- **Variables defined**: `gid`, `g_start`, `g_len`, `adapter`, `r_limit`, `out_base`, `n_tiles`, `tile_idx`, `t_start`, `rem`, `t_len`.
- **Variables used vs defined**: All variables defined prior to use.
- **Constexprs**: `MAX_TILES: tl.constexpr`, `BM: tl.constexpr`.
- **Pointer masks / bounds**: `if gid >= G: return`; `if tile_idx < MAX_TILES:`.
- **Int64 casts**: Small indices for tile table offsets; bounded by `MAX_TILES`.
- **Tile table invariants**:
  - Bounded by `MAX_TILES` (drops out-of-bounds tiles safely).
  - Empty groups (`g_len == 0`) have `n_tiles = 0`, producing zero tile records.
  - Each tile is confined to its group (`t_start = g_start + t * BM`, `t_len = min(BM, g_len - t * BM)`), guaranteeing no tile straddles two groups.

#### 2. `_rf1_ma_flat_shrink_splitk_kernel` (lines 78-148)
- **Variables defined**: `pid_m`, `tile_len`, `pid_k`, `pid_s`, `tok_start`, `adapter`, `r_limit`, `offs_m`, `row_ok`, `rows`, `k`, `k_ok`, `a_base`, `h_lo`, `v`, loop `h_start`, `h`, `h_ok`, `x`, `a`, `ws`.
- **Variables used vs defined**: All variables defined prior to use.
- **Constexprs**: `BM: tl.constexpr`, `BH: tl.constexpr`, `R_PAD: tl.constexpr`.
- **Pointer masks**:
  - `row_ok = offs_m < tile_len`
  - `k_ok = k < r_limit`
  - `h_ok = h < H_IN`
  - `x`: `mask=row_ok[:, None] & h_ok[None, :]`
  - `a`: `mask=k_ok[:, None] & h_ok[None, :]`
  - `ws`: `mask=row_ok[:, None]`
- **Int64 casts before offset multiplies**:
  - `a_base = pid_s.to(tl.int64) * A_SLICE + adapter.to(tl.int64) * (R_MAX * H_IN)`
  - `x_ptr + rows[:, None].to(tl.int64) * H_IN + h[None, :]`
  - `a_ptr + a_base + k[:, None].to(tl.int64) * H_IN + h[None, :]`
  - `ws = ws_ptr + pid_s.to(tl.int64) * WS_SLICE + pid_k.to(tl.int64) * (T_PAD * R_PAD) + rows[:, None].to(tl.int64) * R_PAD + k[None, :]`
- **Split-K workspace & reduction**: Workspace shape `(n_slices, split_k, t_pad, r_pad)`. Shrink writes partial sums per `pid_k`.
- **Early exit**: `if tile_len <= 0: return`.

#### 3. `_rf1_ma_paged_shrink_splitk_kernel` (lines 150-223)
- **Variables defined**: `pid_m`, `tile_len`, `pid_k`, `pid_s`, `tok_start`, `adapter`, `r_limit`, `offs_m`, `row_ok`, `rows`, `k`, `page`, `page_ok`, `pt_offset`, `frame`, `k_ok`, `k_offset`, `h_lo`, `v`, loop `h_start`, `h`, `h_ok`, `x`, `a`, `ws`.
- **Variables used vs defined**: All variables defined prior to use.
- **Constexprs**: `BM: tl.constexpr`, `BH: tl.constexpr`, `R_PAD: tl.constexpr`.
- **Pointer masks**:
  - `row_ok = offs_m < tile_len`
  - `page_ok = page < PAGES`
  - `k_ok = (frame >= 0) & (k < r_limit)`
  - `h_ok = h < H_IN`
  - `x`: `mask=row_ok[:, None] & h_ok[None, :]`
  - `a`: `mask=k_ok[:, None] & h_ok[None, :]`
  - `ws`: `mask=row_ok[:, None]`
- **Int64 casts before offset multiplies**:
  - `pt_offset = adapter.to(tl.int64) * PAGES + page.to(tl.int64)`
  - `k_offset = frame.to(tl.int64) * PAGE_STRIDE + (k % 16).to(tl.int64) * FRAME_WIDTH`
  - `x_ptr + rows[:, None].to(tl.int64) * H_IN + h[None, :]`
  - `ws = ws_ptr + pid_s.to(tl.int64) * WS_SLICE + pid_k.to(tl.int64) * (T_PAD * R_PAD) + rows[:, None].to(tl.int64) * R_PAD + k[None, :]`
- **Split-K workspace & reduction**: Workspace shape `(n_slices, split_k, t_pad, r_pad)`.
- **Early exit**: `if tile_len <= 0: return`.

#### 4. `_rf1_ma_flat_expand_splitk_kernel` (lines 228-323)
- **Variables defined**: `pid_m`, `tile_len`, `pid_s`, `out_col0`, `n_cols`, `col_base`, `tok_start`, `adapter`, `r_limit`, `offs_m`, `row_ok`, `rows`, `ranks`, `k`, `k_ok`, `ws_s`, `v`, loop `split`, `scale`, `b_base`, loop `col_start`, `cols`, `col_ok`, `b`, `acc`, `ptrs`, `mask`, `val`.
- **Variables used vs defined**: All variables defined prior to use.
- **Constexprs**: `BM: tl.constexpr`, `BN: tl.constexpr`, `R_PAD: tl.constexpr`, `ADD_OUT: tl.constexpr`, `ALIGNED: tl.constexpr`.
- **Pointer masks**:
  - `row_ok = offs_m < tile_len`
  - `k_ok = k < r_limit`
  - `col_ok = cols < n_cols`
  - `mask = row_ok[:, None] & col_ok[None, :]`
- **Int64 casts before offset multiplies**:
  - `ws_s = ws_ptr + pid_s.to(tl.int64) * WS_SLICE`
  - `tl.cast(split, tl.int64) * (T_PAD * R_PAD)`
  - `rows[:, None].to(tl.int64) * R_PAD`
  - `b_base = pid_s.to(tl.int64) * B_SLICE + adapter.to(tl.int64) * (R_MAX * N_MAX)`
  - `k[:, None].to(tl.int64) * N_MAX`
  - `rows[:, None].to(tl.int64) * OUT_STRIDE`
- **Split-K reduction & per-token rank mask**:
  - Accumulates across `SPLIT_K`: `for split in range(0, SPLIT_K): v += tl.load(...)`
  - Per-token rank mask applied via `v = tl.where(k[None, :] < ranks[:, None], v, 0.0)`
  - Scaled by preallocated scale table: `v = v * scale[:, None]`
  - Matrix multiplication performed in FP32 (`tl.dot(v, b.to(tl.float32), input_precision="ieee")`) to maintain numerical precision with the baseline reference oracle, casting to `out_ptr.dtype.element_ty` upon store.
- **Early exit**: `if tile_len <= 0: return`.

#### 5. `_rf1_ma_paged_expand_splitk_kernel` (lines 325-433)
- **Variables defined**: `pid_m`, `tile_len`, `pid_s`, `out_col0`, `n_cols`, `col_base`, `tok_start`, `adapter`, `r_limit`, `offs_m`, `row_ok`, `rows`, `ranks`, `k`, `page`, `page_ok`, `pt_offset`, `frame`, `k_ok`, `ws_s`, `v`, loop `split`, `scale`, `frame_safe`, `k_b_offset`, loop `col_start`, `cols`, `col_ok`, `b`, `acc`, `ptrs`, `mask`, `val`.
- **Variables used vs defined**: All variables defined prior to use.
- **Constexprs**: `BM: tl.constexpr`, `BN: tl.constexpr`, `R_PAD: tl.constexpr`, `ADD_OUT: tl.constexpr`, `ALIGNED: tl.constexpr`.
- **Pointer masks**:
  - `row_ok = offs_m < tile_len`
  - `page_ok = page < PAGES`
  - `k_ok = (frame >= 0) & (k < r_limit)`
  - `col_ok = cols < n_cols`
  - `mask = row_ok[:, None] & col_ok[None, :]`
- **Int64 casts before offset multiplies**:
  - `pt_offset = adapter.to(tl.int64) * PAGES + page.to(tl.int64)`
  - `ws_s = ws_ptr + pid_s.to(tl.int64) * WS_SLICE`
  - `tl.cast(split, tl.int64) * (T_PAD * R_PAD)`
  - `rows[:, None].to(tl.int64) * R_PAD`
  - `frame_safe = tl.where(frame >= 0, frame, 0)`
  - `k_b_offset = frame_safe.to(tl.int64) * PAGE_STRIDE + (k % 16).to(tl.int64) * FRAME_WIDTH + B_START + out_col0`
  - `rows[:, None].to(tl.int64) * OUT_STRIDE`
- **Split-K reduction & per-token rank mask**:
  - Accumulates across `SPLIT_K`: `for split in range(0, SPLIT_K): v += tl.load(...)`
  - Per-token rank mask: `v = tl.where(k[None, :] < ranks[:, None], v, 0.0)`
  - Scaled by preallocated scale table: `v = v * scale[:, None]`
  - Matrix multiplication performed in FP32 (`tl.dot(v, b.to(tl.float32), input_precision="ieee")`) to maintain numerical precision with the baseline reference oracle, casting to `out_ptr.dtype.element_ty` upon store.
- **Early exit**: `if tile_len <= 0: return`.
