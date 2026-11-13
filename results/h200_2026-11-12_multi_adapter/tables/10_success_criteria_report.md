### Phase 2 Success Criteria Evaluation

> **Environment:** NVIDIA H200, PACE ICE | PyTorch 2.13.0+cu130, Triton 3.7.1, vLLM 0.32.0 | git commit 506dad2ece50ec5de14604cc07ed22b9ef758a5e

Evaluation of four Phase 2 success criteria on NVIDIA H200.

| Criterion | Status | Quantitative Evidence |
|---|---|---|
| 1. rf1_ma_flat >= 1.5x vs b3_ma and b4_ma (decode, N >= 16) | FAIL | 96 OK configs: min speedup vs b3: 0.65x, b4_copies: 1.51x, b4_shared: 1.52x |
| 2. Page-table overhead rf1_ma / rf1_ma_flat <= 1.05x | PASS | 240 OK configs: mean ratio: 1.024x, max ratio: 1.057x |
| 3. Prefix paging beats true-rank LRU on copied bytes & cold starts under Zipf | PASS | 24 simulation points: Pressured copied (prefix vs true_rank): 1220.8 GB vs 1250.9 GB |
| 4. NCU: rf1_ma reads fewer DRAM bytes than b3_ma | PASS | T=32 N=16 DRAM read: rf1_ma 15.37 MB vs b3_ma 21.29 MB (1.39x reduction) |

### Sweep Implementation Status Breakdown

| Implementation | OK Rows | BROKEN Rows | ERROR Rows | Total Evaluated |
|---|---|---|---|---|
| rf1_ma | 240 | 0 | 0 | 240 |
| rf1_ma_flat | 240 | 0 | 0 | 240 |
| b1_ma_padded | 240 | 0 | 0 | 240 |
| b3_ma | 240 | 0 | 0 | 240 |
| b4_ma_copies | 240 | 0 | 0 | 240 |
| b4_ma_shared | 240 | 0 | 0 | 240 |
