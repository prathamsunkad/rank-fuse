# Multi-Adapter Evaluation Results (NVIDIA H200)

> **Environment:** NVIDIA H200, PACE ICE | PyTorch 2.13.0+cu130, Triton 3.7.1, vLLM 0.32.0 | git commit 506dad2ece50ec5de14604cc07ed22b9ef758a5e

## Executive Summary

This directory contains the Phase 2 multi-adapter benchmark and profiling results on NVIDIA H200 (PACE ICE).
All implementations were measured via CUDA graph replays (>= 200 replays) across decode and prefill workloads,
profiling DRAM memory traffic with Nsight Compute and simulating adapter residency under memory pressure.

## Success Criteria Report

| Criterion | Status | Quantitative Evidence |
|---|---|---|
| 1. rf1_ma_flat >= 1.5x vs b3_ma and b4_ma (decode, N >= 16) | **FAIL** | 96 OK configs: min speedup vs b3: 0.65x, b4_copies: 1.51x, b4_shared: 1.52x |
| 2. Page-table overhead rf1_ma / rf1_ma_flat <= 1.05x | **PASS** | 240 OK configs: mean ratio: 1.024x, max ratio: 1.057x |
| 3. Prefix paging beats true-rank LRU on copied bytes & cold starts under Zipf | **PASS** | 24 simulation points: Pressured copied (prefix vs true_rank): 1220.8 GB vs 1250.9 GB |
| 4. NCU: rf1_ma reads fewer DRAM bytes than b3_ma | **PASS** | T=32 N=16 DRAM read: rf1_ma 15.37 MB vs b3_ma 21.29 MB (1.39x reduction) |

## Sweep Implementation Status Breakdown

| Implementation | OK Rows | BROKEN Rows | ERROR Rows | Total Evaluated |
|---|---|---|---|---|
| rf1_ma | 240 | 0 | 0 | 240 |
| rf1_ma_flat | 240 | 0 | 0 | 240 |
| b1_ma_padded | 240 | 0 | 0 | 240 |
| b3_ma | 240 | 0 | 0 | 240 |
| b4_ma_copies | 240 | 0 | 0 | 240 |
| b4_ma_shared | 240 | 0 | 0 | 240 |

## Result Tables

- [01. Speedup: rf1_ma_flat vs B1 Padded](tables/01_speedup_rf1_vs_b1_padded.md)
- [02. Speedup: rf1_ma_flat vs B3 vLLM Copies](tables/02_speedup_rf1_vs_b3_ma.md)
- [03. Speedup: rf1_ma_flat vs B4 S-LoRA Copies](tables/03_speedup_rf1_vs_b4_ma_copies.md)
- [04. Speedup: rf1_ma_flat vs B4 S-LoRA Shared](tables/04_speedup_rf1_vs_b4_ma_shared.md)
- [05. Page-Table Overhead: rf1_ma vs rf1_ma_flat](tables/05_page_table_overhead.md)
- [06. Decode Performance at N >= 16 vs Baselines](tables/06_decode_n16_vs_b3_b4.md)
- [07. Memory Traffic and Hardware Profiling (NCU)](tables/07_ncu_summary.md)
- [08. Pinned Host-to-Device (H2D) Transfer Latencies](tables/08_h2d_latency_bandwidth.md)
- [09. Adapter Residency Simulator Results](tables/09_simulator_policies.md)
- [10. Success Criteria Evaluation](tables/10_success_criteria_report.md)

## Simulator Graphs

- Hit Rate vs Budget: `tables/fig_sim_hit_rate.png`
- H2D Bytes Copied vs Budget: `tables/fig_sim_bytes_copied.png`

