### Pinned Host-to-Device (H2D) Transfer Latencies

> **Environment:** NVIDIA H200, PACE ICE | PyTorch 2.13.0+cu130, Triton 3.7.1, vLLM 0.32.0 | git commit 506dad2ece50ec5de14604cc07ed22b9ef758a5e

Pinned host-to-device (H2D) PCIe transfer latencies and sustained bandwidth on H200. Measured with cudaEvent timing over >= 50 repeats for single projection (4096x4096) and whole-model Qwen2.5-7B sizes (page 81 MB, adapter 323 MB).

| scope | label | pages | size MB | median ms | p95 ms | median GB/s | p95 GB/s |
|---|---|---|---|---|---|---|---|
| Single 4096x4096 projection | 1 page (r=16) | 1 | 0.25 | 0.01 | 0.01 | 22.38 | 21.72 |
| Single 4096x4096 projection | 2 pages (r=32) | 2 | 0.50 | 0.02 | 0.02 | 31.88 | 30.96 |
| Single 4096x4096 projection | 4 pages (r=64) | 4 | 1.00 | 0.03 | 0.03 | 39.67 | 39.33 |
| Single 4096x4096 projection | Full adapter (r=64) | 4 | 1.00 | 0.03 | 0.03 | 39.86 | 39.48 |
| Whole-model Qwen2.5-7B | 1 page | 1 | 81.00 | 1.54 | 1.55 | 55.00 | 54.88 |
| Whole-model Qwen2.5-7B | 2 pages | 2 | 162.0 | 3.08 | 3.08 | 55.14 | 55.08 |
| Whole-model Qwen2.5-7B | 4 pages | 4 | 324.0 | 6.15 | 6.16 | 55.22 | 55.16 |
| Whole-model Qwen2.5-7B | Full adapter (r=64) | 4 | 323.0 | 6.13 | 6.14 | 55.22 | 55.17 |
