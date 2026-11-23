### HBM traffic per call at decode T = 32, uniform ranks (Nsight Compute)

scripts/ice/ncu_profile.sh <impl> 32 1 4096 4096 64 uniform fp16: one profiled call (per-step metadata built before it), ncu defaults (caches flushed, clocks locked to base, so kernel times are longer than in the timing tables). Sums over every kernel of the call, including buffer zeroing (memset kernels). One NVIDIA H200, fp16, 4096 x 4096, r_max 64.

| impl | kernels | kernel time us (sum) | DRAM read MB | DRAM write MB | DRAM total MB | L2 traffic MB | kernel names |
|---|---|---|---|---|---|---|---|
| b3_vllm | 4 | 20.93 | 7.92 | 0.00 | 7.92 | 16.99 | void at::vectorized_elementwise_kernel<8; void at::vectorized_elementwise_kernel<4; _lora_shrink_kernel; _lora_expand_kernel |
| b4_slora_copies | 4 | 19.04 | 2.91 | 0.00 | 2.91 | 16.16 | void at::vectorized_elementwise_kernel<8; void at::vectorized_elementwise_kernel<8; void bgmv_multi_lora_rank_shrink_kernel<; void bgmv_multi_lora_rank_expand_kernel< |
| b4_slora_shared | 4 | 19.58 | 1.60 | 0.00 | 1.60 | 22.09 | void at::vectorized_elementwise_kernel<8; void at::vectorized_elementwise_kernel<8; void bgmv_multi_lora_rank_shrink_kernel<; void bgmv_multi_lora_rank_expand_kernel< |
| rf1_fused | 2 | 10.18 | 1.46 | 0.00 | 1.46 | 4.98 | _rf1_shrink_splitk_kernel; _rf1_expand_splitk_kernel |
