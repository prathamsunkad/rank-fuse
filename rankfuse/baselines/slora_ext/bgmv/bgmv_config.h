// Vendored from S-LoRA (https://github.com/S-LoRA/S-LoRA, commit c1ddf48, slora/csrc/bgmv/bgmv_config.h),
// Copyright the S-LoRA authors, Apache License 2.0 (see ../LICENSE).
//
// Changes for RankFuse (B4 baseline, rankfuse/baselines/slora_bgmv.py):
//   * bgmv_kernel takes a cudaStream_t (the caller's current PyTorch stream).
//   * Narrow (rank) sizes are 8, 16, 32, 64 instead of 8..128: RankFuse's r_max is 64, and dropping
//     128 keeps every narrow size below every wide size (128 as both would give lora_ops.cc duplicate
//     case labels).
//   * Wide (hidden) sizes are the ones the RankFuse benchmarks use instead of S-LoRA's Llama list:
//     Qwen2.5-0.5B (128, 896, 4864), the 4096 x 4096 test layer, 512 and 1024 for small tests, and
//     Qwen2.5-7B / Llama-3-8B (3584, 14336, 18944) for later. Every wide size is a multiple of 128,
//     which the expand kernel needs (feat_out % (ty * tz) == 0 for every narrow size).

#pragma once

#include <cuda_runtime.h>

template <int feat_in, int feat_out, typename T>
void bgmv_kernel(T* __restrict__ Y, const T* __restrict__ X,
                 const T* __restrict__ W, const int64_t* __restrict__ start_indicies,
                 const int64_t* __restrict__ lora_ranks, const int64_t* __restrict__ loc_indicies,
                 const int64_t* __restrict__ indicies, int64_t qkvo, int64_t batch_size,
                 const T* __restrict__ lora_scales, cudaStream_t stream);

// clang-format off

#define FOR_BGMV_WIDE(f, T, narrow) \
    f(T, narrow, 128) \
    f(T, narrow, 512) \
    f(T, narrow, 896) \
    f(T, narrow, 1024) \
    f(T, narrow, 3584) \
    f(T, narrow, 4096) \
    f(T, narrow, 4864) \
    f(T, narrow, 14336) \
    f(T, narrow, 18944) \

#define FOR_BGMV_WIDE_NARROW(f, T) \
    FOR_BGMV_WIDE(f, T, 8) \
    FOR_BGMV_WIDE(f, T, 16) \
    FOR_BGMV_WIDE(f, T, 32) \
    FOR_BGMV_WIDE(f, T, 64)

// clang-format on
