// Vendored unchanged from S-LoRA (https://github.com/S-LoRA/S-LoRA, commit c1ddf48, slora/csrc/bgmv/bgmv_all.cu).
// Copyright the S-LoRA authors (vec_dtypes.cuh: from FlashInfer/Punica), Apache License 2.0
// (see ../LICENSE). RankFuse adds only this header.

#include "bgmv_config.h"
#include "bgmv_impl.cuh"

FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, nv_half)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, nv_bfloat16)
