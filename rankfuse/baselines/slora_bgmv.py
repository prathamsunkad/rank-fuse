"""B4 (TODO, optional): S-LoRA's heterogeneous-rank kernel.

Not implemented yet. Plan:
1. Add S-LoRA as a pinned submodule under `third_party/S-LoRA`. The relevant code is the
   modified Punica bgmv CUDA kernel in `slora/csrc/` (takes per-request ranks and locations in
   the unified memory pool). Time-box the build on Phoenix to 2-3 days: the repo targets 2023
   PyTorch/CUDA versions.
2. Register each Matryoshka prefix as a separate adapter at its TRUE rank (no padding): the
   memory pool holds rows for rank 4, 8, 16, 32 separately -> sum of distinct ranks rows.
3. If the build fails, write a faithful Triton re-implementation (per-token gather of that
   token's adapter rows, loop over its true rank) and label it "S-LoRA-style" in the report.

Register as `b4_slora` with requires_cuda=True once working.
"""
