"""RankFuse Triton kernels (not implemented yet).

Planned, both registered through `rankfuse.interface.register` with the shared signature
`(x [T, h_in], lora: MatryoshkaLoRA, token_ranks [T]) -> y [T, h_out]`:

* rankfuse_mask.py   - v1. Fused shrink + expand. Grid over (token tile, h_out tile); the
                       rank loop goes to the tile's max rank and masks columns >= each token's
                       rank. v stays in registers/SMEM (never written to HBM).
* rankfuse_ragged.py - v2. Host builds a list of only the valid (token tile, rank tile) work
                       items and launches exactly those, so low-rank tokens never touch high
                       rank tiles. Sorting tokens by rank inside the batch makes tiles uniform.

Both must pass tests/test_correctness.py (it picks them up automatically from the registry
once this package imports them).
"""
