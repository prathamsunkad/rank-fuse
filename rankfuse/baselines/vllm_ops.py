"""B3 (TODO): vLLM's Triton multi-LoRA kernels (Punica-derived shrink/expand).

Not implemented yet. Plan:
1. Pin a vLLM commit and either `pip install vllm==<ver>` on Phoenix or copy the two Triton
   op files from `vllm/lora/ops/triton_ops/` (shrink + expand) into `third_party/vllm_lora/`
   with the license header. Copying avoids pulling in the whole engine.
2. Register each Matryoshka prefix as its OWN adapter, exactly as you would have to in vLLM
   today: adapter k = (A_t[:r_k], B[:r_k]) zero-padded to `max_lora_rank` = lora.r_max.
   That means building stacked buffers of shape [num_distinct_ranks, r_max, h] - duplicate
   storage of the shared prefix is the point of this baseline (Claim 1).
3. Map each token to its adapter index (rank -> adapter id) and call shrink then expand.
4. Verify with ncu that the kernel actually reads the zero-padded rows; the roofline model
   (`rankfuse.roofline`, impl "separate_padded") assumes it does.

Register as `b3_vllm` with requires_cuda=True once working.
"""
