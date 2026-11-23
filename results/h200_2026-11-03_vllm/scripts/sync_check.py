"""Does a decode/prefill step make a GPU->CPU sync? vLLM's own checker: VLLM_GPU_SYNC_CHECK=error
raises on any synchronizing CUDA call after warm-up that is not inside gpu_sync_allowed().
Stock vLLM wraps its LoRA metadata build (torch.unique / sort + a .item()) in gpu_sync_allowed();
RankFuse computes r_limit on the host and makes no such call.

    VLLM_GPU_SYNC_CHECK=error [RANKFUSE_VLLM=1] python sync_check.py <graphs|eager>
"""
import os
import sys

from vllm import LLM, SamplingParams

path = os.path.join(os.environ["RF_ADAPTERS"], "qwen2.5-0.5b", "matryoshka-0")
rf = os.environ.get("RANKFUSE_VLLM") == "1"
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", enable_lora=True, max_lora_rank=64, max_loras=1, max_model_len=2048,
          gpu_memory_utilization=0.3, enforce_eager=sys.argv[1] == "eager")
if rf:
    from rankfuse.integration import MatryoshkaLoRARequest
    reqs = [MatryoshkaLoRARequest("matryoshka-0", 1, path, matryoshka_rank=r) for r in (64, 1, 8, 32, 2, 16, 4, 64)]
else:
    from vllm.lora.request import LoRARequest
    reqs = [LoRARequest("matryoshka-0", 1, path)] * 8
ps = [f"Tell me a fact about the number {i}." * (1 + i) for i in range(8)]
outs = llm.generate(ps, SamplingParams(temperature=0, max_tokens=32), lora_request=reqs, use_tqdm=False)
print(f"SYNC CHECK PASSED ({'rankfuse' if rf else 'stock'}, {sys.argv[1]}): {sum(len(o.outputs[0].token_ids) for o in outs)} tokens, "
      f"VLLM_GPU_SYNC_CHECK={os.environ.get('VLLM_GPU_SYNC_CHECK')}")
