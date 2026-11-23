"""RankFuse serving integrations.

    vllm_plugin   vLLM 0.32 `vllm.general_plugins` entry point, active with RANKFUSE_VLLM=1
    request       MatryoshkaLoRARequest (LoRARequest + matryoshka_rank)
    punica        RankFusePunicaWrapper (per-token rank buffer, the kernel swap)
    ops           torch.ops.rankfuse.lora_linear (the custom op the wrapper launches)

Importing this package imports nothing from vLLM; `MatryoshkaLoRARequest` is loaded on first use.
"""


def __getattr__(name):
    if name == "MatryoshkaLoRARequest":
        from rankfuse.integration.request import MatryoshkaLoRARequest

        return MatryoshkaLoRARequest
    raise AttributeError(name)
