"""MatryoshkaLoRARequest: vLLM's LoRARequest with a trailing `matryoshka_rank` field.

    from rankfuse.integration import MatryoshkaLoRARequest
    req = MatryoshkaLoRARequest("matryoshka-0", 1, path_to_r64_adapter, matryoshka_rank=8)
    llm.generate(prompts, params, lora_request=req)

`matryoshka_rank=None` (the default) means the whole adapter (vLLM's padded max_lora_rank).

LoRARequest is a msgspec struct with array_like=True, so its fields travel between the API
process and the engine core as a positional array. msgspec lets a subclass append fields, so
the rank is the last element and the wire format of the existing fields is unchanged. The
engine-core side must decode into this class (vllm_plugin patches EngineCoreRequest's
annotation for that). Identity stays on lora_name, exactly as LoRARequest: every rank of one
adapter is the same adapter to vLLM's scheduler and slot manager (one lora_int_id, one CPU
cache entry, one GPU slot).
"""

from vllm.lora.request import LoRARequest


class MatryoshkaLoRARequest(LoRARequest):
    matryoshka_rank: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.matryoshka_rank is not None and self.matryoshka_rank < 1:
            raise ValueError(f"matryoshka_rank must be >= 1 or None, got {self.matryoshka_rank}")

    # Same identity as LoRARequest (name only), also across the two classes, so ranks of one
    # adapter collapse to one entry in the set vLLM passes to set_active_adapters.
    def __eq__(self, value: object) -> bool:
        return isinstance(value, LoRARequest) and self.lora_name == value.lora_name

    def __hash__(self) -> int:
        return hash(self.lora_name)
