"""Stock vLLM loads Qwen2.5-0.5B and every generated Matryoshka adapter (MatLoRA plan, P0.4).

Registers the full rank-64 adapter and each rank-r copy as separate LoRA adapters (the stock
vLLM baseline layout), decodes greedily with each, and checks that the r64 copy produces the
same tokens as the full adapter (they hold identical weights).

    python scripts/make_adapters.py --model qwen2.5-0.5b --r-max 64 --out $RF_ADAPTERS/qwen2.5-0.5b
    python scripts/ice/vllm_lora_check.py $RF_ADAPTERS/qwen2.5-0.5b     # inside a GPU job
"""

import json
import sys
import time
from pathlib import Path

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPTS = [
    "The capital of France is",
    "Explain low-rank adaptation in one sentence:",
    "def fibonacci(n):",
    "Three facts about the H200 GPU:",
]


def main() -> None:
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    root = Path(sys.argv[1])
    meta = json.loads((root / "matryoshka-0" / "matryoshka.json").read_text())
    tiers = meta["tiers"]
    adapters = [("full", root / "matryoshka-0")] + [(f"r{r}", root / f"matryoshka-0-r{r}") for r in tiers]
    print(f"vllm {vllm.__version__}; adapters: {[n for n, _ in adapters]}", flush=True)

    t0 = time.time()
    llm = LLM(
        model=MODEL, enable_lora=True, max_lora_rank=meta["r_max"], max_loras=len(adapters),
        max_model_len=1024, gpu_memory_utilization=0.4, seed=0,
    )
    print(f"engine ready in {time.time() - t0:.0f}s", flush=True)
    params = SamplingParams(temperature=0.0, max_tokens=16)

    def run(lora):
        outs = llm.generate(PROMPTS, params, lora_request=lora, use_tqdm=False)
        return [list(o.outputs[0].token_ids) for o in outs], outs[0].outputs[0].text

    base_ids, base_text = run(None)
    print(f"{'base':>5}: {base_text!r}")
    results = {}
    for i, (name, path) in enumerate(adapters, start=1):
        ids, text = run(LoRARequest(name, i, str(path)))
        results[name] = ids
        changed = sum(a != b for a, b in zip(ids, base_ids))
        print(f"{name:>5}: {text!r}  (prompts differing from base: {changed}/{len(PROMPTS)})")

    same = results["full"] == results[f"r{meta['r_max']}"]
    print(f"full == r{meta['r_max']} copy (identical tokens): {same}")
    print("VLLM_LORA_CHECK", "PASS" if same else "FAIL")


if __name__ == "__main__":
    main()
