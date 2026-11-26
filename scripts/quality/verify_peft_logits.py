"""Step 1 of the serving-path check (mlora env: transformers 4.57.6 + PEFT).

Loads the base model and the rank-r prefix of an exported Matryoshka adapter through PEFT and saves
the logits of a fixed batch, for comparison with RankFuse's B0 reference and RF1 kernel
(verify_rankfuse_logits.py, shared env).
    python verify_peft_logits.py --model <path> --adapter <full r_max PEFT dir> --ranks 8,64,256 --out f.pt
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer:",
    "### Instruction:\nExplain why the sky is blue in two sentences, and then say what color a sunset is and why.\n\n### Response:\nThe sky",
    "The capital of France is Paris. The Eiffel Tower was completed in 1889 and is one of the most visited monuments",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    if n < 2:\n        return n\n",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--ranks", default="8,64,256")
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    enc = [tok(p)["input_ids"] for p in PROMPTS]
    L = min(a.seq, *map(len, enc))  # equal lengths, no padding
    ids = torch.tensor([e[:L] for e in enc]).cuda()
    res = {"input_ids": ids.cpu(), "ranks": [], "logits": {}}
    base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype)).cuda().eval()
    with torch.no_grad():
        res["logits"][0] = base(input_ids=ids).logits.float().cpu()
    full = load_file(str(Path(a.adapter) / "adapter_model.safetensors"))
    cfg = json.loads((Path(a.adapter) / "adapter_config.json").read_text())
    for r in [int(x) for x in a.ranks.split(",")]:
        d = Path(tempfile.mkdtemp(dir=Path(a.out).parent))
        sl = {k: (v[:r] if "lora_A" in k else v[:, :r]).contiguous() for k, v in full.items()}
        save_file(sl, str(d / "adapter_model.safetensors"))
        (d / "adapter_config.json").write_text(json.dumps(dict(cfg, r=r, lora_alpha=float(r))))
        m = PeftModel.from_pretrained(base, str(d)).eval()  # PEFT upcasts bf16 adapter weights to fp32
        with torch.no_grad():
            res["logits"][r] = m(input_ids=ids).logits.float().cpu()
        base = m.unload()
        shutil.rmtree(d)
        print("rank", r, "max|peft - base| =", (res["logits"][r] - res["logits"][0]).abs().max().item())
    res["dtype"] = a.dtype
    res["ranks"] = sorted(k for k in res["logits"] if k)
    torch.save(res, a.out)


if __name__ == "__main__":
    main()
