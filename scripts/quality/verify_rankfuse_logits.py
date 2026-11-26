"""Step 2 of the serving-path check (shared RankFuse env: torch 2.13, transformers 5.x, Triton).

Patches every LoRA target projection of the HF base model with base(x) + delta, where delta is
computed by a RankFuse implementation (b0_reference or rf1_fused) from the *full* r_max adapter
(A_t = lora_A.weight, B = lora_B.weight^T) and a per-token rank vector. Compares with the PEFT
logits saved by verify_peft_logits.py, (a) with every sequence at rank r, and (b) with one mixed
batch whose sequences carry different ranks (the per-request knob).
    python verify_rankfuse_logits.py --model <path> --adapter <dir> --peft f.pt --out report.json
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

import rankfuse  # noqa: F401  (registers kernels)
from rankfuse.interface import REGISTRY, MatryoshkaLoRA

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
STATE = {"impl": None, "token_ranks": None}


class Patched(nn.Module):
    def __init__(self, base, lora):
        super().__init__()
        self.base, self.lora = base, lora

    def forward(self, x):
        y = self.base(x)
        impl = REGISTRY[STATE["impl"]]
        x2 = x.reshape(-1, x.shape[-1])
        d = impl(x2, self.lora, STATE["token_ranks"])
        return y + d.to(y.dtype).reshape(*y.shape[:-1], -1)


def cmp(a, b):
    lp_a, lp_b = a.log_softmax(-1), b.log_softmax(-1)
    kl = (lp_a.exp() * (lp_a - lp_b)).sum(-1)
    return dict(max_abs=(a - b).abs().max().item(), mean_abs=(a - b).abs().mean().item(),
                argmax_agree=(a.argmax(-1) == b.argmax(-1)).float().mean().item(), max_kl=kl.max().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--peft", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--impls", default="b0_reference,rf1_fused")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    a = ap.parse_args()
    ref = torch.load(a.peft)
    ids = ref["input_ids"].cuda()
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).cuda().eval()
    w = load_file(str(Path(a.adapter) / "adapter_model.safetensors"), device="cuda")
    n = 0
    for name, mod in list(model.named_modules()):
        for cn, child in list(mod.named_children()):
            if cn in TARGETS and isinstance(child, nn.Linear):
                key = f"base_model.model.{name}.{cn}"
                lora = MatryoshkaLoRA(A_t=w[key + ".lora_A.weight"].to(getattr(torch, a.dtype)).contiguous(),
                                      B=w[key + ".lora_B.weight"].T.to(getattr(torch, a.dtype)).contiguous())
                setattr(mod, cn, Patched(child, lora))
                n += 1
    B, L = ids.shape
    report = {"dtype": a.dtype, "patched_layers": n, "batch": [B, L], "results": []}

    def run(impl, seq_ranks):
        STATE["impl"] = impl
        STATE["token_ranks"] = torch.tensor(seq_ranks, dtype=torch.int32).repeat_interleave(L).cuda()
        with torch.no_grad():
            return model(input_ids=ids).logits.float().cpu()

    # base-model control (rank 0 everywhere): cross-env / cross-transformers-version drift
    out0 = run("b0_reference", [0] * B)
    report["results"].append(dict(impl="b0_reference", case="base (rank 0)", **cmp(out0, ref["logits"][0])))
    for impl in a.impls.split(","):
        for r in ref["ranks"]:
            out = run(impl, [r] * B)
            report["results"].append(dict(impl=impl, case=f"all r={r}", **cmp(out, ref["logits"][r]),
                                          vs_base_max_abs=(out - out0).abs().max().item()))
        ranks = ref["ranks"]
        mixed = [ranks[i % len(ranks)] for i in range(B)]
        out = run(impl, mixed)
        for i, r in enumerate(mixed):
            report["results"].append(dict(impl=impl, case=f"mixed batch {mixed}: seq {i} at r={r}",
                                          **cmp(out[i : i + 1], ref["logits"][r][i : i + 1])))
    for row in report["results"]:
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
    Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
