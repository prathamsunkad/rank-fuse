"""Export a trained adapter (lora_final.pt from mlora_train.py) to Hugging Face PEFT format.

Layout matches scripts/make_adapters.py: lora_A.weight = A_t[:r] ([r, in], RankFuse's A_t) and
lora_B.weight = B[:r].T ([out, r]); lora_alpha = r so PEFT's alpha/r scale is 1.0 at every rank
(the paper's s_r = 1). Rank r of the full adapter is lora_A[:r], lora_B[:, :r].

  --order prefix  keep the trained order (Matryoshka, or naive truncation of ordinary LoRA)
  --order svd     re-factor each layer's delta B^T A into its SVD (sigma descending, split as
                  sqrt(sigma) on both sides), so the rank-r prefix is the best rank-r approximation
                  of that layer's delta (Eckart-Young). Computed in fp64 from the fp32 weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def svd_reorder(A_t: torch.Tensor, B: torch.Tensor):
    """A_t [R, in], B [R, out]; delta^T = A_t^T B ([in, out]). Returns (A_t', B') in SVD order."""
    A = A_t.double()
    Bd = B.double()
    Qa, Ra = torch.linalg.qr(A.T)  # A^T = Qa Ra, Qa [in, R]
    Qb, Rb = torch.linalg.qr(Bd.T)  # B^T = Qb Rb, Qb [out, R]
    # delta = B^T A = Qb (Rb Ra^T) Qa^T
    U, S, Vh = torch.linalg.svd(Rb @ Ra.T, full_matrices=False)  # k = min(R, in, out) directions
    s = S.sqrt()
    B_new = (Qb @ U * s).T  # [k, out]
    A_new = s[:, None] * (Vh @ Qa.T)  # [k, in]
    R = A_t.shape[0]
    if B_new.shape[0] < R:  # delta has rank <= k < R (e.g. a 128-wide k_proj): pad with zero directions
        A_new = torch.cat([A_new, A_new.new_zeros(R - A_new.shape[0], A_new.shape[1])])
        B_new = torch.cat([B_new, B_new.new_zeros(R - B_new.shape[0], B_new.shape[1])])
    return A_new.float(), B_new.float()


def to_peft(state: dict, r: int, order: str, dtype=torch.bfloat16):
    out = {}
    for name, (A_t, B) in state.items():
        if order == "svd":
            A_t, B = svd_reorder(A_t, B)
        out[f"base_model.model.{name}.lora_A.weight"] = A_t[:r].to(dtype).contiguous()
        out[f"base_model.model.{name}.lora_B.weight"] = B[:r].T.to(dtype).contiguous()
    return out


def peft_config(base_model: str, r: int) -> dict:
    return {
        "base_model_name_or_path": base_model, "peft_type": "LORA", "task_type": "CAUSAL_LM",
        "r": r, "lora_alpha": float(r), "lora_dropout": 0.0, "target_modules": TARGETS, "bias": "none",
        "fan_in_fan_out": False, "inference_mode": True, "modules_to_save": None,
        "use_rslora": False, "use_dora": False,
    }


def write(directory: Path, tensors: dict, base_model: str, r: int, extra: dict | None = None):
    directory.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(directory / "adapter_model.safetensors"))
    (directory / "adapter_config.json").write_text(json.dumps(peft_config(base_model, r), indent=2))
    if extra is not None:
        (directory / "matryoshka.json").write_text(json.dumps(extra, indent=2))


def load_state(src: Path) -> dict:
    """{module name: (A_t [R, in], B [R, out])} from lora_final.pt or from an exported PEFT folder."""
    src = Path(src)
    if src.is_dir():
        from safetensors.torch import load_file

        w = load_file(str(src / "adapter_model.safetensors"))
        names = sorted({k[len("base_model.model."):-len(".lora_A.weight")] for k in w if k.endswith(".lora_A.weight")})
        return {n: (w[f"base_model.model.{n}.lora_A.weight"], w[f"base_model.model.{n}.lora_B.weight"].T.contiguous())
                for n in names}
    return torch.load(src, map_location="cpu", weights_only=False)


def export(lora_final: Path, out: Path, base_model: str, r: int, order: str, extra: dict | None = None):
    state = load_state(lora_final)
    write(out, to_peft(state, r, order), base_model, r, extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-final", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-model", required=True, help="HF id recorded in adapter_config.json")
    ap.add_argument("--rank", type=int, required=True, help="prefix rank to export")
    ap.add_argument("--order", choices=["prefix", "svd"], default="prefix")
    a = ap.parse_args()
    export(Path(a.lora_final), Path(a.out), a.base_model, a.rank, a.order)


if __name__ == "__main__":
    main()
