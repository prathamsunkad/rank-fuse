"""Write random Matryoshka LoRA adapters in Hugging Face PEFT format for vLLM experiments.

    python scripts/make_adapters.py --model llama3-8b --r-max 64 --out adapters/llama3-8b
    python scripts/make_adapters.py --config path/to/config.json --base-model my/model \
        --r-max 64 --num-adapters 8 --out adapters/my-model

For each adapter i (seed --seed + i) this writes, under --out:

    matryoshka-{i}/          the full rank-r_max adapter, plus matryoshka.json listing the
                             rank tiers. RankFuse loads this once and serves every tier from it.
    matryoshka-{i}-r{r}/     the rank-r prefix as its own adapter, one per tier. Stock vLLM
                             cannot use prefixes, so its baseline registers these separately.

Each directory holds adapter_config.json and adapter_model.safetensors (or adapter_model.bin
when the safetensors package is missing, or with --format bin).

Layout. For a projection with weight [out, in], PEFT stores lora_A.weight [r, in] (the same
layout as RankFuse's A_t) and lora_B.weight [out, r] (B transposed). The rank-r copy is
lora_A[:r] and lora_B[:, :r] of the full adapter, so every copy is an exact prefix.

Scaling. PEFT multiplies the delta by lora_alpha / r. Each copy gets lora_alpha = scale * r, so
every tier runs at the same effective scale (--scale, default 1.0); matryoshka.json records it.

Weights are random: A ~ N(0, 1/in), B ~ N(0, init_scale^2 / r). Timing does not depend on the
values. Model dimensions come from a preset (check it against the model's config.json) or from
a config.json passed with --config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

PRESETS = {
    "llama3-8b": dict(
        base_model="meta-llama/Llama-3.1-8B-Instruct", hidden_size=4096, intermediate_size=14336,
        num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8, head_dim=128,
    ),
    "llama3.2-1b": dict(
        base_model="meta-llama/Llama-3.2-1B-Instruct", hidden_size=2048, intermediate_size=8192,
        num_hidden_layers=16, num_attention_heads=32, num_key_value_heads=8, head_dim=64,
    ),
    "qwen2.5-7b": dict(
        base_model="Qwen/Qwen2.5-7B-Instruct", hidden_size=3584, intermediate_size=18944,
        num_hidden_layers=28, num_attention_heads=28, num_key_value_heads=4, head_dim=128,
    ),
    "qwen2.5-0.5b": dict(
        base_model="Qwen/Qwen2.5-0.5B-Instruct", hidden_size=896, intermediate_size=4864,
        num_hidden_layers=24, num_attention_heads=14, num_key_value_heads=2, head_dim=64,
    ),
}

ALL_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def default_tiers(r_max: int) -> list[int]:
    """Powers of two from 1 up to r_max, as in rankfuse.workloads.default_tiers."""
    tiers = [1]
    while tiers[-1] * 2 < r_max:
        tiers.append(tiers[-1] * 2)
    return tiers + [r_max] if r_max > 1 else [1]


def dims_from_config(cfg: dict) -> dict:
    """Model dimensions from a Hugging Face config.json dict (Llama/Qwen-style decoders)."""
    heads = cfg["num_attention_heads"]
    return dict(
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=heads,
        num_key_value_heads=cfg.get("num_key_value_heads", heads),
        head_dim=cfg.get("head_dim") or cfg["hidden_size"] // heads,
    )


def projection_shapes(dims: dict) -> dict[str, tuple[str, int, int]]:
    """target module -> (parent module, in_features, out_features)."""
    h, inter = dims["hidden_size"], dims["intermediate_size"]
    q_out = dims["num_attention_heads"] * dims["head_dim"]
    kv_out = dims["num_key_value_heads"] * dims["head_dim"]
    return {
        "q_proj": ("self_attn", h, q_out),
        "k_proj": ("self_attn", h, kv_out),
        "v_proj": ("self_attn", h, kv_out),
        "o_proj": ("self_attn", q_out, h),
        "gate_proj": ("mlp", h, inter),
        "up_proj": ("mlp", h, inter),
        "down_proj": ("mlp", inter, h),
    }


def build_adapter(
    dims: dict, r_max: int, targets: list[str], seed: int, dtype: torch.dtype, init_scale: float
) -> dict[str, torch.Tensor]:
    """Full rank-r_max adapter as a PEFT state dict."""
    gen = torch.Generator().manual_seed(seed)
    shapes = projection_shapes(dims)
    tensors = {}
    for layer in range(dims["num_hidden_layers"]):
        for name in targets:
            parent, fan_in, fan_out = shapes[name]
            prefix = f"base_model.model.model.layers.{layer}.{parent}.{name}"
            a = torch.randn(r_max, fan_in, generator=gen) / fan_in**0.5
            b = torch.randn(fan_out, r_max, generator=gen) * (init_scale / r_max**0.5)
            tensors[f"{prefix}.lora_A.weight"] = a.to(dtype)
            tensors[f"{prefix}.lora_B.weight"] = b.to(dtype)
    return tensors


def slice_adapter(tensors: dict[str, torch.Tensor], r: int) -> dict[str, torch.Tensor]:
    """The rank-r prefix of a Matryoshka adapter: lora_A[:r], lora_B[:, :r]."""
    out = {}
    for key, t in tensors.items():
        out[key] = (t[:r] if key.endswith("lora_A.weight") else t[:, :r]).contiguous()
    return out


def peft_config(base_model: str, r: int, scale: float, targets: list[str]) -> dict:
    return {
        "base_model_name_or_path": base_model,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": r,
        "lora_alpha": scale * r,
        "lora_dropout": 0.0,
        "target_modules": targets,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "modules_to_save": None,
        "use_rslora": False,
        "use_dora": False,
    }


def write_adapter(directory: Path, tensors: dict[str, torch.Tensor], config: dict, fmt: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(json.dumps(config, indent=2))
    if fmt == "safetensors":
        from safetensors.torch import save_file

        save_file(tensors, str(directory / "adapter_model.safetensors"))
    else:
        torch.save(tensors, directory / "adapter_model.bin")


def resolve_format(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import safetensors  # noqa: F401
    except ImportError:
        print("safetensors is not installed; writing adapter_model.bin instead")
        return "bin"
    return "safetensors"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", choices=sorted(PRESETS), help="built-in model dimensions")
    src.add_argument("--config", type=Path, help="a Hugging Face config.json to read dimensions from")
    p.add_argument("--base-model", help="base_model_name_or_path to record (default: the preset's)")
    p.add_argument("--r-max", type=int, required=True)
    p.add_argument("--tiers", type=int, nargs="+", help="rank tiers (default: 1, 2, 4, ..., r_max)")
    p.add_argument("--targets", nargs="+", default=ALL_TARGETS, choices=ALL_TARGETS)
    p.add_argument("--num-adapters", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scale", type=float, default=1.0, help="effective LoRA scale for every tier")
    p.add_argument("--init-scale", type=float, default=0.02, help="std of B relative to 1/sqrt(r)")
    p.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    p.add_argument("--format", choices=["auto", "safetensors", "bin"], default="auto")
    p.add_argument("--no-slices", action="store_true", help="skip the per-tier copies")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)

    if a.model:
        dims = dict(PRESETS[a.model])
        base_model = a.base_model or dims.pop("base_model")
        dims.pop("base_model", None)
    else:
        dims = dims_from_config(json.loads(a.config.read_text()))
        base_model = a.base_model or str(a.config.parent)
    tiers = a.tiers or default_tiers(a.r_max)
    if sorted(set(tiers)) != tiers or tiers[-1] != a.r_max or tiers[0] < 1:
        raise SystemExit(f"tiers must be increasing, positive, and end at r_max; got {tiers}")
    fmt = resolve_format(a.format)
    dtype = DTYPES[a.dtype]

    for i in range(a.num_adapters):
        tensors = build_adapter(dims, a.r_max, a.targets, a.seed + i, dtype, a.init_scale)
        full_dir = a.out / f"matryoshka-{i}"
        write_adapter(full_dir, tensors, peft_config(base_model, a.r_max, a.scale, a.targets), fmt)
        meta = {"r_max": a.r_max, "tiers": tiers, "scale": a.scale, "seed": a.seed + i, "dims": dims}
        (full_dir / "matryoshka.json").write_text(json.dumps(meta, indent=2))
        written = [full_dir]
        if not a.no_slices:
            for r in tiers:
                d = a.out / f"matryoshka-{i}-r{r}"
                write_adapter(d, slice_adapter(tensors, r), peft_config(base_model, r, a.scale, a.targets), fmt)
                written.append(d)
        size_mb = sum(t.numel() * t.element_size() for t in tensors.values()) / 2**20
        print(f"adapter {i}: {size_mb:.1f} MiB at rank {a.r_max}; wrote {', '.join(d.name for d in written)}")


if __name__ == "__main__":
    main()
