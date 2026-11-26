"""GPU memory of adapters vs KV cache, and how many adapters fit under a fixed budget.

Two sources:

1. Measured, from a vLLM server log (parse_vllm_log): "Model loading took X GiB" (weights plus
   LoRA slot buffers, which vLLM allocates while loading the model, plus RankFuse's rank-major
   copies when the plugin is on), "Available KV cache memory: X GiB" and "GPU KV cache size: N
   tokens". Adapter memory of a LoRA engine = its model-loading memory - the base engine's.

2. Analytic (lora_bytes / fit_table), from the model's config.json, for a dense Qwen2/Llama
   decoder with LoRA on q, k, v, o, gate, up, down. Bytes per adapter "slot" at rank r:

       per layer, A: r * (3 * hidden [q, k, v each have their own A] + q_out [o] + 2 * hidden
                         [gate, up] + inter [down])
                  B: r * (q_out + 2 * kv_out [q, k, v] + hidden [o] + 2 * inter + hidden [down])

   times layers times dtype bytes. How each system stores a Matryoshka adapter so that every
   request can run at its own rank t in `tiers`:

       stock vLLM    one slot per (adapter, tier), every slot padded to max_lora_rank
                     -> len(tiers) * slot(r_pad)
       S-LoRA-like   one unpadded copy per (adapter, tier) in a unified pool
                     -> slot(sum(tiers))
       RankFuse      one copy at r_max serves every tier -> slot(r_max)
         Stage 1     as built today: vLLM's slot buffer + a rank-major copy -> 2 * slot(r_max),
                     B copy of q|k|v padded to the widest slice (one adapter only)
       RankFuse +    only the rank rows in use are resident (16-row pages):
         paging      -> slot(ceil(max rank used / 16) * 16)

    python bench/e2e/memory.py fit --config <hf config.json> --r-max 256 \
        --tiers 8 16 32 64 128 256 --budget-gb 4 8 16
    python bench/e2e/memory.py log <server.log>
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

GIB = 1024**3


def dims(cfg: dict) -> dict:
    h = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    hd = cfg.get("head_dim") or h // heads
    kv = cfg.get("num_key_value_heads", heads)
    return dict(hidden=h, inter=cfg["intermediate_size"], layers=cfg["num_hidden_layers"],
                q_out=heads * hd, kv_out=kv * hd)


def lora_bytes(d: dict, r: int, dtype_bytes: int = 2) -> int:
    """One adapter slot at rank r (all 7 projections, all layers), vLLM layout."""
    h, i, q, kv = d["hidden"], d["inter"], d["q_out"], d["kv_out"]
    a = r * (3 * h + q + 2 * h + i)
    b = r * (q + 2 * kv + h + 2 * i + h)
    return (a + b) * d["layers"] * dtype_bytes


def rankfuse_stage1_copy_bytes(d: dict, r: int, dtype_bytes: int = 2) -> int:
    """The plugin's rank-major copies A [S, r, h_in], B [S, r, n_max] per layer (one slot)."""
    h, i, q, kv = d["hidden"], d["inter"], d["q_out"], d["kv_out"]
    a = r * (3 * h + q + 2 * h + i)
    b = r * (3 * max(q, kv) + h + 2 * i + h)
    return (a + b) * d["layers"] * dtype_bytes


def r_pad_vllm(r: int) -> int:
    for p in (1, 8, 16, 32, 64, 128, 256, 320, 512):
        if p >= r:
            return p
    raise ValueError(r)


def per_adapter_bytes(d: dict, r_max: int, tiers: list[int], dtype_bytes: int = 2) -> dict:
    r_pad = r_pad_vllm(r_max)
    return {
        "stock vLLM (per-tier slots, padded)": len(tiers) * lora_bytes(d, r_pad, dtype_bytes),
        "stock vLLM (one r_max slot, every request at r_max)": lora_bytes(d, r_pad, dtype_bytes),
        "S-LoRA-like (per-tier unpadded copies)": lora_bytes(d, sum(tiers), dtype_bytes),
        "RankFuse (one r_max copy)": lora_bytes(d, r_max, dtype_bytes),
        "RankFuse Stage 1 as built (slot + rank-major copy)": lora_bytes(d, r_pad, dtype_bytes)
        + rankfuse_stage1_copy_bytes(d, r_max, dtype_bytes),
    } | {
        f"RankFuse + paging (resident to rank {r})": lora_bytes(d, math.ceil(r / 16) * 16, dtype_bytes)
        for r in sorted(set(tiers)) if r < r_max
    }


def fit_table(d: dict, r_max: int, tiers: list[int], budgets_gb: list[float], dtype_bytes: int = 2) -> str:
    per = per_adapter_bytes(d, r_max, tiers, dtype_bytes)
    lines = ["| storage | MiB / adapter | " + " | ".join(f"fit in {b:g} GiB" for b in budgets_gb) + " |",
             "|---|---|" + "---|" * len(budgets_gb)]
    for name, b in per.items():
        lines.append(f"| {name} | {b / 2**20:.0f} | " + " | ".join(str(int(g * GIB // b)) for g in budgets_gb) + " |")
    return "\n".join(lines)


_PATTERNS = {
    "model_load_gib": r"Model loading took ([0-9.]+) GiB",
    "kv_avail_gib": r"Available KV cache memory: ([0-9.]+) GiB",
    "kv_tokens": r"KV cache size: ([0-9,]+) tokens",
    "max_concurrency": r"Maximum concurrency for [0-9,]+ tokens per request: ([0-9.]+)x",
    "graph_capture_gib": r"Graph capturing finished in [0-9.]+ secs?, took ([0-9.\-]+) GiB",
}


def parse_vllm_log(path: str | Path) -> dict:
    text = Path(path).read_text(errors="replace")
    out = {}
    for k, pat in _PATTERNS.items():
        m = re.findall(pat, text)
        if m:
            v = m[-1].replace(",", "")
            out[k] = float(v) if "." in v or k.endswith("gib") else int(v)
    out["rankfuse_frontend"] = ("[serve_vllm] RankFuse front end installed" in text
                                or "RankFuse front end: per-request rank" in text)
    out["rankfuse_plugin"] = "RankFuse vLLM plugin active" in text
    out["rankfuse_rank_skip"] = (re.findall(r"RankFuse rank skipping under CUDA graphs: kernel (\w+)", text) or [None])[-1]
    out["rankfuse_modules"] = (re.findall(r"RankFuse: fused Matryoshka LoRA kernels on (\d+) LoRA modules", text) or [None])[-1]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--config", required=True)
    f.add_argument("--r-max", type=int, default=256)
    f.add_argument("--tiers", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    f.add_argument("--budget-gb", type=float, nargs="+", default=[2, 4, 8, 16])
    f.add_argument("--dtype-bytes", type=int, default=2)
    lg = sub.add_parser("log")
    lg.add_argument("path")
    a = p.parse_args(argv)
    if a.cmd == "fit":
        d = dims(json.loads(Path(a.config).read_text()))
        print(f"model dims {d}, r_max {a.r_max}, tiers {a.tiers}")
        print(fit_table(d, a.r_max, a.tiers, a.budget_gb, a.dtype_bytes))
    else:
        print(json.dumps(parse_vllm_log(a.path), indent=1))


if __name__ == "__main__":
    main()
