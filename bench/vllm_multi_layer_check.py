"""Layer-level check of multi-adapter RankFuse inside a running vLLM engine against B0.

Runs one eager engine in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0, RANKFUSE_KEEP_STEP=1),
serves the triples of bench/vllm_multi_check.py in waves, and for the first CALLS_PER_STEP LoRA
layer calls of every step recomputes the kernel's delta into a separate buffer (same metadata, same
weights view the engine uses) and compares it with an fp32 B0 reference built from the adapters'
safetensors files on disk (not from vLLM's slots or the page pool, so packing, page-table and
slot-mapping bugs all show up):

    ref[t, slice s] = x[t] @ A_s[:r_t].T @ B_s[:, :r_t].T     (A/B: the PEFT checkpoint of t's adapter)

Rows of rank-0 tokens must be exactly zero. Prints max |err| / max |ref| per (module type, rank
class) and fails above TOL.

    RANKFUSE_VLLM=1 RANKFUSE_MULTI=1 [RANKFUSE_PAGING=1 RANKFUSE_PAGE_FRAMES=..] \
        python bench/vllm_multi_layer_check.py out.json
"""

import json
import os
import random
import re
import sys

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["RANKFUSE_KEEP_STEP"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from vllm_multi_check import (  # noqa: E402
    ANSWER, R_MAX, TOPICS, adapter_dir, engine_kwargs, prompts, triples,
)

TOL = 2e-2
CALLS_PER_STEP = int(os.environ.get("CALLS_PER_STEP", "8"))
SLICES = {"qkv_proj": ("q_proj", "k_proj", "v_proj"), "gate_up_proj": ("gate_proj", "up_proj"),
          "o_proj": ("o_proj",), "down_proj": ("down_proj",)}


def main(out_path):
    from safetensors.torch import load_file
    from vllm import LLM, SamplingParams

    from rankfuse.integration import MatryoshkaLoRARequest, multi_ops
    from rankfuse.kernels.rankfuse_multi import multi_lora

    paged = os.environ.get("RANKFUSE_PAGING", "0") == "1"
    llm = LLM(**engine_kwargs("paged" if paged else "dense", "eager"))
    ctx = multi_ops.CONTEXT
    am, wrapper = ctx["manager"], ctx["wrapper"]
    ckpt: dict[int, dict] = {}

    def weights_of(lid):  # lora_int_id -> PEFT tensors of the full adapter
        if lid not in ckpt:
            ckpt[lid] = load_file(os.path.join(adapter_dir(lid - 1), "adapter_model.safetensors"))
        return ckpt[lid]

    def row_to_id():
        if paged:
            return {slot: a for a, slot in am._rf_pager._ids.items()}
        return {i: lid for i, lid in enumerate(am.lora_index_to_id) if lid is not None}

    stats = {}
    seen = {"step": -1, "calls": 0, "checked": 0}

    def hook(stage, y, x, wid, token_ranks, meta):
        if stage != "pre":
            return
        if wrapper.steps != seen["step"]:
            seen["step"], seen["calls"] = wrapper.steps, 0
        seen["calls"] += 1
        if seen["calls"] > CALLS_PER_STEP:
            return
        name = multi_ops.NAMES[wid]
        mo = re.match(r".*layers\.(\d+)\.(self_attn|mlp)\.(\w+)$", name)
        layer, parent, kind = int(mo.group(1)), mo.group(2), mo.group(3)
        W = multi_ops.weights(wid)
        d = torch.zeros_like(y)
        multi_lora(x, W, token_ranks, meta, out=d, add=False)
        st = wrapper.last_step
        rows, ranks = st["rows"], st["ranks"]
        ids = row_to_id()
        ref = torch.zeros(y.shape, dtype=torch.float32, device=y.device)
        xf = x.float()
        Tn = x.shape[0]
        pairs = {}
        for t in range(Tn):
            if int(ranks[t]) > 0:
                pairs.setdefault((int(rows[t]), int(ranks[t])), []).append(t)
        for (row, r), toks in pairs.items():
            ck = weights_of(ids[row])
            idx = torch.tensor(toks, device=y.device)
            c = 0
            for sl in SLICES[kind]:
                pre = f"base_model.model.model.layers.{layer}.{parent}.{sl}"
                A = ck[f"{pre}.lora_A.weight"].to(y.device).float()  # [r_max, in]
                B = ck[f"{pre}.lora_B.weight"].to(y.device).float()  # [out, r_max]
                n = B.shape[0]
                ref[idx, c:c + n] = (xf[idx] @ A[:r].T) @ B[:, :r].T
                c += n
        df = d.float()
        zero_rows = torch.from_numpy(ranks[: x.shape[0]] == 0).to(y.device)
        nonzero_in_zero_rows = float(df[zero_rows].abs().max()) if zero_rows.any() else 0.0
        for cls in ("low", "full"):
            sel = [t for t in range(x.shape[0]) if ranks[t] > 0 and
                   ((ranks[t] >= R_MAX[ids[int(rows[t])] - 1]) == (cls == "full"))]
            if not sel:
                continue
            idx = torch.tensor(sel, device=y.device)
            err = float((df[idx] - ref[idx]).abs().max())
            scale = float(ref[idx].abs().max()) or 1.0
            key = f"{kind}/{cls}"
            s = stats.setdefault(key, {"max_rel": 0.0, "calls": 0, "tokens": 0, "zero_rows_max": 0.0})
            s["max_rel"] = max(s["max_rel"], err / scale)
            s["calls"] += 1
            s["tokens"] += len(sel)
            s["zero_rows_max"] = max(s["zero_rows_max"], nonzero_in_zero_rows)
        seen["checked"] += 1

    multi_ops.HOOK = hook
    T = triples()
    ps = prompts()
    texts = [p + "\n" + ANSWER.format(t=t) for p, t in zip(ps, TOPICS)]
    order = list(range(len(T)))
    random.Random(3).shuffle(order)
    sp = SamplingParams(temperature=0.0, max_tokens=4)

    def req(a, r):
        return None if r == 0 else MatryoshkaLoRARequest(f"m{a}", 1 + a, adapter_dir(a), matryoshka_rank=r)

    for i in range(0, len(order), 8):
        wave = order[i:i + 8]
        llm.generate([texts[T[k][0]] for k in wave], sp, lora_request=[req(T[k][1], T[k][2]) for k in wave],
                     use_tqdm=False)
    multi_ops.HOOK = None
    ok = all(s["max_rel"] <= TOL and s["zero_rows_max"] == 0.0 for s in stats.values()) and seen["checked"] > 0
    pager = getattr(am, "_rf_pager", None)
    res = {"ok": ok, "checked_calls": seen["checked"], "stats": stats, "paged": paged,
           "pager": None if pager is None else dict(frames=pager.usable_frames, loads=pager.stats.page_loads,
                                                    evictions=pager.stats.evictions)}
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    print("LAYER CHECK", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main(sys.argv[1])
