"""Re-score finished runs on the train hold-out with per-rank CE and accuracy (no test data).

    python val_metrics.py --model <path> --runs <run dir> [<run dir> ...]

Accuracy is computed on the labelled tokens of each hold-out example:
  boolq       choice accuracy: logit(" yes") vs logit(" no") at the answer position (what lm-eval scores)
  other tasks token accuracy: argmax over the vocabulary == label, averaged over answer tokens
Writes <run>/val_metrics.json = {"ce": {rank: ..}, "acc": {rank: ..}}. Used for LR selection on BoolQ,
where cross-entropy rewards calibration more than accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlora_train import collate, inject, p_vector, pow2_ranks, set_rank  # noqa: E402
from qtasks import load_task  # noqa: E402


@torch.no_grad()
def score(model, data, tok, task, micro=32):
    yes = tok(" yes", add_special_tokens=False)["input_ids"][0]
    no = tok(" no", add_special_tokens=False)["input_ids"][0]
    ce_sum, n_tok, correct, n = 0.0, 0, 0, 0
    order = sorted(range(len(data)), key=lambda i: len(data[i]["input_ids"]))
    for s in range(0, len(order), micro):
        ids, lab, att = collate([data[i] for i in order[s : s + micro]], tok.pad_token_id, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
        tgt = lab[:, 1:]
        m = tgt != -100
        logits = model.lm_head(hidden[m]).float()
        ce_sum += torch.nn.functional.cross_entropy(logits, tgt[m], reduction="sum").item()
        n_tok += int(m.sum())
        if task == "boolq":
            pred = torch.where(logits[:, yes] > logits[:, no], yes, no)
            correct += int((pred == tgt[m]).sum())
            n += int(m.sum())
        else:
            correct += int((logits.argmax(-1) == tgt[m]).sum())
            n += int(m.sum())
    return ce_sum / n_tok, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--p-serve", action="store_true",
                    help="paper-variant check: serve rank r as x (A*P)[:, :r] B[:r] (trained P kept) -> val_metrics_pserve.json")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    cache = {}
    for rd in map(Path, a.runs):
        outf = rd / ("val_metrics_pserve.json" if a.p_serve else "val_metrics.json")
        if outf.exists() or not (rd / "lora_final.pt").exists():
            continue
        meta = json.loads((rd / "meta.json").read_text())
        if a.p_serve and meta["method"] != "mlora":
            continue
        if meta["task"] not in cache:
            cache[meta["task"]] = load_task(meta["task"], tok, meta["max_len"], seed=42)[1]
        val = cache[meta["task"]]
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).cuda().eval()
        mods = inject(model, meta["rank"], None)
        st = torch.load(rd / "lora_final.pt", map_location="cpu")
        for k, m in mods.items():
            m.lora_A_t.data.copy_(st[k][0].float())
            m.lora_B.data.copy_(st[k][1].float())
            if a.p_serve:
                m.lora_A_t.data.mul_(p_vector(meta["rank"], pow2_ranks(meta["rank"])).to(m.lora_A_t.device)[:, None])
        out = {"ce": {}, "acc": {}}
        for r in [0] + pow2_ranks(meta["rank"]):  # rank 0 = empty prefix = base model, for reference
            set_rank(mods, r)
            out["ce"][r], out["acc"][r] = score(model, val, tok, meta["task"])
        outf.write_text(json.dumps(out, indent=1))
        print(rd.name, "pserve" if a.p_serve else "", "acc", {r: round(v, 3) for r, v in out["acc"].items()},
              "ce", {r: round(v, 3) for r, v in out["ce"].items()}, flush=True)
        del model


if __name__ == "__main__":
    main()
