"""Training data for the quality-vs-rank study (MatryoshkaLoRA-style supervised fine-tuning).

Each loader returns (train, val): lists of {"input_ids": [...], "labels": [...]} where prompt tokens
carry label -100. `val` is a seeded hold-out taken from the *train* split and is used only for
learning-rate selection (validation loss), so the benchmark test/validation splits stay unseen.

Prompt formats
  gsm8k     "Question: {q}\nAnswer:" + " {answer}<eos>"  -- lm-eval's gsm8k doc_to_text/target.
            (MatryoshkaLoRA's data/gsm8k.py uses "Answer: " + "{answer}"; we move the space into
            the target so training tokenisation matches what lm-eval feeds the model.)
  platypus  Alpaca-style "### Instruction ... ### Response:\n" + output<eos>, exactly as
            MatryoshkaLoRA's data/platypus.py. Evaluated on ARC-Challenge and HellaSwag (paper 3.3).
  boolq     lm-eval's boolq format "{passage}\nQuestion: {question}?\nAnswer:" + " yes"/" no".
"""

from __future__ import annotations

import random

from datasets import load_dataset

DATASETS = {
    "gsm8k": ("openai/gsm8k", "main", "train"),
    "platypus": ("garage-bAInd/Open-Platypus", None, "train"),
    "boolq": ("super_glue", "boolq", "train"),
}


def _encode(tok, prompt: str, answer: str, max_len: int, truncate_prompt_left: bool = False):
    p = tok(prompt, add_special_tokens=True)["input_ids"]
    a = tok(answer, add_special_tokens=False)["input_ids"]
    if truncate_prompt_left and len(p) + len(a) > max_len:
        keep = max_len - len(a)
        if keep < 16:
            return None
        p = p[-keep:]
    ids = (p + a)[:max_len]
    labels = ([-100] * len(p) + a)[:max_len]
    if all(l == -100 for l in labels):
        return None
    return {"input_ids": ids, "labels": labels}


def _format(task: str, ex: dict, eos: str):
    if task == "gsm8k":
        return f"Question: {ex['question']}\nAnswer:", f" {ex['answer']}{eos}", False
    if task == "platypus":
        if ex["input"]:
            prompt = f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n"
        return prompt, ex["output"] + eos, False
    if task == "boolq":
        return f"{ex['passage']}\nQuestion: {ex['question']}?\nAnswer:", " yes" if ex["label"] == 1 else " no", True
    raise ValueError(task)


def load_task(task: str, tok, max_len: int = 512, val_frac: float = 0.05, val_max: int = 1000, seed: int = 42):
    path, name, split = DATASETS[task]
    ds = load_dataset(path, name, split=split) if name else load_dataset(path, split=split)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    n_val = min(val_max, int(round(val_frac * len(ds))))
    val_idx, train_idx = set(idx[:n_val]), idx[n_val:]
    eos = tok.eos_token
    train, val = [], []
    for i in range(len(ds)):
        prompt, answer, left = _format(task, ds[i], eos)
        enc = _encode(tok, prompt, answer, max_len, left)
        if enc is None:
            continue
        (val if i in val_idx else train).append(enc)
    return train, val
