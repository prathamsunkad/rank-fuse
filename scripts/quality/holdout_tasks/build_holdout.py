"""Write the train-split hold-out data used for LR selection (never a test split) as local JSON, so
lm-eval scores it with the real task's prompt, few-shot setup and metric.

  gsm8k_holdout      the 374 GSM8K train problems held out of training (qtasks.load_task split,
                     seed 42); 8-shot exemplars drawn from the remaining train problems
  boolq_holdout      the 471 BoolQ train examples held out of training; 0-shot
  arc_challenge_val  ARC-Challenge *validation* split (lm-eval's arc_challenge scores test)
  hellaswag_holdout  1000 HellaSwag *train* examples (seed 42); exemplars from the rest of train
                     (lm-eval's hellaswag scores validation)
    python build_holdout.py <out dir>
"""
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)


def split_idx(n, val_frac=0.05, val_max=1000, seed=42):  # identical to qtasks.load_task
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_val = min(val_max, int(round(val_frac * n)))
    return set(idx[:n_val])


def dump(name, rows):
    with open(out / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(name, len(rows))


g = load_dataset("openai/gsm8k", "main", split="train")
h = split_idx(len(g))
dump("gsm8k_holdout", [g[i] for i in range(len(g)) if i in h])
dump("gsm8k_fewshot", [g[i] for i in range(len(g)) if i not in h])

b = load_dataset("super_glue", "boolq", split="train")
h = split_idx(len(b))
dump("boolq_holdout", [{k: b[i][k] for k in ("passage", "question", "label")} for i in range(len(b)) if i in h])

hs = load_dataset("Rowan/hellaswag", split="train")
idx = list(range(len(hs)))
random.Random(42).shuffle(idx)
hold = set(idx[:1000])
keep = ("ctx_a", "ctx_b", "activity_label", "endings", "label")
dump("hellaswag_holdout", [{k: hs[i][k] for k in keep} for i in sorted(hold)])
dump("hellaswag_fewshot", [{k: hs[i][k] for k in keep} for i in idx[1000:3000]])
