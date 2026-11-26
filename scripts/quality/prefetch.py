"""Download (login node) the training sets and the lm-eval task data into $HF_HOME, so GPU jobs can
run with HF_HUB_OFFLINE=1 / HF_DATASETS_OFFLINE=1.   python prefetch.py"""

import sys

from datasets import load_dataset

for path, name in [("openai/gsm8k", "main"), ("garage-bAInd/Open-Platypus", None), ("super_glue", "boolq")]:
    ds = load_dataset(path, name) if name else load_dataset(path)
    print(path, name, {k: len(v) for k, v in ds.items()})

from lm_eval.tasks import TaskManager, get_task_dict  # noqa: E402

tm = TaskManager()
td = get_task_dict(["gsm8k", "arc_challenge", "hellaswag", "boolq"], tm)
for k, t in td.items():
    print(k, type(t).__name__, getattr(t, "DATASET_PATH", None), getattr(t, "DATASET_NAME", None))

# Models are not downloaded here: 0.5B and 7B Instruct already exist in $R/hf (read-only).
