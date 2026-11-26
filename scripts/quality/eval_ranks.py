"""Evaluate base model + many rank-r adapters with lm-evaluation-harness on one vLLM engine.

    python eval_ranks.py --model <hf path> --manifest specs.json --out <results dir>

specs.json is a list of {"name", "tasks": {task: num_fewshot}, and either "base": true or
"lora_final": <path>, "rank": r, "order": "prefix"|"svd", "base_model": hf id}. Each adapter is
exported to PEFT (rank-r prefix, lora_alpha = r) in a temporary folder, evaluated through vLLM's
LoRA path, then deleted (scratch quota). Results go to <out>/<name>__<task>.json; existing results
are skipped, so the job can be resubmitted after a timeout.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_peft import export  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--max-lora-rank", type=int, default=256)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None, help="debug: examples per task")
    ap.add_argument("--include-path", default=None, help="extra lm-eval task YAMLs (holdout_tasks/)")
    a = ap.parse_args()

    import lm_eval
    from lm_eval.models.vllm_causallms import VLLM
    from vllm.lora.request import LoRARequest

    specs = json.loads(Path(a.manifest).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = [s for s in specs if any(not (out / f"{s['name']}__{t}.json").exists() for t in s["tasks"])]
    print(f"{len(todo)}/{len(specs)} specs to run")
    if not todo:
        return
    t0 = time.time()
    from lm_eval.tasks import TaskManager

    tm = TaskManager(include_path=a.include_path) if a.include_path else TaskManager()
    lm = VLLM(pretrained=a.model, dtype="bfloat16", batch_size="auto", seed=1234, gpu_memory_utilization=a.gpu_mem,
              max_model_len=a.max_model_len, enable_lora=True, max_lora_rank=a.max_lora_rank, max_loras=1,
              max_cpu_loras=1)
    print(f"engine up in {time.time() - t0:.0f}s", flush=True)
    tmp_root = Path(a.tmp)
    for k, s in enumerate(todo, start=1):
        tdir = None
        if s.get("base"):
            lm.lora_request = None
        else:
            tdir = tmp_root / f"{s['name']}"
            if tdir.exists():
                shutil.rmtree(tdir)
            export(Path(s["lora_final"]), tdir, s["base_model"], s["rank"], s.get("order", "prefix"))
            lm.lora_request = LoRARequest(s["name"], 1000 + k, str(tdir))
        for task, shots in s["tasks"].items():
            f = out / f"{s['name']}__{task}.json"
            if f.exists():
                continue
            t1 = time.time()
            res = lm_eval.simple_evaluate(model=lm, tasks=[task], num_fewshot=shots, limit=a.limit, task_manager=tm,
                                          log_samples=False, random_seed=0, numpy_random_seed=1234,
                                          torch_random_seed=1234, fewshot_random_seed=1234)
            rec = dict(spec=s, task=task, num_fewshot=shots, metrics=res["results"][task],
                       n_samples=res.get("n-samples", {}).get(task), eval_secs=time.time() - t1,
                       lm_eval_version=lm_eval.__version__ if hasattr(lm_eval, "__version__") else None)
            f.write_text(json.dumps(rec, indent=1, default=str))
            print(f"[{k}/{len(todo)}] {s['name']} {task}: "
                  + ", ".join(f"{m}={v:.4f}" for m, v in res["results"][task].items() if isinstance(v, float))
                  + f" ({time.time() - t1:.0f}s)", flush=True)
        if tdir is not None:
            shutil.rmtree(tdir, ignore_errors=True)


if __name__ == "__main__":
    main()
