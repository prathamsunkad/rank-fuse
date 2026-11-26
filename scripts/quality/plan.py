"""Bookkeeping for the quality-vs-rank study: training queues, LR selection, adapter export, eval manifests.

  python plan.py queue  --mk q05 --task gsm8k --lrs 2e-5,5e-5,1e-4 --configs mlora:256,lora:8,lora:32 [--extra "--micro 16"]
  python plan.py select --mk q05 --task gsm8k          # best LR per (method, rank) by validation loss
  python plan.py manifest --mk q05 --task gsm8k --ranks 1,2,4,8,16,32,64,128,256
  python plan.py clean  --mk q05 --task gsm8k          # delete lora_final.pt of non-selected runs

Run folders: $RQ/quality_runs/<mk>/<task>/<method>_r<rank>_lr<lr>_s<seed>/
Selected adapters are exported (bf16 PEFT, rank order kept) to $RQ/adapters/<model dir>/<task>/<method>-r<rank>/.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

RQ = Path(os.environ.get("RQ", "/storage/ice1/4/7/psunkad3/rankfuse"))
MODELS = {  # mk -> (env var with local path, HF id, adapters/ folder name)
    "q05": ("QWEN05", "Qwen/Qwen2.5-0.5B-Instruct", "qwen2.5-0.5b"),
    "q7": ("QWEN7", "Qwen/Qwen2.5-7B-Instruct", "qwen2.5-7b"),
}
EVAL_TASKS = {"gsm8k": {"gsm8k": 8}, "platypus": {"arc_challenge": 25, "hellaswag": 10}, "boolq": {"boolq": 0}}
SEL_RANKS = [8, 16, 32, 64, 128, 256]
HOLDOUT_TASKS = {"gsm8k": {"gsm8k_holdout": 8}, "boolq": {"boolq_holdout": 0},
                 "platypus": {"arc_challenge_val": 25, "hellaswag_holdout": 10}}
HOLDOUT_METRIC = {"gsm8k_holdout": "exact_match,strict-match", "boolq_holdout": "acc,none",
                  "arc_challenge_val": "acc,none", "hellaswag_holdout": "acc,none"}
NESTED = ("mlora", "mlora_sum", "mlora_k2", "dylora")
ACC_SELECT = {"boolq"}  # single-token classification: select on hold-out accuracy, not CE  # Matryoshka LR chosen by mean val loss over these ranks


def run_dir(mk, task, method, rank, lr, seed=42):
    return RQ / "quality_runs" / mk / task / f"{method}_r{rank}_lr{lr}_s{seed}"


def runs(mk, task):
    base = RQ / "quality_runs" / mk / task
    out = []
    for d in sorted(base.glob("*_r*_lr*_s*")):
        if not (d / "done.json").exists():
            continue
        meta = json.loads((d / "meta.json").read_text())
        done = json.loads((d / "done.json").read_text())
        out.append(dict(dir=d, task=meta["task"], method=meta["method"], rank=meta["rank"], lr=meta["lr"], seed=meta["seed"],
                        val={int(k): v for k, v in done["val_loss"].items()}, meta=meta, done=done))
    return out


def score(r):
    """Secondary column only: hold-out validation CE (lower is better); nested adapters: mean over SEL_RANKS."""
    val = r["val"]
    if r["method"] in ("mlora", "mlora_sum", "mlora_k2", "dylora"):
        return statistics.mean(val[k] for k in SEL_RANKS if k in val)
    return val[r["rank"]]


def cmd_queue(a):
    env, _, _ = MODELS[a.mk]
    lines = []
    for cfg in a.configs.split(","):
        method, rank = cfg.split(":")
        for lr in a.lrs.split(","):
            out = run_dir(a.mk, a.task, method, rank, lr, a.seed)
            lines.append(f"--model ${env} --task {a.task} --method {method} --rank {rank} --lr {lr} --seed {a.seed} "
                         f"--out {out} {a.extra}".strip())
    q = Path(a.file)
    q.parent.mkdir(parents=True, exist_ok=True)
    q.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} runs -> {q}")


def holdout_acc(mk, task, run_name, order, r):
    """Hold-out task accuracy (mean over the task's hold-out evals) of run `run_name` served at rank r."""
    vals = []
    for t in HOLDOUT_TASKS[task]:
        f = RQ / "quality_runs" / mk / task / "holdout" / f"{run_name}@{order}{r}__{t}.json"
        if not f.exists():
            return None
        vals.append(json.loads(f.read_text())["metrics"][HOLDOUT_METRIC[t]])
    return statistics.mean(vals)


def cmd_holdout_manifest(a):
    _, hf_id, _ = MODELS[a.mk]
    specs = []
    for r in runs(a.mk, a.task):
        if not (r["dir"] / "lora_final.pt").exists():
            continue
        name = r["dir"].name
        if r["method"] in NESTED:
            todo = [("prefix", k) for k in SEL_RANKS if k <= r["rank"]]
        elif r["rank"] == 256:  # ordinary r256 LoRA: full rank + both truncations
            todo = [("prefix", k) for k in SEL_RANKS] + [("svd", k) for k in SEL_RANKS if k < 256]
        else:
            todo = [("prefix", r["rank"])]
        for order, k in todo:
            specs.append(dict(name=f"{name}@{order}{k}", lora_final=str(r["dir"] / "lora_final.pt"), rank=k,
                              order=order, base_model=hf_id, tasks=HOLDOUT_TASKS[a.task]))
    specs.append(dict(name="base@prefix0", base=True, tasks=HOLDOUT_TASKS[a.task]))
    p = RQ / "quality_runs" / a.mk / a.task / "holdout_manifest.json"
    p.write_text(json.dumps(specs, indent=1))
    print(f"{len(specs)} hold-out specs -> {p}")


def cmd_select(a):
    from export_peft import export

    rs = [r for r in runs(a.mk, a.task) if r["seed"] == a.seed]
    # Selection rule (all methods): highest task accuracy on the train-split hold-out (holdout_tasks/),
    # never a test split. Nested adapters: mean over SEL_RANKS; separate LoRAs: at their own rank;
    # truncation baselines (key lora_r256_trunc / _svd): the r256 LoRA's mean truncated accuracy.
    def acc(r, order="prefix", ks=None):
        ks = ks or ([k for k in SEL_RANKS if k <= r["rank"]] if r["method"] in NESTED else [r["rank"]])
        v = [holdout_acc(a.mk, a.task, r["dir"].name, order if k < 256 else "prefix", k) for k in ks]
        return None if any(x is None for x in v) else statistics.mean(v)

    cands = {}
    for r in rs:
        cands.setdefault((r["method"], r["rank"], "main"), []).append((acc(r), r))
        if r["method"] == "lora" and r["rank"] == 256:
            cands.setdefault(("lora", 256, "trunc"), []).append((acc(r, "prefix", SEL_RANKS), r))
            cands.setdefault(("lora", 256, "svd"), []).append((acc(r, "svd", SEL_RANKS), r))
    best = {}
    for key, lst in cands.items():
        lst = [(x, r) for x, r in lst if x is not None]
        if lst:
            best[key] = max(lst, key=lambda t: t[0])
        missing = [r["dir"].name for x, r in cands[key] if x is None]
        if missing:
            print(f"WARNING {key}: no hold-out accuracy yet for {missing}")
    _, hf_id, folder = MODELS[a.mk]
    sel = {}
    for (method, rank, kind), (hacc, r) in sorted(best.items()):
        grid = sorted((x["lr"], None if v is None else round(v, 4), round(score(x), 4))
                      for v, x in cands[(method, rank, kind)])
        key = f"{method}_r{rank}" + ("" if kind == "main" else f"_{kind}")
        print(f"{key:18s} best lr {r['lr']:g}  holdout acc {hacc:.4f}  grid (lr, holdout acc, val CE) {grid}")
        dest = RQ / "adapters" / folder / a.task / (f"{method}-r{rank}" + (f"-s{r['seed']}" if r["seed"] != 42 else ""))
        dest = dest.parent / (dest.name + f"-lr{r['lr']:g}")
        sel[key] = dict(dir=str(r["dir"]), adapter=str(dest), lr=r["lr"], holdout_acc=hacc, val_ce=score(r), grid=grid,
                        seed=r["seed"], train_secs=r["done"]["train_secs"], gpu=r["meta"]["gpu"])
        if not a.no_export and not (dest / "adapter_model.safetensors").exists():
            extra = dict(method=method, r_max=rank, tiers=[2**i for i in range(rank.bit_length()) if 2**i <= rank],
                         scale=1.0, layout="rank r = lora_A[:r], lora_B[:, :r]; lora_alpha = r (scale 1.0)",
                         source_run=str(r["dir"]), lr=r["lr"], seed=r["seed"], base_model=hf_id,
                         nested=method == "mlora")
            export(r["dir"] / "lora_final.pt", dest, hf_id, rank, "prefix", extra)
    p = RQ / "quality_runs" / a.mk / a.task / ("selected.json" if a.seed == 42 else f"selected_s{a.seed}.json")
    p.write_text(json.dumps(sel, indent=1))
    print("->", p)


def cmd_manifest(a):
    sel = json.loads((RQ / "quality_runs" / a.mk / a.task / "selected.json").read_text())
    _, hf_id, _ = MODELS[a.mk]
    tasks = EVAL_TASKS[a.task]
    ranks = [int(x) for x in a.ranks.split(",")]
    specs = [dict(name="base", base=True, tasks=tasks)]

    def add(name, key, r, order):
        specs.append(dict(name=name, lora_final=sel[key]["adapter"], rank=r, order=order,
                          base_model=hf_id, tasks=tasks, method_key=key))

    for mkey in ("mlora_r256", "mlora_sum_r256", "mlora_k2_r256", "dylora_r256"):
        if mkey in sel:
            for r in ranks:
                add(f"{mkey}@{r}", mkey, r, "prefix")
    for key in sel:
        if key.startswith("lora_") and key.count("_") == 1:
            k = int(key.split("_r")[1])
            add(f"lora_r{k}", key, k, "prefix")
    if not a.no_trunc:
        for r in ranks:
            if r < 256:
                if "lora_r256_trunc" in sel:
                    add(f"lora_r256_trunc@{r}", "lora_r256_trunc", r, "prefix")
                if "lora_r256_svd" in sel:
                    add(f"lora_r256_svd@{r}", "lora_r256_svd", r, "svd")
        for kind in ("trunc", "svd"):  # their untruncated r=256 point (may be a different LR than lora_r256)
            if f"lora_r256_{kind}" in sel and sel[f"lora_r256_{kind}"]["dir"] != sel.get("lora_r256", {}).get("dir"):
                add(f"lora_r256_{kind}@256", f"lora_r256_{kind}", 256, "prefix")
    p = Path(a.out or RQ / "quality_runs" / a.mk / a.task / "manifest.json")
    p.write_text(json.dumps(specs, indent=1))
    print(f"{len(specs)} specs -> {p}")


def cmd_clean(a):
    sel = json.loads((RQ / "quality_runs" / a.mk / a.task / "selected.json").read_text())
    exported = {v["dir"] for v in sel.values() if (Path(v["adapter"]) / "adapter_model.safetensors").exists()}
    freed = 0
    for r in runs(a.mk, a.task):
        f = r["dir"] / "lora_final.pt"
        if f.exists() and (str(r["dir"]) in exported or str(r["dir"]) not in {v["dir"] for v in sel.values()}):
            freed += f.stat().st_size
            if a.yes:
                f.unlink()
            print(("deleted " if a.yes else "would delete ") + str(f))
    print(f"{freed / 1e9:.2f} GB")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("queue")
    q.add_argument("--mk", required=True)
    q.add_argument("--task", required=True)
    q.add_argument("--lrs", required=True)
    q.add_argument("--configs", required=True)
    q.add_argument("--seed", type=int, default=42)
    q.add_argument("--extra", default="")
    q.add_argument("--file", required=True)
    s = sp.add_parser("select")
    s.add_argument("--mk", required=True)
    s.add_argument("--task", required=True)
    s.add_argument("--no-export", action="store_true")
    s.add_argument("--seed", type=int, default=42)
    h = sp.add_parser("holdout-manifest")
    h.add_argument("--mk", required=True)
    h.add_argument("--task", required=True)
    m = sp.add_parser("manifest")
    m.add_argument("--mk", required=True)
    m.add_argument("--task", required=True)
    m.add_argument("--ranks", default="8,16,32,64,128,256")
    m.add_argument("--no-trunc", action="store_true")
    m.add_argument("--out", default=None)
    c = sp.add_parser("clean")
    c.add_argument("--mk", required=True)
    c.add_argument("--task", required=True)
    c.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    dict(queue=cmd_queue, select=cmd_select, manifest=cmd_manifest, clean=cmd_clean,
         **{"holdout-manifest": cmd_holdout_manifest})[a.cmd](a)


if __name__ == "__main__":
    main()
