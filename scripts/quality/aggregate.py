"""Collect eval JSONs into CSVs and draw accuracy-vs-rank plots.

    python aggregate.py --results results/quality_2026-11-24

Reads <results>/<mk>/<train task>/raw/<name>__<eval task>.json (written by eval_ranks.py) and writes
<results>/quality_vs_rank.csv (long form, one row per metric), <results>/summary_<mk>.csv (wide:
primary metric, one row per method, one column per rank) and <results>/plots/<mk>_<eval task>.png.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

PRIMARY = {  # eval task -> (metric key, label)
    "gsm8k": ("exact_match,strict-match", "GSM8K 8-shot exact match (strict)"),
    "arc_challenge": ("acc,none", "ARC-Challenge 25-shot acc"),
    "hellaswag": ("acc,none", "HellaSwag 10-shot acc"),
    "boolq": ("acc,none", "BoolQ 0-shot acc"),
}
MODEL_NAMES = {"q05": "Qwen2.5-0.5B-Instruct", "q7": "Qwen2.5-7B-Instruct"}
SERIES = [  # (method, label, color, marker, linestyle) -- fixed order, validated palette
    ("mlora_sum", "Matryoshka, exact nested loss (one r256 adapter)", "#2a78d6", "o", "-"),
    ("mlora", "MatryoshkaLoRA, paper single-pass (one r256 adapter)", "#6250d6", "^", "-"),
    ("lora", "LoRA trained at rank r (separate adapters)", "#eb6834", "s", "none"),
    ("mlora_k2", "Matryoshka, sampled nested loss (r256 + 1 rank/step)", "#e87ba4", "P", "-"),
    ("dylora", "DyLoRA (1 random rank/step)", "#008300", "X", "-."),
    ("trunc", "LoRA r256, first r rows", "#1baf7a", "v", "--"),
    ("svd", "LoRA r256, SVD top-r", "#eda100", "D", ":"),
]


def parse_name(name: str):
    if name == "base":
        return "base", None, 0
    for pre in ("mlora_k2", "dylora"):
        if name.startswith(pre + "_r256@"):
            return pre, 256, int(name.split("@")[1])
    if name.startswith("mlora_sum_r256@"):
        return "mlora_sum", 256, int(name.split("@")[1])
    if name.startswith("mlora_r256@"):
        return "mlora", 256, int(name.split("@")[1])
    if name.startswith("lora_r256_trunc@"):
        return "trunc", 256, int(name.split("@")[1])
    if name.startswith("lora_r256_svd@"):
        return "svd", 256, int(name.split("@")[1])
    if name.startswith("lora_r"):
        k = int(name[len("lora_r"):])
        return "lora", k, k
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    root = Path(a.results)
    rows = []
    for f in sorted(root.glob("*/*/raw/*.json")):
        mk, train_task = f.parts[-4], f.parts[-3]
        rec = json.loads(f.read_text())
        method, rmax, r = parse_name(rec["spec"]["name"])
        if method == "lora" and r == 256:
            pass
        for key, v in rec["metrics"].items():
            if not isinstance(v, (int, float)) or "stderr" in key:
                continue
            se = rec["metrics"].get(key.replace(",", "_stderr,", 1))
            rows.append(dict(model=MODEL_NAMES.get(mk, mk), mk=mk, train_task=train_task, eval_task=rec["task"],
                             shots=rec["num_fewshot"], metric=key, method=method, r_max=rmax, rank=r,
                             value=v, stderr=se if isinstance(se, (int, float)) else "", eval_secs=round(rec["eval_secs"], 1),
                             n=rec.get("n_samples", {}).get("effective") if isinstance(rec.get("n_samples"), dict) else ""))
    if not rows:
        print("no results yet")
        return
    cols = list(rows[0].keys())
    with open(root / "quality_vs_rank.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {root / 'quality_vs_rank.csv'}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (root / "plots").mkdir(exist_ok=True)
    for mk in sorted({r["mk"] for r in rows}):
        ranks_all = sorted({r["rank"] for r in rows if r["mk"] == mk and r["rank"]})
        with open(root / f"summary_{mk}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["eval_task", "metric", "method"] + [f"r{k}" for k in ranks_all] + ["base"])
            for et, (metric, label) in PRIMARY.items():
                sub = [r for r in rows if r["mk"] == mk and r["eval_task"] == et and r["metric"] == metric]
                if not sub:
                    continue
                base = next((r["value"] for r in sub if r["method"] == "base"), None)
                tab = {}
                for m, *_ in SERIES:
                    pts = {r["rank"]: r["value"] for r in sub if r["method"] == m}
                    if m in ("trunc", "svd"):  # the untruncated r256 point is the r256 LoRA itself
                        full = next((r["value"] for r in sub if r["method"] == "lora" and r["rank"] == 256), None)
                        if full is not None and pts:
                            pts[256] = full
                    if pts:
                        tab[m] = pts
                        w.writerow([et, metric, m] + [f"{100 * pts[k]:.2f}" if k in pts else "" for k in ranks_all]
                                   + [f"{100 * base:.2f}" if base is not None else ""])
                fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
                for m, lab, col, mar, ls in SERIES:
                    if m not in tab:
                        continue
                    ks = sorted(tab[m])
                    ax.plot(ks, [100 * tab[m][k] for k in ks], color=col, marker=mar, ms=6, lw=2,
                            linestyle=ls if ls != "none" else "none", label=lab)
                if base is not None:
                    ax.axhline(100 * base, color="#52514e", lw=1.2, ls=(0, (4, 3)), label="Base model (no adapter)")
                ax.set_xscale("log", base=2)
                ax.set_xticks(ranks_all)
                ax.set_xticklabels([str(k) for k in ranks_all])
                ax.set_xlabel("Inference rank r")
                ax.set_ylabel("Accuracy (%)")
                ax.set_title(f"{MODEL_NAMES.get(mk, mk)}: {label}", fontsize=10)
                ax.grid(True, color="#e6e5e0", lw=0.8)
                for s in ("top", "right"):
                    ax.spines[s].set_visible(False)
                ax.legend(fontsize=7, frameon=False, loc="lower right")
                fig.tight_layout()
                out = root / "plots" / f"{mk}_{et}.png"
                fig.savefig(out)
                plt.close(fig)
                print("plot ->", out)


if __name__ == "__main__":
    main()
