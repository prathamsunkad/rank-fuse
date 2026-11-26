"""Step 3 of the serving-path check: RankFuse's vLLM 0.32 plugin (RANKFUSE_VLLM=1) at per-request rank r
against the PEFT export at rank r (shared env; PYTHONPATH=<this worktree> so this worktree's rankfuse
is the one imported).

Scores the token sequences saved by verify_peft_logits.py teacher-forced (prompt_logprobs) through ONE
engine holding the full r_max Matryoshka adapter, with every request at rank r and with a mixed-rank
batch, and compares each token's log-probability with PEFT's at the same rank.
    RANKFUSE_VLLM=1 PYTHONPATH=$WT python verify_vllm_plugin.py --model <path> --adapter <dir> --peft f.pt --out r.json
"""
import argparse
import json
import os
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--peft", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eager", action="store_true")
    a = ap.parse_args()
    assert os.environ.get("RANKFUSE_VLLM") == "1"
    # Importing rankfuse (Triton kernels) initialises CUDA; a forked engine core then hangs in its
    # first pinned-memory allocation. Spawn the engine and import rankfuse only after it is up.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    ref = torch.load(a.peft)
    ids = ref["input_ids"]
    B, L = ids.shape
    # PEFT log-prob of token t given tokens < t, positions 1..L-1
    peft_lp = {r: torch.log_softmax(lg, -1)[:, :-1].gather(-1, ids[:, 1:, None])[..., 0] for r, lg in ref["logits"].items()}
    meta = json.loads((Path(a.adapter) / "adapter_config.json").read_text())
    llm = LLM(model=a.model, enable_lora=True, max_lora_rank=meta["r"], max_loras=1, max_model_len=1024,
              gpu_memory_utilization=0.5, seed=0, dtype="bfloat16", enforce_eager=a.eager)
    import rankfuse
    from rankfuse.integration import MatryoshkaLoRARequest
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    prompts = [TokensPrompt(prompt_token_ids=ids[i].tolist()) for i in range(B)]

    def score(seq_ranks):
        reqs = [None if r == 0 else MatryoshkaLoRARequest("mlora", 1, a.adapter, matryoshka_rank=r) for r in seq_ranks]
        outs = llm.generate(prompts, sp, lora_request=reqs, use_tqdm=False)
        lp = torch.zeros(B, L - 1)
        for i, o in enumerate(outs):
            for t in range(1, L):
                lp[i, t - 1] = o.prompt_logprobs[t][ids[i, t].item()].logprob
        return lp

    rep = dict(vllm=vllm.__version__, rankfuse_file=rankfuse.__file__, eager=a.eager, adapter=a.adapter, results=[])
    ranks = [0] + ref["ranks"]
    for r in ranks:
        lp = score([r] * B)
        d = (lp - peft_lp[r]).abs()
        dist = {k: (lp - v).abs().mean().item() for k, v in peft_lp.items()}
        rep["results"].append(dict(case=f"all r={r}", max_abs_dlogprob=d.max().item(), mean_abs_dlogprob=d.mean().item(),
                                   mean_abs_vs_base=(peft_lp[r] - peft_lp[0]).abs().mean().item(),
                                   closest_peft_rank=min(dist, key=dist.get), dist_to_each_peft_rank=dist))
    mixed = [ranks[1 + i % (len(ranks) - 1)] if i % 4 != 3 else 0 for i in range(B)]
    lp = score(mixed)
    for i, r in enumerate(mixed):
        d = (lp[i] - peft_lp[r][i]).abs()
        dist = {k: (lp[i] - v[i]).abs().mean().item() for k, v in peft_lp.items()}
        rep["results"].append(dict(case=f"mixed {mixed}: seq {i} r={r}", max_abs_dlogprob=d.max().item(),
                                   mean_abs_dlogprob=d.mean().item(), closest_peft_rank=min(dist, key=dist.get)))
    for row in rep["results"]:
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
    Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
