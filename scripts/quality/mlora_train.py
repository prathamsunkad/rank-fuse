"""Small resumable trainer for MatryoshkaLoRA and ordinary LoRA (quality-vs-rank study).

Method (MatryoshkaLoRA, arXiv 2605.07850, Eq. 3/7 and repo IST-DASLab/MatryoshkaLoRA@66dd464,
adapters/matryoshka.py mask_type="diag"): one forward pass per step with
    y = x W0^T + ((x A_t^T) * P) B,        P_i = sum_{r in S, r >= i+1} s_r
i.e. the sum over the nested rank set S of s_r * A_r B_r. Inference at rank r drops P and uses
s_r * x A_t[:r]^T B[:r]. We use s_r = 1 (the paper's main setting), S = {1, 2, 4, ..., r_max}.
Ordinary LoRA ("lora") is the same layer with P = 1 (= the repo's LoraLayer with scaling "none").
"mlora_sum" is the exact nested objective that the paper's single pass approximates (its Appendix B):
    L = mean_{r in R_t} CE(W0 + A_r B_r),  R_t = S (--ranks-per-step 0) or r_max plus K-1 ranks
drawn uniformly without replacement from S \ {r_max} each step; one forward/backward per rank,
no P, scale 1, i.e. every prefix is trained exactly as it is served.
"mlora_k2" is its sampled estimator: r_max plus one rank drawn uniformly from S \ {r_max} per step.
"dylora" draws a single rank uniformly from S each step (the paper's DyLoRA baseline, Eq. 2).
Initialisation follows the repo: B = 0, A ~ kaiming_uniform(a=sqrt(5)) on the repo's [in, r]
tensor, whose fan_in is r, i.e. A ~ U(-1/sqrt(r), 1/sqrt(r)).

Optimisation mirrors the repo's HF Trainer defaults: AdamW (betas 0.9/0.999, eps 1e-8, wd 0),
linear decay to 0, no warmup, grad-norm clip 1.0, bf16 autocast; loss is the token mean over the
effective batch. Differences (documented in the README): LoRA master weights are fp32 (the repo
keeps them bf16), a 5% train hold-out is used for LR selection, and no wandb.

Checkpoints (LoRA weights + optimiser + step) are written every --ckpt-every steps to
<out>/ckpt.pt and the run resumes from there automatically.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qtasks import load_task  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def pow2_ranks(r_max: int) -> list[int]:
    return [2**i for i in range(int(math.log2(r_max)) + 1)]


def p_vector(r_max: int, train_ranks: list[int]) -> torch.Tensor:
    """Algorithm 1 of the paper with s_r = 1: P[i] = #{r in S : r >= i + 1}."""
    return torch.tensor([sum(1 for r in train_ranks if r >= i + 1) for i in range(r_max)], dtype=torch.float32)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, P: torch.Tensor | None):
        super().__init__()
        self.base = base
        self.r = r
        dev = base.weight.device
        bound = 1.0 / math.sqrt(r)
        self.lora_A_t = nn.Parameter(torch.empty(r, base.in_features, device=dev).uniform_(-bound, bound))
        self.lora_B = nn.Parameter(torch.zeros(r, base.out_features, device=dev))
        self.register_buffer("P", None if P is None else P.to(dev), persistent=False)
        self.inf_rank: int | None = None  # None -> training form (with P)

    def forward(self, x):
        y = self.base(x)
        r = self.inf_rank
        if r is None:
            h = F.linear(x, self.lora_A_t.to(x.dtype))
            if self.P is not None:
                h = h * self.P.to(h.dtype)
            return y + F.linear(h, self.lora_B.t().to(x.dtype))
        h = F.linear(x, self.lora_A_t[:r].to(x.dtype))
        return y + F.linear(h, self.lora_B[:r].t().to(x.dtype))


def inject(model, r: int, P):
    for p in model.parameters():
        p.requires_grad_(False)
    mods = {}
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if child_name in TARGETS and isinstance(child, nn.Linear):
                full = f"{name}.{child_name}" if name else child_name
                new = LoRALinear(child, r, P)
                setattr(mod, child_name, new)
                mods[full] = new
    return mods


def set_rank(mods, r):
    for m in mods.values():
        m.inf_rank = r


def lora_state(mods):
    return {k: (m.lora_A_t.detach().cpu(), m.lora_B.detach().cpu()) for k, m in mods.items()}


def load_lora_state(mods, st):
    for k, m in mods.items():
        a, b = st[k]
        m.lora_A_t.data.copy_(a)
        m.lora_B.data.copy_(b)


def collate(items, pad_id, device):
    L = max(len(it["input_ids"]) for it in items)
    ids = torch.full((len(items), L), pad_id, dtype=torch.long)
    lab = torch.full((len(items), L), -100, dtype=torch.long)
    att = torch.zeros((len(items), L), dtype=torch.long)
    for i, it in enumerate(items):
        n = len(it["input_ids"])
        ids[i, :n] = torch.tensor(it["input_ids"])
        lab[i, :n] = torch.tensor(it["labels"])
        att[i, :n] = 1
    return ids.to(device), lab.to(device), att.to(device)


def sum_loss(model, ids, lab, att):
    """Summed next-token CE over labelled positions. lm_head runs only on those positions (same
    value as full logits + ignore_index, far less memory: BoolQ has one label token per example)."""
    hidden = model.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
    tgt = lab[:, 1:]
    m = tgt != -100
    logits = model.lm_head(hidden[m]).float()
    return F.cross_entropy(logits, tgt[m], reduction="sum")


@torch.no_grad()
def val_loss(model, data, pad_id, device, micro):
    model.eval()
    tot, n = 0.0, 0
    order = sorted(range(len(data)), key=lambda i: len(data[i]["input_ids"]))
    for s in range(0, len(order), micro):
        items = [data[i] for i in order[s : s + micro]]
        ids, lab, att = collate(items, pad_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tot += sum_loss(model, ids, lab, att).item()
        n += int((lab[:, 1:] != -100).sum())
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local path of the HF base model")
    ap.add_argument("--task", required=True, choices=["gsm8k", "platypus", "boolq"])
    ap.add_argument("--method", required=True, choices=["mlora", "lora", "mlora_sum", "mlora_k2", "dylora"])
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch", type=int, default=32, help="effective batch (sequences)")
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ranks-per-step", type=int, default=0, help="mlora_sum: 0 = all of S, else K incl. r_max")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--val-ranks", default="", help="comma list; default all powers of two <= rank")
    ap.add_argument("--max-steps", type=int, default=0, help="debug: stop after N steps")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done.json").exists():
        print("already done:", out)
        return
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    model.config.use_cache = False
    train_ranks = pow2_ranks(args.rank)
    P = p_vector(args.rank, train_ranks) if args.method == "mlora" else None
    mods = inject(model, args.rank, P)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for m in mods.values() for p in (m.lora_A_t, m.lora_B)]
    n_params = sum(p.numel() for p in params)

    t0 = time.time()
    train, val = load_task(args.task, tok, args.max_len, seed=42)  # split seed fixed across runs
    steps_per_epoch = len(train) // args.batch
    total = int(steps_per_epoch * args.epochs)
    if args.max_steps:
        total = min(total, args.max_steps)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / total))
    step, log = 0, []
    ck = out / "ckpt.pt"
    if ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)
        load_lora_state(mods, st["lora"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, log = st["step"], st["log"]
        torch.set_rng_state(st["rng"])
        print(f"resumed from step {step}")
    meta = dict(vars(args), train_ranks=train_ranks, n_train=len(train), n_val=len(val), total_steps=total,
                trainable_params=n_params, P=None if P is None else P.tolist(), gpu=torch.cuda.get_device_name(),
                torch=torch.__version__)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "P"}))

    def order_for_epoch(e):
        o = list(range(len(train)))
        random.Random(args.seed * 1000 + e).shuffle(o)
        return o

    model.train()
    set_rank(mods, None)
    t_train = time.time()
    run_time = 0.0 if not log else log[-1].get("elapsed", 0.0)
    while step < total:
        rank_rng = random.Random(args.seed * 7919 + step)
        e, i = divmod(step, steps_per_epoch)
        o = order_for_epoch(e)
        batch = [train[j] for j in o[i * args.batch : (i + 1) * args.batch]]
        ntok = sum(sum(1 for l in b["labels"][1:] if l != -100) for b in batch)
        loss_acc = 0.0
        if args.method in ("mlora_sum", "mlora_k2"):
            others = [r for r in train_ranks if r != args.rank]
            kk = 2 if args.method == "mlora_k2" else args.ranks_per_step
            k = len(train_ranks) if kk <= 0 else min(kk, len(train_ranks))
            step_ranks = [args.rank] + (others if k == len(train_ranks) else rank_rng.sample(others, k - 1))
        elif args.method == "dylora":  # DyLoRA (Valipour et al. 2023): one rank drawn uniformly from S per step
            step_ranks = [rank_rng.choice(train_ranks)]
        else:
            step_ranks = [None]
        for s in range(0, len(batch), args.micro):
            ids, lab, att = collate(batch[s : s + args.micro], tok.pad_token_id, device)
            for r in step_ranks:
                set_rank(mods, r)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = sum_loss(model, ids, lab, att) / (ntok * len(step_ranks))
                loss.backward()
                loss_acc += loss.item()
        set_rank(mods, None)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0 or step == total:
            el = run_time + time.time() - t_train
            log.append(dict(step=step, loss=loss_acc, grad_norm=gn, lr=sched.get_last_lr()[0], elapsed=el))
            print(f"step {step}/{total} loss {loss_acc:.4f} gn {gn:.3f} lr {sched.get_last_lr()[0]:.2e} {el:.0f}s", flush=True)
        if step % args.ckpt_every == 0 and step < total:
            torch.save(dict(lora=lora_state(mods), opt=opt.state_dict(), sched=sched.state_dict(), step=step, log=log,
                            rng=torch.get_rng_state()), str(ck) + ".tmp")
            os.replace(str(ck) + ".tmp", ck)
    train_secs = run_time + time.time() - t_train

    # validation loss per inference rank (LR selection), then save final weights
    vr = [int(x) for x in args.val_ranks.split(",")] if args.val_ranks else train_ranks
    vl = {}
    t_val = time.time()
    for r in vr:
        set_rank(mods, r)
        vl[r] = val_loss(model, val, tok.pad_token_id, device, args.micro * 2)
        print(f"val loss rank {r}: {vl[r]:.4f}", flush=True)
    # final weights in bf16 (what serving uses; scratch quota is tight)
    torch.save({k: (a.to(torch.bfloat16), b.to(torch.bfloat16)) for k, (a, b) in lora_state(mods).items()},
               out / "lora_final.pt")
    done = dict(val_loss=vl, train_secs=train_secs, val_secs=time.time() - t_val, wall_secs=time.time() - t0,
                peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9, log=log)
    (out / "done.json").write_text(json.dumps(done, indent=1))
    if ck.exists():
        ck.unlink()
    print("done", out, f"train {train_secs:.0f}s")


if __name__ == "__main__":
    main()
