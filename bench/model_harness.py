"""Whole-model harness: Qwen2.5-0.5B in plain PyTorch (transformers) with swappable LoRA backends.

Every one of the 168 adapted projections (q, k, v, o, gate, up, down in 24 layers) is wrapped in
LoRALinear: out = base(x), then the Matryoshka LoRA delta is added into out by one backend:

    none     no LoRA
    torch    reference: fp32 shrink, per-token rank mask, fp32 expand (slow, obviously correct)
    b1       b1_padded's body (PyTorch: pad to the batch maximum rank, mask, two cuBLAS matmuls)
    b3       vLLM's _lora_shrink / _lora_expand, the rank tiers registered as padded adapters
             (rankfuse/baselines/vllm_ops.py); metadata built once per forward for all layers
    rf1      rf1_fused's split-K path (rankfuse_lora, add=True)
    rf2seg   segment RF2 (rankfuse_rf2_segment.py), add=True
    rf2bkt   bucketed RF2 (rankfuse_rf2_bucketed.py, per-bucket kernels on concurrent streams)
    rf2auto  per batch, rf2bkt where use_rf2 picks it (only with ctx.graphed), else rf1

Each request carries one rank; a forward call sets the per-token ranks once (Ctx.set_batch) and
all layers share them, as a server would. Batches are flattened request-major (token b*S + s).
The adapter is the generated Matryoshka adapter (scripts/make_adapters.py --model qwen2.5-0.5b
--r-max 64 --out $RF_ADAPTERS/qwen2.5-0.5b).

    python bench/model_harness.py check     # every backend against the reference (GPU job)
"""

import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rankfuse  # noqa: E402,F401  (registers implementations)
from rankfuse.baselines import padded, vllm_ops  # noqa: E402
from rankfuse.interface import MatryoshkaLoRA  # noqa: E402
from rankfuse.kernels import rankfuse_fused as rf  # noqa: E402
from rankfuse.kernels import rankfuse_rf2_bucketed as bkt  # noqa: E402
from rankfuse.kernels import rankfuse_rf2_segment as seg  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = os.path.join(os.environ.get("RF_ADAPTERS", "."), "qwen2.5-0.5b", "matryoshka-0", "adapter_model.safetensors")
TIERS = [1, 2, 4, 8, 16, 32, 64]
BACKENDS = ["none", "torch", "b1", "b3", "rf1", "rf2seg", "rf2bkt", "rf2auto"]
PROJ = {"self_attn": ["q_proj", "k_proj", "v_proj", "o_proj"], "mlp": ["gate_proj", "up_proj", "down_proj"]}


class Ctx:
    """State shared by all LoRA layers for one forward call."""

    def __init__(self, r_max=64):
        self.backend = "none"
        self.graphed = True  # the forward is replayed from a CUDA graph (rf2auto's rule needs it)
        self.r_max = r_max
        self.copy_of_rank = None

    def set_batch(self, request_ranks, tokens_per_request, device="cuda"):
        ranks = torch.tensor(request_ranks, dtype=torch.int32).repeat_interleave(tokens_per_request).to(device)
        self.ranks = ranks
        self.r = int(max(request_ranks))
        if self.copy_of_rank is None:
            self.copy_of_rank = torch.full((self.r_max + 1,), -1, dtype=torch.int32, device=device)
            for i, r in enumerate(TIERS):
                self.copy_of_rank[r] = i
        self.b3_step = vllm_ops.Step(self.copy_of_rank, len(TIERS), self.r_max, ranks)
        self.plan = bkt.Plan(ranks, device)
        self.auto_rf2 = bkt.use_rf2(self.plan, tokens_per_request, self.graphed)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, lora_a: torch.Tensor, lora_b: torch.Tensor, ctx: Ctx):
        super().__init__()
        self.base, self.ctx = base, ctx
        dt, dev = base.weight.dtype, base.weight.device
        self.A_t = lora_a.to(dev, dt).contiguous()  # [r_max, in]  (PEFT lora_A layout = RankFuse A_t)
        self.B = lora_b.to(dev, dt).T.contiguous()  # [r_max, out]
        self.lora = MatryoshkaLoRA(self.A_t, self.B)
        self.b3 = vllm_ops.AdapterCopies(self.lora, TIERS)  # same copy order as ctx.copy_of_rank

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        y = self.base(x2)
        c = self.ctx
        if c.backend != "none" and c.r > 0:
            if c.backend == "torch":
                v = x2.float() @ self.A_t.float().T
                v = v * (torch.arange(v.shape[1], device=v.device) < c.ranks[:, None]).float()
                y = (y.float() + v @ self.B.float()).to(y.dtype)
            elif c.backend == "b1":
                y += padded._padded_body(x2, self.lora, c.ranks, c.r, None)
            elif c.backend == "b3":
                vllm_ops.apply(x2, self.b3, c.b3_step, y, add=True)
            elif c.backend == "rf1" or (c.backend == "rf2auto" and not c.auto_rf2):
                rf.rankfuse_lora(x2, self.A_t, self.B, c.ranks, out=y, add=True, r_limit=c.r)
            elif c.backend == "rf2seg":
                seg.rf2_segment(x2, self.A_t, self.B, c.ranks, c.r, out=y, add=True)
            elif c.backend in ("rf2bkt", "rf2auto"):
                bkt.rf2_bucketed(x2, self.A_t, self.B, c.ranks, c.plan, out=y, add=True)
            else:
                raise ValueError(c.backend)
        return y.reshape(*shape[:-1], y.shape[-1])


def load_model():
    """Qwen2.5-0.5B (fp16, sdpa) with all 168 projections wrapped; returns (model, ctx)."""
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
    weights = load_file(ADAPTER)
    ctx = Ctx()
    n = 0
    for i, layer in enumerate(model.model.layers):
        for block, names in PROJ.items():
            mod = getattr(layer, block)
            for name in names:
                key = f"base_model.model.model.layers.{i}.{block}.{name}"
                wrapped = LoRALinear(getattr(mod, name), weights[f"{key}.lora_A.weight"], weights[f"{key}.lora_B.weight"], ctx)
                setattr(mod, name, wrapped)
                n += 1
    assert n == 168, n
    return model, ctx


def logits(model, ctx, backend, ids, request_ranks):
    """Prefill logits [B, S, V] and one decode step's logits [B, V] for one backend."""
    ctx.backend = backend
    B, S = ids.shape
    with torch.no_grad():
        ctx.set_batch(request_ranks, S)
        out = model(input_ids=ids, use_cache=True)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        ctx.set_batch(request_ranks, 1)
        dec = model(input_ids=nxt, past_key_values=out.past_key_values, use_cache=True)
    return out.logits.float(), dec.logits[:, -1].float()


def compare(a, b):
    return (a - b).abs().max().item(), (a - b).abs().mean().item(), (a.argmax(-1) == b.argmax(-1)).float().mean().item()


def check():
    """Every backend against the fp32 reference (prefill and one decode step), the unwrapped
    model against the plain model, and rank routing inside a mixed batch."""
    print(torch.cuda.get_device_name(), flush=True)
    plain = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
    model, ctx = load_model()
    gen = torch.Generator().manual_seed(0)
    cases = [
        ("8 requests x 64 tokens, ranks 1..64 + rank 0", torch.randint(1000, 30000, (8, 64), generator=gen), [1, 2, 4, 8, 16, 32, 64, 0]),
        ("4 requests x 200 tokens (>128 tokens: tiled paths)", torch.randint(1000, 30000, (4, 200), generator=gen), [4, 64, 16, 1]),
        ("16 requests x 16 tokens, mostly rank 4 + one 64", torch.randint(1000, 30000, (16, 16), generator=gen), [4] * 15 + [64]),
    ]
    failures = 0
    with torch.no_grad():
        for label, ids, ranks in cases:
            ids = ids.cuda()
            print(f"\n=== {label} ===", flush=True)
            ref_p, ref_d = logits(model, ctx, "torch", ids, ranks)
            none_p, none_d = logits(model, ctx, "none", ids, ranks)
            plain_p = plain(input_ids=ids).logits.float()
            mx, mn, ag = compare(none_p, plain_p)
            ok = mx < 1e-3
            failures += not ok
            print(f"  none vs plain model     max {mx:.2e}  mean {mn:.2e}  top-1 agree {ag:.4f}  {'ok' if ok else 'FAIL'}")
            mx, mn, ag = compare(ref_p, none_p)
            print(f"  LoRA effect (torch vs none): max {mx:.3f}  mean {mn:.4f}  top-1 agree {ag:.4f}  (signal size)")
            signal = mn
            for backend in BACKENDS[2:]:
                p, d = logits(model, ctx, backend, ids, ranks)
                pmx, pmn, pag = compare(p, ref_p)
                dmx, dmn, dag = compare(d, ref_d)
                ok = pmn < 0.1 * signal and pag >= 0.99 and dmn < 0.1 * signal and dag >= 0.99
                failures += not ok
                print(f"  {backend:<7} prefill: max {pmx:.3f} mean {pmn:.4f} top-1 {pag:.4f} | decode: max {dmx:.3f} "
                      f"mean {dmn:.4f} top-1 {dag:.4f}  {'ok' if ok else 'FAIL'}", flush=True)
        # rank routing: request b alone at its rank gives the same logits as inside the mixed batch
        ids = cases[0][1].cuda()
        ranks = cases[0][2]
        mixed, _ = logits(model, ctx, "rf1", ids, ranks)
        worst = 0.0
        for b in (0, 3, 6, 7):
            alone, _ = logits(model, ctx, "rf1", ids[b:b + 1], [ranks[b]])
            worst = max(worst, (alone[0] - mixed[b]).abs().mean().item())
        ok = worst < 0.1 * signal
        failures += not ok
        print(f"\nrank routing (rf1): request alone vs in mixed batch, worst mean diff {worst:.4f}  {'ok' if ok else 'FAIL'}")
    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
    return failures


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        sys.exit(1 if check() else 0)
    print(__doc__)
