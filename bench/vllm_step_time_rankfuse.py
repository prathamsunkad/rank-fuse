"""Level-2 step time: stock vLLM LoRA vs RankFuse inside vLLM 0.32 (Qwen2.5-0.5B-Instruct).

Same method as bench/vllm_step_time.py (one engine per process, prefix caching off, wall time of
llm.generate, median of 3 decode / 5 prefill runs after a warm-up):

    decode    B requests (B = 1, 8, 32, 128) with 128-token random prompts; generate 16 and 144
              tokens with ignore_eos; step time = (t_144 - t_16) / 128, which removes the prefill
    prefill   one request of L prompt tokens (L = 128, 512, 2048, 8192), max_tokens = 1
    prefill8  the same L tokens as 8 requests of L/8 tokens (a mixed-rank prefill batch)

Engines (mode):
    base       no LoRA (enable_lora=False)
    stock64    stock vLLM, max_lora_rank 64, max_loras 1, every request on the rank-64 copy
               (matryoshka-0-r64)
    stockmix   stock vLLM, max_lora_rank 64, max_loras 7 (= tiers present): request i uses the
               per-tier copy matryoshka-0-r{MIX[i % 8]}, each copy its own adapter and slot
    rfmix      RANKFUSE_VLLM=1, max_loras 1, one adapter (matryoshka-0) with
               matryoshka_rank = MIX[i % 8]
    rf64       RANKFUSE_VLLM=1, every request at matryoshka_rank 64

MIX = 64, 1, 2, 4, 8, 16, 32, 64: every tier, uniform, with r_max twice (so a batch of 8 or more
always contains rank 64, which is the worst case for padding to the batch maximum). Batch 1
decode and the one-request prefill use the first request's rank: 64 for every engine, so those
rows compare kernels at rank 64. prefill8 has all tiers.

    python bench/vllm_step_time_rankfuse.py <mode> <graphs|eager> <out.jsonl>     # GPU job
    python bench/vllm_step_time_rankfuse.py summary <out.jsonl> [out.md]
"""

import json
import os
import random
import statistics
import sys
import time

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTERS = os.path.join(os.environ.get("RF_ADAPTERS", "."), "qwen2.5-0.5b")
BATCHES = [1, 8, 32, 128]
PROMPT_LENS = [128, 512, 2048, 8192]
SHORT, LONG = 16, 144
MIX = [64, 1, 2, 4, 8, 16, 32, 64]
MODES = ["base", "stock64", "stockmix", "rfmix", "rf64"]


def run(mode, graphs, out_path):
    assert mode in MODES, mode
    rankfuse = mode.startswith("rf")
    assert (os.environ.get("RANKFUSE_VLLM") == "1") == rankfuse, "RANKFUSE_VLLM=1 exactly for rf* modes"
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    kwargs = dict(model=MODEL, max_model_len=16384, max_num_batched_tokens=16384, max_num_seqs=256,
                  gpu_memory_utilization=0.6, enforce_eager=(graphs == "eager"), enable_prefix_caching=False,
                  seed=0)
    full = os.path.join(ADAPTERS, "matryoshka-0")
    if mode != "base":
        kwargs.update(enable_lora=True, max_lora_rank=64, max_loras=7 if mode == "stockmix" else 1)
        if mode == "stockmix":
            kwargs.update(max_cpu_loras=7)
    if rankfuse:
        from rankfuse.integration import MatryoshkaLoRARequest

        def req(i):
            return MatryoshkaLoRARequest("matryoshka-0", 1, full, matryoshka_rank=64 if mode == "rf64" else MIX[i % len(MIX)])
    elif mode == "stock64":
        def req(i):
            return LoRARequest("r64", 1, os.path.join(ADAPTERS, "matryoshka-0-r64"))
    elif mode == "stockmix":
        tier_ids = {r: k + 1 for k, r in enumerate(sorted(set(MIX)))}

        def req(i):
            r = MIX[i % len(MIX)]
            return LoRARequest(f"matryoshka-0-r{r}", tier_ids[r], os.path.join(ADAPTERS, f"matryoshka-0-r{r}"))
    else:
        def req(i):
            return None

    t0 = time.perf_counter()
    llm = LLM(**kwargs)
    startup = time.perf_counter() - t0
    from rankfuse.integration import vllm_plugin

    # The plugin must really be loaded (a stale rankfuse.egg-info on sys.path can hide its entry point).
    assert vllm_plugin._DONE == (os.environ.get("RANKFUSE_VLLM") == "1"), "RankFuse plugin state does not match RANKFUSE_VLLM"
    rng = random.Random(0)

    def prompts(n, length):
        return [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 30000) for _ in range(length)]) for _ in range(n)]

    def timed(ps, max_tokens):
        sp = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
        reqs = [req(i) for i in range(len(ps))]
        t = time.perf_counter()
        outs = llm.generate(ps, sp, lora_request=None if reqs[0] is None else reqs, use_tqdm=False)
        dt = time.perf_counter() - t
        assert all(len(o.outputs[0].token_ids) == max_tokens for o in outs)
        return dt

    res = {"decode_ms": {}, "prefill_ms": {}, "prefill8_ms": {}}
    for b in BATCHES:
        ps = prompts(b, 128)
        timed(ps, SHORT)  # warm-up
        steps = [(timed(ps, LONG) - timed(ps, SHORT)) / (LONG - SHORT) * 1e3 for _ in range(3)]
        res["decode_ms"][b] = statistics.median(steps)
        print(f"{mode} {graphs} decode B={b}: {res['decode_ms'][b]:.3f} ms/step", flush=True)
    for length in PROMPT_LENS:
        for key, n in (("prefill_ms", 1), ("prefill8_ms", 8)):
            ps = prompts(n, length // n)
            timed(ps, 1)
            res[key][length] = statistics.median(timed(ps, 1) * 1e3 for _ in range(5))
        print(f"{mode} {graphs} prefill L={length}: {res['prefill_ms'][length]:.2f} ms, "
              f"8 requests: {res['prefill8_ms'][length]:.2f} ms", flush=True)
    row = dict(mode=mode, graphs=graphs, startup_s=startup, mix=MIX, **res)
    with open(out_path, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def summary(path, md_path=None):
    rows = [json.loads(line) for line in open(path)]
    by = {(r["mode"], r["graphs"]): r for r in rows}  # the last row of a (mode, graphs) pair wins
    out = []
    for graphs in ("graphs", "eager"):
        base = by.get(("base", graphs))
        if base is None:
            continue
        modes = [m for m in MODES if (m, graphs) in by]
        out.append(f"\n### CUDA graphs {'on (default)' if graphs == 'graphs' else 'off (enforce_eager)'}\n")
        for key, title, sizes, label in (
            ("decode_ms", "Decode step time, ms (LoRA share of the step)", BATCHES, "batch"),
            ("prefill_ms", "Prefill, one request of L tokens, ms (LoRA share)", PROMPT_LENS, "tokens"),
            ("prefill8_ms", "Prefill, 8 requests of L/8 tokens (mixed ranks), ms (LoRA share)", PROMPT_LENS, "tokens"),
        ):
            out.append(f"{title}\n")
            extra = [h for h in ("stock64/rf64", "stockmix/rfmix") if all((m, graphs) in by for m in h.split("/"))]
            out.append("| " + label + " | " + " | ".join(modes) + "".join(f" | {h}" for h in extra) + " |")
            out.append("|---" * (1 + len(modes) + len(extra)) + "|")
            for s in sizes:
                cells = []
                for m in modes:
                    t = by[(m, graphs)][key][str(s)]
                    b0 = base[key][str(s)]
                    cells.append(f"{t:.3f}" if m == "base" else f"{t:.3f} ({(t - b0) / t * 100:.0f}%)")
                for h in extra:
                    a, b = h.split("/")
                    cells.append(f"{by[(a, graphs)][key][str(s)] / by[(b, graphs)][key][str(s)]:.2f}x")
                out.append(f"| {s} | " + " | ".join(cells) + " |")
            out.append("")
    out.append("Engine startup, s: " + ", ".join(f"{m}/{g} {r['startup_s']:.0f}" for (m, g), r in sorted(by.items())))
    text = "\n".join(out)
    print(text)
    if md_path:
        with open(md_path, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        run(sys.argv[1], sys.argv[2], sys.argv[3])
