"""How much of a model step is LoRA? Stock vLLM (0.32), Qwen2.5-0.5B-Instruct (MatLoRA plan, P2.2).

One engine per process (vLLM does not release the GPU cleanly):
    base     no LoRA (enable_lora=False)
    lora64   enable_lora, max_lora_rank 64, one rank-64 adapter (our generated Matryoshka copy)
    lora16   enable_lora, max_lora_rank 16, one rank-16 adapter (vLLM pads every adapter to
             max_lora_rank, so a mid rank only costs less in an engine configured for it)
each with CUDA graphs on (default) or off (enforce_eager). LoRA engines also time requests
that carry no adapter ("lora enabled, unused").

Decode step time at B requests: prompts of 128 random tokens, generate 16 and 144 tokens with
ignore_eos; (t_144 - t_16) / 128 removes the prefill. Prefill: one request of L prompt tokens,
max_tokens=1. Prefix caching is off so repeated prompts are recomputed. Wall-clock time of
llm.generate, median of 3 (decode) or 5 (prefill) runs after a warm-up.

    python bench/vllm_step_time.py <base|lora64|lora16> <graphs|eager> <out.jsonl>   # GPU job
    python bench/vllm_step_time.py summary <out.jsonl>

Needs the adapter copies from scripts/make_adapters.py (--model qwen2.5-0.5b --r-max 64 --out
$RF_ADAPTERS/qwen2.5-0.5b). Measured on an H200 with graphs on: LoRA is 52/44/32/8% of a decode
step at batch 1/8/32/128 and 3-63% of a prefill; rank 16 costs barely less than rank 64, so the
cost looks dominated by per-call overhead, not GPU math. A 0.5B model overstates the share.
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


def run(mode, graphs, out_path):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    kwargs = dict(model=MODEL, max_model_len=16384, max_num_batched_tokens=16384, max_num_seqs=256,
                  gpu_memory_utilization=0.6, enforce_eager=(graphs == "eager"), enable_prefix_caching=False,
                  seed=0)
    lora = None
    if mode != "base":
        rank = int(mode[4:])
        kwargs.update(enable_lora=True, max_lora_rank=rank, max_loras=1)
        lora = LoRARequest(f"r{rank}", 1, os.path.join(ADAPTERS, f"matryoshka-0-r{rank}"))
    t0 = time.perf_counter()
    llm = LLM(**kwargs)
    startup = time.perf_counter() - t0
    rng = random.Random(0)

    def prompts(n, length):
        return [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 30000) for _ in range(length)]) for _ in range(n)]

    def timed(ps, max_tokens, req):
        sp = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
        t = time.perf_counter()
        outs = llm.generate(ps, sp, lora_request=req, use_tqdm=False)
        dt = time.perf_counter() - t
        assert all(len(o.outputs[0].token_ids) == max_tokens for o in outs)
        return dt

    def measure(req):
        res = {"decode_ms": {}, "prefill_ms": {}}
        for b in BATCHES:
            ps = prompts(b, 128)
            timed(ps, SHORT, req)  # warm-up
            steps = []
            for _ in range(3):
                steps.append((timed(ps, LONG, req) - timed(ps, SHORT, req)) / (LONG - SHORT) * 1e3)
            res["decode_ms"][b] = statistics.median(steps)
        for length in PROMPT_LENS:
            ps = prompts(1, length)
            timed(ps, 1, req)
            res["prefill_ms"][length] = statistics.median(timed(ps, 1, req) * 1e3 for _ in range(5))
        return res

    rows = [dict(mode=mode, graphs=graphs, adapter=lora is not None, startup_s=startup, **measure(lora))]
    if lora is not None:
        rows.append(dict(mode=mode + "-unused", graphs=graphs, adapter=False, startup_s=startup, **measure(None)))
    with open(out_path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for r in rows:
        print(json.dumps(r), flush=True)


def summary(path):
    rows = [json.loads(l) for l in open(path)]
    by = {(r["mode"], r["graphs"]): r for r in rows}
    for graphs in ("graphs", "eager"):
        base = by.get(("base", graphs))
        if base is None:
            continue
        print(f"\n=== CUDA graphs {'ON' if graphs == 'graphs' else 'OFF (eager)'} ===")
        modes = [m for m in ("base", "lora16-unused", "lora16", "lora64-unused", "lora64") if (m, graphs) in by]
        print("decode step time, ms (LoRA share = extra time over base / LoRA step time)")
        print(f"{'batch':<8}" + "".join(f"{m:>16}" for m in modes))
        for b in BATCHES:
            line = f"{b:<8}"
            for m in modes:
                t = by[(m, graphs)]["decode_ms"][str(b)]
                share = (t - base["decode_ms"][str(b)]) / t * 100
                line += f"{t:>9.2f}" + (f" ({share:3.0f}%)" if m != "base" else " " * 7)
            print(line)
        print("prefill time, ms (one request)")
        print(f"{'tokens':<8}" + "".join(f"{m:>16}" for m in modes))
        for length in PROMPT_LENS:
            line = f"{length:<8}"
            for m in modes:
                t = by[(m, graphs)]["prefill_ms"][str(length)]
                share = (t - base["prefill_ms"][str(length)]) / t * 100
                line += f"{t:>9.2f}" + (f" ({share:3.0f}%)" if m != "base" else " " * 7)
            print(line)
    print("\nengine startup, s: " + ", ".join(f"{m}/{g} {r['startup_s']:.0f}" for (m, g), r in by.items() if not m.endswith("unused")))


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary(sys.argv[2])
    else:
        run(sys.argv[1], sys.argv[2], sys.argv[3])
