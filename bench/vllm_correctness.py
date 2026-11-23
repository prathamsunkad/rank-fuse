"""P4.6 correctness gate: RankFuse inside vLLM 0.32 against stock vLLM (Qwen2.5-0.5B-Instruct).

Each engine runs in its own process (vLLM does not release the GPU cleanly); `compare` checks
the saved outputs. Every engine: enable_lora, max_lora_rank 64, max_loras 1, bf16, default
prefix caching (on), CUDA graphs on (default) or off (enforce_eager).

Two kinds of output per case, on 24 prompts:
    score   teacher-forced: each engine scores the SAME token sequences (prompt + a fixed
            reference answer, ~110 tokens) with prompt_logprobs=2, i.e. the logprob of every
            given token and the top-2 tokens at every position. No trajectory divergence, so
            this is the gate.
    gen     greedy generation of 48 tokens with top-5 logprobs (reported, not gated, see below).

Cases:
    stock     RANKFUSE_VLLM unset.
                base          no adapter
                full          matryoshka-0 (the r_max = 64 adapter)
                tier{r}       the per-tier copy matryoshka-0-r{r}, r in 1, 2, 4, ..., 64
                solo_base     control: every prompt alone, no adapter
                solo_full     control: every prompt alone, matryoshka-0
    rankfuse  RANKFUSE_VLLM=1, one adapter (matryoshka-0, lora_int_id 1) and per-request ranks:
                base, full    as above (full: matryoshka_rank None = the whole adapter)
                tier{r}       every request at matryoshka_rank r
                mixed         one batch, request i at rank MIX[i % len(MIX)] (0 = no adapter)
                solo          every request of `mixed` alone, one call each

    python bench/vllm_correctness.py run <stock|rankfuse> <graphs|eager> out.json   # GPU job
    python bench/vllm_correctness.py compare <dir with {stock,rankfuse}_{graphs,eager}.json>

Why not bit-exact, and the rule used instead. Stock vLLM's shrink accumulates split-K partials
with atomics (nondeterministic order) and RankFuse sums rank components in a different order, so
the two agree to bf16 rounding, not bit for bit. The base model alone is not batch-invariant
either: the bf16 GEMMs pick different algorithms per token count, so stock vLLM's own logprobs
move when only the batch composition changes. The stock controls (each prompt alone vs in the
batch) measure that noise, and the gate compares RankFuse against it:

    pass  (1) top-1 agreement on every DECISIVE position (the reference's top-1 leads its top-2
              by >= DECISIVE nats; logits come out of the bf16 lm_head in steps of 0.125 at the
              magnitudes seen here), and
          (2) mean |d logprob| of the scored tokens <= NOISE_FACTOR x the stock control's mean
              (floor 0.01), with the 99th percentile likewise.

The greedy generations are reported as exact / near-tie / diverged sequence counts next to the
same counts for the stock control: once two runs pick different tokens at a near-tie they
continue from different prefixes, so a free-running comparison mostly measures tie flips.
VLLM_BATCH_INVARIANT=1 was not used: it changes stock vLLM's kernels (the reference would no
longer be the default engine), and RankFuse's split-K configuration depends on the token count,
so it is not batch-invariant either.
"""

import json
import os
import statistics
import sys
import time

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTERS = os.path.join(os.environ.get("RF_ADAPTERS", "."), "qwen2.5-0.5b")
TIERS = [1, 2, 4, 8, 16, 32, 64]
MIX = [64, 1, 8, 0, 2, 32, 4, 16, 0, 64, 16, 1]  # 0 = request without an adapter
MAX_TOKENS = 48
TIE_NATS = 0.25  # greedy report: near-tie = the two tokens within two bf16 logit steps
DECISIVE = 0.5  # gate (1): positions where the reference's top-1 leads by at least this
NOISE_FACTOR = 2.0  # gate (2)

TOPICS = [
    "the water cycle", "how a CPU cache works", "the causes of the French Revolution", "photosynthesis",
    "how vaccines train the immune system", "the rules of chess", "plate tectonics", "how a bill becomes law",
    "the history of the printing press", "black holes", "how compilers optimize loops", "the Krebs cycle",
    "supply and demand", "how GPS finds your position", "the life cycle of a star", "TCP congestion control",
    "the Roman Republic", "how neural networks learn", "ocean tides", "the theory of evolution",
    "how a refrigerator works", "public key cryptography", "the structure of DNA", "monsoons",
]
ANSWER = ("Sure. Here is a short explanation of {t}. The main idea is simple once you see the parts and how they "
          "connect. For example, imagine a small everyday situation where {t} shows up, and follow each step. "
          "A common misconception is that it happens all at once, but in fact it is a gradual process.")


def prompts():
    # Long enough (> 2 KV blocks of 16 tokens) that prefix caching has full blocks to reuse.
    return [
        f"You are a patient teacher. A student asks you to explain {t} in simple words, with one concrete "
        f"example and one common misconception. Answer in a short paragraph. Topic: {t}."
        for t in TOPICS
    ]


def score_texts():
    return [p + "\n" + ANSWER.format(t=t) for p, t in zip(prompts(), TOPICS)]


def run(engine, graphs, out_path):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(model=MODEL, enable_lora=True, max_lora_rank=64, max_loras=1, max_model_len=2048,
              max_num_seqs=64, gpu_memory_utilization=0.45, enforce_eager=(graphs == "eager"), seed=0)
    from rankfuse.integration import vllm_plugin

    # The plugin must really be loaded (a stale rankfuse.egg-info on sys.path can hide its entry point).
    assert vllm_plugin._DONE == (engine == "rankfuse") == (os.environ.get("RANKFUSE_VLLM") == "1"), \
        "RankFuse plugin state does not match the engine / RANKFUSE_VLLM"
    gen_sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, logprobs=5)
    score_sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=2)
    ps, ss = prompts(), score_texts()
    full_path = os.path.join(ADAPTERS, "matryoshka-0")

    def gen(prompt_list, reqs):
        outs = llm.generate(prompt_list, gen_sp, lora_request=reqs, use_tqdm=False)
        res = []
        for o in outs:
            c = o.outputs[0]
            top = [sorted(((t, lp.logprob) for t, lp in d.items()), key=lambda e: -e[1])[:5] for d in c.logprobs]
            res.append({"tokens": list(c.token_ids), "top": top, "text": c.text})
        return res

    def score(text_list, reqs):
        outs = llm.generate(text_list, score_sp, lora_request=reqs, use_tqdm=False)
        res = []
        for o in outs:
            ids = o.prompt_token_ids
            rows = []
            for i, d in enumerate(o.prompt_logprobs):
                if d is None:
                    continue
                ranked = sorted(((t, lp.logprob) for t, lp in d.items()), key=lambda e: -e[1])
                rows.append([ids[i], d[ids[i]].logprob, ranked[0][0], ranked[0][1],
                             ranked[1][1] if len(ranked) > 1 else None])
            res.append(rows)  # per position: token, its logprob, top-1 token, top-1 lp, top-2 lp
        return res

    def both(name, reqs_for, solo=False):
        if solo:
            cases["gen_" + name] = [gen([p], reqs_for(i))[0] for i, p in enumerate(ps)]
            cases["score_" + name] = [score([s], reqs_for(i))[0] for i, s in enumerate(ss)]
        else:
            reqs = [reqs_for(i) for i in range(len(ps))]
            reqs = None if all(r is None for r in reqs) else reqs
            cases["gen_" + name] = gen(ps, reqs)
            cases["score_" + name] = score(ss, reqs)

    cases = {}
    t0 = time.time()
    both("base", lambda i: None)
    if engine == "stock":
        both("full", lambda i: LoRARequest("matryoshka-0", 1, full_path))
        for r in TIERS:
            path = os.path.join(ADAPTERS, f"matryoshka-0-r{r}")
            both(f"tier{r}", lambda i, r=r, path=path: LoRARequest(f"matryoshka-0-r{r}", 100 + r, path))
        both("solo_base", lambda i: None, solo=True)
        both("solo_full", lambda i: LoRARequest("matryoshka-0", 1, full_path), solo=True)
    else:
        from rankfuse.integration import MatryoshkaLoRARequest

        def req(r):
            return None if r == 0 else MatryoshkaLoRARequest("matryoshka-0", 1, full_path, matryoshka_rank=r)

        both("full", lambda i: MatryoshkaLoRARequest("matryoshka-0", 1, full_path))
        for r in TIERS:
            both(f"tier{r}", lambda i, r=r: req(r))
        both("mixed", lambda i: req(MIX[i % len(MIX)]))
        both("solo", lambda i: req(MIX[i % len(MIX)]), solo=True)
        cases["mixed_ranks"] = [MIX[i % len(MIX)] for i in range(len(ps))]
    cases["_meta"] = {"engine": engine, "graphs": graphs, "seconds": time.time() - t0,
                      "rankfuse_vllm": os.environ.get("RANKFUSE_VLLM", "0")}
    with open(out_path, "w") as f:
        json.dump(cases, f)
    print(f"wrote {out_path} ({time.time() - t0:.0f} s of generation)", flush=True)


# ------------------------------------------------------------------------------- comparison

def score_diff(xs, ys):
    """Teacher-forced comparison of two runs (ys = reference): |d logprob| stats, decisive top-1."""
    d, decisive, agree = [], 0, 0
    for a, b in zip(xs, ys):
        assert [r[0] for r in a] == [r[0] for r in b], "scored sequences differ"
        for ra, rb in zip(a, b):
            d.append(abs(ra[1] - rb[1]))
            if rb[4] is not None and rb[3] - rb[4] >= DECISIVE:
                decisive += 1
                agree += ra[2] == rb[2]
    d.sort()
    return {"mean": statistics.mean(d), "p99": d[int(0.99 * (len(d) - 1))], "max": d[-1], "positions": len(d),
            "decisive": decisive, "decisive_agree": agree}


def greedy_seq(a, b):
    """('exact' | 'tie' | 'diverged', first differing position or None)."""
    for i, (x, y) in enumerate(zip(a["tokens"], b["tokens"])):
        if x != y:
            gaps = []
            for run in (a, b):
                lp = dict((t, l) for t, l in run["top"][i])
                if x not in lp or y not in lp:
                    return "diverged", i
                gaps.append(abs(lp[x] - lp[y]))
            return ("tie" if max(gaps) < TIE_NATS else "diverged"), i
    return ("exact" if len(a["tokens"]) == len(b["tokens"]) else "diverged"), None


def greedy_stats(xs, ys):
    stats = {"exact": 0, "tie": 0, "diverged": 0}
    for a, b in zip(xs, ys):
        stats[greedy_seq(a, b)[0]] += 1
    return stats


def compare(d):
    results, all_ok = [], True
    for graphs in ("graphs", "eager"):
        sp, rp = os.path.join(d, f"stock_{graphs}.json"), os.path.join(d, f"rankfuse_{graphs}.json")
        if not (os.path.exists(sp) and os.path.exists(rp)):
            print(f"missing {sp} or {rp}")
            all_ok = False
            continue
        s, r = json.load(open(sp)), json.load(open(rp))
        ranks = r["mixed_ranks"]

        def stock_at(kind):
            return [s[f"{kind}_base"][i] if k == 0 else s[f"{kind}_tier{k}"][i] for i, k in enumerate(ranks)]

        control = score_diff(s["score_full"], s["score_solo_full"])
        control_base = score_diff(s["score_base"], s["score_solo_base"])
        noise = max(control["mean"], control_base["mean"], 0.01)
        noise99 = max(control["p99"], control_base["p99"], 0.01)
        effect = score_diff(s["score_full"], s["score_base"])
        effect_r = score_diff(s["score_tier1"], s["score_tier64"])
        print(f"\n=== CUDA graphs {'on' if graphs == 'graphs' else 'off (enforce_eager)'} ===")
        print(f"  stock noise (same engine, batched vs alone): mean |dlp| {control['mean']:.4f} (adapter), "
              f"{control_base['mean']:.4f} (base); p99 {control['p99']:.3f} / {control_base['p99']:.3f}")
        print(f"  signal: stock matryoshka-0 vs base mean |dlp| {effect['mean']:.3f}; rank 1 vs rank 64 {effect_r['mean']:.3f}")
        print(f"  gate: decisive (top-1 lead >= {DECISIVE}) top-1 agreement 100%, mean <= {NOISE_FACTOR} x {noise:.4f}, "
              f"p99 <= {NOISE_FACTOR} x {noise99:.3f}")
        print(f"  {'check':<48}{'mean':>8}{'p99':>8}{'max':>8}{'decisive agree':>17}   greedy exact/tie/diverged")
        checks = [
            ("control, stock: adapter batched vs alone", "full", s, "solo_full", s),
            ("base: rankfuse engine vs stock", "base", r, "base", s),
            ("all requests at r_max=64 vs stock matryoshka-0", "tier64", r, "full", s),
            ("matryoshka_rank None vs stock matryoshka-0", "full", r, "full", s),
        ] + [(f"rank {t} vs stock matryoshka-0-r{t}", f"tier{t}", r, f"tier{t}", s) for t in TIERS] + [
            ("mixed-rank batch vs per-request runs (rankfuse)", "mixed", r, "solo", r),
            ("mixed-rank batch vs stock per-tier adapters", "mixed", r, None, s),
        ]
        for label, xa, A, yb, Bsrc in checks:
            xs_s, xs_g = A["score_" + xa], A["gen_" + xa]
            if yb is None:
                ys_s, ys_g = stock_at("score"), stock_at("gen")
            else:
                ys_s, ys_g = Bsrc["score_" + yb], Bsrc["gen_" + yb]
            st = score_diff(xs_s, ys_s)
            g = greedy_stats(xs_g, ys_g)
            ok = (st["decisive_agree"] == st["decisive"] and st["mean"] <= NOISE_FACTOR * noise
                  and st["p99"] <= NOISE_FACTOR * noise99)
            if label.startswith("control"):
                ok = True  # the yardstick itself
            all_ok &= ok
            print(f"  {label:<48}{st['mean']:>8.4f}{st['p99']:>8.3f}{st['max']:>8.3f}"
                  f"{st['decisive_agree']:>9}/{st['decisive']:<7}   {g['exact']:>2}/{g['tie']:>2}/{g['diverged']:>2}"
                  f"   {'ok' if ok else 'FAIL'}")
            results.append({"graphs": graphs, "label": label, "passed": ok, **st, "greedy": g})
    print("\n=== CUDA graphs vs eager, same engine (teacher-forced) ===")
    for engine in ("stock", "rankfuse"):
        a, b = os.path.join(d, f"{engine}_graphs.json"), os.path.join(d, f"{engine}_eager.json")
        if os.path.exists(a) and os.path.exists(b):
            ga, gb = json.load(open(a)), json.load(open(b))
            for case in ("full", "tier8", "base"):
                st = score_diff(ga["score_" + case], gb["score_" + case])
                g = greedy_stats(ga["gen_" + case], gb["gen_" + case])
                print(f"  {engine + ' ' + case + ': graphs vs eager':<48}{st['mean']:>8.4f}{st['p99']:>8.3f}{st['max']:>8.3f}"
                      f"{st['decisive_agree']:>9}/{st['decisive']:<7}   {g['exact']:>2}/{g['tie']:>2}/{g['diverged']:>2}")
                results.append({"graphs": "graphs-vs-eager", "label": f"{engine} {case}", **st, "greedy": g})
    with open(os.path.join(d, "correctness_summary.json"), "w") as f:
        json.dump({"passed": all_ok, "decisive_nats": DECISIVE, "noise_factor": NOISE_FACTOR, "tie_nats": TIE_NATS,
                   "checks": results}, f, indent=1)
    print(f"\n{'CORRECTNESS GATE PASSED' if all_ok else 'CORRECTNESS GATE FAILED'}")
    return all_ok


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        sys.exit(0 if compare(sys.argv[2]) else 1)
    run(sys.argv[2], sys.argv[3], sys.argv[4])
