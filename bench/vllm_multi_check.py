"""Multi-adapter RankFuse inside vLLM 0.32 against stock vLLM (Qwen2.5-0.5B-Instruct).

Every engine scores the same list of TRIPLES (prompt, adapter, rank) and writes per-triple results;
`compare` checks them against stock vLLM.

    adapters   NUM_ADAPTERS Matryoshka adapters with different r_max (R_MAX below, up to 256),
               written by scripts/make_adapters.py into $RF_ADAPTERS/vllm-multi (see `make`).
    stock      RANKFUSE_VLLM unset. (adapter i, rank r) is the per-tier copy matryoshka-i-r{r}
               (full rank / r = r_max: matryoshka-i); rank 0: no adapter. Two runs: `stock`
               (all triples in one call) and `stock_ctl` (the same triples in shuffled waves),
               whose difference is the batch-composition noise of stock vLLM.
    rankfuse   RANKFUSE_VLLM=1 RANKFUSE_MULTI=1 (+ RANKFUSE_PAGING=1 for the paged engines): one
               MatryoshkaLoRARequest per adapter at matryoshka_rank r (None = full).

Per triple: teacher-forced scoring (prompt + fixed answer, prompt_logprobs=2: the logprob of every
given token and the top-2) and a greedy generation of GEN_TOKENS tokens.

    python bench/vllm_multi_check.py make                             # adapters (CPU)
    python bench/vllm_multi_check.py run <engine> <graphs|eager> <batching> out.json
        engine: stock | dense | paged (paged reads RANKFUSE_PAGE_FRAMES)
        batching: all (one call) | waves (shuffled waves of WAVE triples, forces adapter churn)
    python bench/vllm_multi_check.py compare <dir>

Gate, as bench/vllm_correctness.py: (1) top-1 agreement on every decisive position (reference
top-1 leads top-2 by >= 0.5 nats) and (2) mean and p99 |d logprob| <= 2 x the stock control's
(floor 0.01). Greedy sequences are reported as exact / near-tie / diverged next to the control.
"""

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vllm_correctness import ANSWER, TOPICS, greedy_seq, prompts, score_diff  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ROOT = os.path.join(os.environ.get("RF_ADAPTERS", "."), "vllm-multi")
R_MAX = [16, 32, 64, 64, 64, 64, 128, 128, 256, 256, 64, 32]  # adapter i's r_max
NUM_ADAPTERS = len(R_MAX)
TIERS = [8, 16, 32, 64, 128, 256]
GEN_TOKENS = 32
WAVE = 6
MAX_LORA_RANK = 256


def adapter_dir(i, r=None):
    base = os.path.join(ROOT, f"r{R_MAX[i]}", f"matryoshka-{i}")
    return base if r is None or r >= R_MAX[i] else f"{base}-r{r}"


def triples():
    """(prompt index, adapter, rank); rank None = full, 0 = no adapter."""
    rng = random.Random(1234)
    out = []
    for k in range(48):
        a = (k * 5) % NUM_ADAPTERS
        choices = [None, 0] + [t for t in TIERS if t <= R_MAX[a]]
        out.append((k % len(TOPICS), a, rng.choice(choices)))
    # every adapter at least once at full rank and once below its r_max
    for a in range(NUM_ADAPTERS):
        out.append(((a * 7) % len(TOPICS), a, None))
        out.append(((a * 11 + 3) % len(TOPICS), a, max(8, R_MAX[a] // 2)))
    return out


def make():
    import subprocess

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for i, r in enumerate(R_MAX):
        out = os.path.join(ROOT, f"r{r}")
        if os.path.exists(os.path.join(out, f"matryoshka-{i}", "adapter_config.json")):
            continue
        tiers = [t for t in TIERS if t < r] + [r]
        # --seed i with --num-adapters 1 writes matryoshka-0; rename to matryoshka-i
        tmp = os.path.join(out, f"_tmp{i}")
        subprocess.check_call([sys.executable, os.path.join(here, "scripts", "make_adapters.py"), "--model", "qwen2.5-0.5b",
                               "--r-max", str(r), "--tiers", *map(str, tiers), "--seed", str(100 + i),
                               "--init-scale", os.environ.get("INIT_SCALE", "0.1"), "--dtype", "bf16", "--out", tmp])
        for d in os.listdir(tmp):
            os.rename(os.path.join(tmp, d), os.path.join(out, d.replace("matryoshka-0", f"matryoshka-{i}")))
        os.rmdir(tmp)
    print("adapters in", ROOT)


def engine_kwargs(engine, graphs):
    kw = dict(model=MODEL, enable_lora=True, max_lora_rank=MAX_LORA_RANK, max_model_len=1024, max_num_seqs=64,
              max_num_batched_tokens=2048, gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.45")),
              enforce_eager=(graphs == "eager"), seed=0, dtype="bfloat16")
    if graphs == "graphs":
        kw["compilation_config"] = {"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 48, 64],
                                    "cudagraph_specialize_lora": False}
    if engine == "paged":
        # max_loras: distinct adapters per step (vLLM requires max_cpu_loras >= max_loras)
        ml = int(os.environ.get("MAX_LORAS", "16"))
        kw.update(max_loras=ml, max_cpu_loras=max(ml, int(os.environ.get("MAX_CPU_LORAS", "16"))))
    else:
        kw.update(max_loras=int(os.environ.get("MAX_LORAS", "6")), max_cpu_loras=16)
    return kw


def run(engine, graphs, batching, out_path):
    import faulthandler

    # Watchdog: dump every thread's stack (and keep going) if the run hangs, RF_WATCHDOG seconds.
    if os.environ.get("RF_WATCHDOG"):
        faulthandler.dump_traceback_later(int(os.environ["RF_WATCHDOG"]), repeat=True, file=sys.stderr)
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    want = {"stock": None, "dense": "0", "paged": "1"}[engine]
    assert (os.environ.get("RANKFUSE_VLLM") == "1") == (want is not None), "RANKFUSE_VLLM does not match the engine"
    if want is not None:
        assert os.environ.get("RANKFUSE_MULTI") == "1" and os.environ.get("RANKFUSE_PAGING", "0") == want
    llm = LLM(**engine_kwargs(engine, graphs))
    from rankfuse.integration import vllm_plugin

    assert vllm_plugin._DONE == (want is not None), "plugin state does not match the engine"
    import rankfuse

    T = triples()
    ps = prompts()
    texts = [p + "\n" + ANSWER.format(t=t) for p, t in zip(ps, TOPICS)]

    def req(a, r):
        if r == 0:
            return None
        if engine == "stock":
            rr = None if r is None or r >= R_MAX[a] else r
            name = f"m{a}" if rr is None else f"m{a}-r{rr}"
            lid = 1 + a if rr is None else 1000 + a * 1000 + rr
            return LoRARequest(name, lid, adapter_dir(a, rr))
        from rankfuse.integration import MatryoshkaLoRARequest

        return MatryoshkaLoRARequest(f"m{a}", 1 + a, adapter_dir(a), matryoshka_rank=r)

    gen_sp = SamplingParams(temperature=0.0, max_tokens=GEN_TOKENS, logprobs=5)
    score_sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=2)

    order = list(range(len(T)))
    if batching == "all":
        waves = [order]
    else:
        random.Random(7).shuffle(order)
        waves = [order[i:i + WAVE] for i in range(0, len(order), WAVE)]

    res = {"score": [None] * len(T), "gen": [None] * len(T)}
    t0 = time.time()
    for wave in waves:
        reqs = [req(T[k][1], T[k][2]) for k in wave]
        reqs_arg = None if all(r is None for r in reqs) else reqs
        outs = llm.generate([texts[T[k][0]] for k in wave], score_sp, lora_request=reqs_arg, use_tqdm=False)
        for k, o in zip(wave, outs):
            ids = o.prompt_token_ids
            rows = []
            for i, d in enumerate(o.prompt_logprobs):
                if d is None:
                    continue
                ranked = sorted(((t, lp.logprob) for t, lp in d.items()), key=lambda e: -e[1])
                rows.append([ids[i], d[ids[i]].logprob, ranked[0][0], ranked[0][1],
                             ranked[1][1] if len(ranked) > 1 else None])
            res["score"][k] = rows
        outs = llm.generate([ps[T[k][0]] for k in wave], gen_sp, lora_request=reqs_arg, use_tqdm=False)
        for k, o in zip(wave, outs):
            c = o.outputs[0]
            top = [sorted(((t, lp.logprob) for t, lp in d.items()), key=lambda e: -e[1])[:5] for d in c.logprobs]
            res["gen"][k] = {"tokens": list(c.token_ids), "top": top, "text": c.text}
    res["_meta"] = {"engine": engine, "graphs": graphs, "batching": batching, "seconds": time.time() - t0,
                    "rankfuse": rankfuse.__file__, "page_frames": os.environ.get("RANKFUSE_PAGE_FRAMES"),
                    "triples": T}
    res["_meta"]["paging"] = paging_stats(llm)
    with open(out_path, "w") as f:
        json.dump(res, f)
    print(f"wrote {out_path} ({time.time() - t0:.0f} s); paging {res['_meta']['paging']}", flush=True)


def paging_stats(llm):
    """Pager counters when the engine core runs in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0)."""
    try:
        from rankfuse.integration import paging, vllm_multi

        p = paging.current_pager()
        if p is None:
            return None
        s = p.stats
        return dict(frames=p.usable_frames, page_loads=s.page_loads, evictions=s.evictions, bytes=s.bytes_copied,
                    ensure_calls=s.ensure_calls, registered=len(p.registered()), sched=dict(vllm_multi.SCHED_STATS))
    except Exception as e:  # noqa: BLE001
        return repr(e)


def compare(d):
    runs = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            with open(os.path.join(d, f)) as fh:
                runs[f[:-5]] = json.load(fh)
    lines = []
    ok_all = True
    for g in ("graphs", "eager"):
        ref_name, ctl_name = f"stock_{g}_all", f"stock_{g}_waves"
        if ref_name not in runs:
            continue
        ref = runs[ref_name]
        ctl = score_diff(runs[ctl_name]["score"], ref["score"]) if ctl_name in runs else None
        if ctl:
            lines.append(f"[{g}] control {ctl_name} vs {ref_name}: mean {ctl['mean']:.4f} p99 {ctl['p99']:.4f} "
                         f"max {ctl['max']:.3f} decisive {ctl['decisive_agree']}/{ctl['decisive']}")
        floor_mean = max(0.01, ctl["mean"]) if ctl else 0.01
        floor_p99 = max(0.01, ctl["p99"]) if ctl else 0.05
        for name, run_ in runs.items():
            if name.startswith("stock") or f"_{g}_" not in name:
                continue
            sd = score_diff(run_["score"], ref["score"])
            ok = sd["decisive_agree"] == sd["decisive"] and sd["mean"] <= 2 * floor_mean and sd["p99"] <= 2 * floor_p99
            ok_all &= ok
            cnt = {"exact": 0, "tie": 0, "diverged": 0}
            for a, b in zip(run_["gen"], ref["gen"]):
                cnt[greedy_seq(a, b)[0]] += 1
            # per-rank-class breakdown: full / below r_max / no adapter
            T = ref["_meta"]["triples"]
            cls = {}
            for k, (p, a, r) in enumerate(T):
                c = "none" if r == 0 else ("full" if r is None or r >= R_MAX[a] else "low")
                cls.setdefault(c, []).append(k)
            per = {c: score_diff([run_["score"][k] for k in ks], [ref["score"][k] for k in ks])["mean"]
                   for c, ks in cls.items()}
            lines.append(f"[{g}] {name}: {'PASS' if ok else 'FAIL'} mean {sd['mean']:.4f} p99 {sd['p99']:.4f} "
                         f"max {sd['max']:.3f} decisive {sd['decisive_agree']}/{sd['decisive']}; greedy {cnt}; "
                         f"mean by class {json.dumps({k: round(v, 4) for k, v in per.items()})}; "
                         f"paging {run_['_meta'].get('paging')}")
        if ctl_name in runs:
            cnt = {"exact": 0, "tie": 0, "diverged": 0}
            for a, b in zip(runs[ctl_name]["gen"], ref["gen"]):
                cnt[greedy_seq(a, b)[0]] += 1
            lines.append(f"[{g}] control greedy {cnt}")
    lines.append("ALL PASS" if ok_all else "SOME FAIL")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "make":
        make()
    elif cmd == "run":
        run(*sys.argv[2:6])
    elif cmd == "compare":
        compare(sys.argv[2])
    else:
        raise SystemExit(__doc__)
