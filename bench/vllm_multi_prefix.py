"""Does vLLM's prefix-cache key have to include the Matryoshka rank? (RankFuse multi-adapter engine)

LoRA on q/k/v makes every K/V entry depend on the rank, so two requests on the same adapter at
different ranks must not share KV blocks. The plugin adds ("matryoshka_rank", r) to the LoRA extra
hash keys (vllm_plugin.py, `_gen_lora_extra_hash_keys`). This script checks, in one in-process
engine with prefix caching on (VLLM_ENABLE_V1_MULTIPROCESSING=0):

  with_rank     generate from prompt P on adapter a at rank LO, then at rank HI: the HI request gets no
                cached tokens and its greedy tokens / logprobs equal a cold HI run (cache reset first); a repeat of
                HI hits the cache and still matches.
  without_rank  the same sequence with the rank removed from the hash: HI reuses LO's blocks
                (cached tokens > 0) and its logprobs differ from the cold HI run. This is what the
                key prevents.

    RANKFUSE_VLLM=1 RANKFUSE_MULTI=1 [RANKFUSE_PAGING=1 ...] python bench/vllm_multi_prefix.py <engine> out.json
"""

import json
import os
import sys

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vllm_multi_check import ANSWER, TOPICS, adapter_dir, engine_kwargs, prompts  # noqa: E402

ADAPTER, LO, HI = 6, 8, 128  # adapter 6 has r_max 128


def main(engine, out_path):
    from vllm import LLM, SamplingParams
    from vllm.v1.core import kv_cache_utils

    from rankfuse.integration import MatryoshkaLoRARequest

    kw = engine_kwargs(engine, "eager")
    kw["enable_prefix_caching"] = True
    llm = LLM(**kw)
    text = prompts()[3] + "\n" + ANSWER.format(t=TOPICS[3])
    # Generation, not prompt_logprobs: vLLM skips prefix-cache reads for prompt_logprobs requests.
    sp = SamplingParams(temperature=0.0, max_tokens=16, logprobs=1)

    def score(rank):
        req = MatryoshkaLoRARequest(f"m{ADAPTER}", 1 + ADAPTER, adapter_dir(ADAPTER), matryoshka_rank=rank)
        o = llm.generate([text], sp, lora_request=[req], use_tqdm=False)[0]
        c = o.outputs[0]
        lps = [d[t].logprob for t, d in zip(c.token_ids, c.logprobs)]
        return (lps, list(c.token_ids)), int(o.num_cached_tokens or 0)

    def diff(a, b):
        """|d logprob| of the first generated token (read straight from the prompt's KV, cached or
        not) and the length of the common greedy prefix (later tokens can differ after a bf16
        near-tie, since a cache hit computes only the uncached tail)."""
        (la, ta), (lb, tb) = a, b
        same = 0
        while same < min(len(ta), len(tb)) and ta[same] == tb[same]:
            same += 1
        return {"first_token_dlp": abs(la[0] - lb[0]) if ta[0] == tb[0] else float("inf"), "common_prefix": same}

    res = {}
    for label in ("with_rank", "without_rank"):
        if label == "without_rank":
            orig = kv_cache_utils._gen_lora_extra_hash_keys

            def no_rank(request, orig=orig):
                keys = orig(request)
                if keys and len(keys[0]) >= 2 and keys[0][-2] == "matryoshka_rank":
                    keys = [keys[0][:-2]] + keys[1:]
                return keys

            kv_cache_utils._gen_lora_extra_hash_keys = no_rank
        llm.reset_prefix_cache()
        cold_hi, c0 = score(HI)
        llm.reset_prefix_cache()
        lo, c1 = score(LO)
        hi_after_lo, c2 = score(HI)
        hi_repeat, c3 = score(HI)
        res[label] = dict(cached_cold_hi=c0, cached_lo=c1, cached_hi_after_lo=c2, cached_hi_repeat=c3,
                          maxdiff_hi_after_lo_vs_cold=diff(hi_after_lo, cold_hi),
                          maxdiff_hi_repeat_vs_cold=diff(hi_repeat, cold_hi),
                          maxdiff_lo_vs_hi=diff(lo, cold_hi))
    w, wo = res["with_rank"], res["without_rank"]
    d = lambda r, k: r[k]["first_token_dlp"]  # noqa: E731
    res["ok"] = (w["cached_hi_after_lo"] == 0 and d(w, "maxdiff_hi_after_lo_vs_cold") < 1e-3
                 and w["maxdiff_hi_after_lo_vs_cold"]["common_prefix"] == 16
                 and w["cached_hi_repeat"] > 0 and d(w, "maxdiff_hi_repeat_vs_cold") < 0.1
                 and wo["cached_hi_after_lo"] > 0 and d(wo, "maxdiff_hi_after_lo_vs_cold") > 0.1)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    print("PREFIX CHECK", "PASS" if res["ok"] else "FAIL")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
