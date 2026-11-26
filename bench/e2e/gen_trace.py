"""Workload traces for the end-to-end serving benchmark: one JSONL file every system replays.

    python bench/e2e/gen_trace.py --out traces/mix.jsonl --n 2000 --seed 0 \
        --arrival poisson --prompt-len lognormal:256:0.8:16:2048 --output-len lognormal:128:0.7:8:512 \
        --adapters 1 --zipf 1.0 --rank uniform:8,16,32,64,128,256

File format: line 1 is a header {"kind": "rankfuse-e2e-trace", "version": 1, "args": {...},
"stats": {...}}; every other line is one request:

    {"i": 0, "t": 0.734, "adapter": 0, "rank": 32, "prompt_len": 211, "output_len": 97,
     "prompt": [token ids]}

`t` is the arrival time in seconds at a mean rate of ONE request per second ("unit-rate trace").
The client replays it at rate R by sending request i at t_i / R, so every point of a rate sweep
replays the same requests, the same lengths, adapters and ranks, and the same arrival pattern
(burstiness is scale-free: Poisson stays Poisson, a gamma process keeps its CV, an on/off process
keeps its duty cycle). Prompts are explicit token ids, so prompt lengths are exact on every
system and no tokenizer round trip can change them.

Arrivals (--arrival)
    poisson          exponential gaps (CV 1)
    gamma:<cv>       gamma-distributed gaps with coefficient of variation cv (cv > 1 is bursty)
    onoff:<on>:<off> Poisson during ON periods, nothing during OFF; period lengths exponential
                     with means <on> and <off> seconds (at unit rate; they scale with 1/R too).
                     The ON rate is (on + off) / on so the long-run mean stays 1.
    burst            every request at t = 0 (a closed batch: max throughput)

Lengths (--prompt-len, --output-len)
    fixed:<n>
    uniform:<lo>:<hi>
    lognormal:<median>:<sigma>:<lo>:<hi>     clipped to [lo, hi]
    sharegpt:<path.json>                     (prompt and output lengths from a LOCAL ShareGPT
                                             json, tokenized with --tokenizer; no download)

Adapters (--adapters N --zipf a): adapter k in 0..N-1 with P(k) proportional to 1/(k+1)^a
(a = 0 is uniform). Adapter ids are indices; each backend maps them to its own names.

Rank (--rank)
    fixed:<r>
    uniform:<r1,r2,...>              uniform over the listed ranks
    skewed:<alpha>:<r1,r2,...>       P(r) proportional to (r / r_min)^-alpha: skewed toward low
                                     ranks (alpha = 1: 8 -> 0.51, 16 -> 0.25, ..., 256 -> 0.016)
    weights:<r1>=<w1>,<r2>=<w2>,...  explicit
    none                             no rank (rank: null): the server decides (policy / full)

--shared-prefix <n> gives every request of an adapter the same first n prompt tokens (a system
prompt), for prefix-cache experiments; 0 (default) makes prompts independent random tokens.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

VOCAB_LO, VOCAB_HI = 1000, 30000   # random prompt token ids (as bench/vllm_step_time.py)


# ------------------------------------------------------------------------------------- arrivals

def arrivals(spec: str, n: int, rng: np.random.Generator) -> np.ndarray:
    kind, *p = spec.split(":")
    if kind == "burst":
        return np.zeros(n)
    if kind == "poisson":
        gaps = rng.exponential(1.0, n)
    elif kind == "gamma":
        cv = float(p[0])
        shape = 1.0 / cv**2
        gaps = rng.gamma(shape, 1.0 / shape, n)          # mean 1, CV cv
    elif kind == "onoff":
        on, off = float(p[0]), float(p[1])
        rate_on = (on + off) / on
        t, out, in_on, left = 0.0, [], True, rng.exponential(on)
        while len(out) < n:
            if in_on:
                g = rng.exponential(1.0 / rate_on)
                if g <= left:
                    left -= g
                    t += g
                    out.append(t)
                    continue
                t += left
                in_on, left = False, rng.exponential(off)
            else:
                t += left
                in_on, left = True, rng.exponential(on)
        t = np.asarray(out)
        return t - t[0]
    else:
        raise ValueError(f"unknown arrival process {spec!r}")
    t = np.cumsum(gaps)
    return t - t[0]


# --------------------------------------------------------------------------------------- lengths

def lengths(spec: str, n: int, rng: np.random.Generator) -> np.ndarray:
    kind, *p = spec.split(":")
    if kind == "fixed":
        return np.full(n, int(p[0]))
    if kind == "uniform":
        return rng.integers(int(p[0]), int(p[1]) + 1, n)
    if kind == "lognormal":
        median, sigma, lo, hi = float(p[0]), float(p[1]), int(p[2]), int(p[3])
        x = np.exp(rng.normal(np.log(median), sigma, n))
        return np.clip(np.round(x), lo, hi).astype(int)
    raise ValueError(f"unknown length distribution {spec!r}")


def sharegpt_lengths(path: str, tokenizer: str, n: int, rng: np.random.Generator,
                     max_prompt: int, max_output: int) -> tuple[np.ndarray, np.ndarray]:
    """(prompt, output) token lengths of the first human/gpt turn of LOCAL ShareGPT conversations."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer)
    data = json.loads(Path(path).read_text())
    rng.shuffle(data)
    pl, ol = [], []
    for conv in data:
        turns = conv.get("conversations") or conv.get("items") or []
        if len(turns) < 2:
            continue
        p = len(tok(turns[0]["value"]).input_ids)
        o = len(tok(turns[1]["value"]).input_ids)
        if 4 <= p <= max_prompt and 4 <= o <= max_output:
            pl.append(p)
            ol.append(o)
        if len(pl) == n:
            break
    if len(pl) < n:
        raise ValueError(f"only {len(pl)} usable ShareGPT conversations, need {n}")
    return np.asarray(pl), np.asarray(ol)


# ----------------------------------------------------------------------------- adapters & ranks

def zipf_probs(n: int, a: float) -> np.ndarray:
    w = 1.0 / np.arange(1, n + 1) ** a
    return w / w.sum()


def rank_dist(spec: str) -> tuple[list[int | None], np.ndarray]:
    kind, *p = spec.split(":")
    if kind == "none":
        return [None], np.ones(1)
    if kind == "fixed":
        return [int(p[0])], np.ones(1)
    if kind == "uniform":
        ranks = [int(r) for r in p[0].split(",")]
        return ranks, np.full(len(ranks), 1.0 / len(ranks))
    if kind == "skewed":
        alpha, ranks = float(p[0]), sorted(int(r) for r in p[1].split(","))
        w = (np.asarray(ranks, dtype=float) / ranks[0]) ** -alpha
        return ranks, w / w.sum()
    if kind == "weights":
        pairs = [x.split("=") for x in p[0].split(",")]
        ranks, w = [int(a) for a, _ in pairs], np.asarray([float(b) for _, b in pairs])
        return ranks, w / w.sum()
    raise ValueError(f"unknown rank distribution {spec!r}")


# ------------------------------------------------------------------------------------- generate

def generate(args) -> tuple[dict, list[dict]]:
    rng = np.random.default_rng(args.seed)
    n = args.n
    t = arrivals(args.arrival, n, rng)
    if args.prompt_len.startswith("sharegpt:"):
        pl, ol = sharegpt_lengths(args.prompt_len.split(":", 1)[1], args.tokenizer, n, rng,
                                  args.max_prompt, args.max_output)
        if not args.output_len.startswith("sharegpt"):
            ol = lengths(args.output_len, n, rng)
    else:
        pl = lengths(args.prompt_len, n, rng)
        ol = lengths(args.output_len, n, rng)
    adapters = rng.choice(args.adapters, size=n, p=zipf_probs(args.adapters, args.zipf))
    ranks, rp = rank_dist(args.rank)
    rank_idx = rng.choice(len(ranks), size=n, p=rp)
    prefixes = {}
    reqs = []
    for i in range(n):
        a, L = int(adapters[i]), int(pl[i])
        body = rng.integers(VOCAB_LO, VOCAB_HI, L).tolist()
        if args.shared_prefix:
            if a not in prefixes:
                prefixes[a] = np.random.default_rng((args.seed, 7919, a)).integers(
                    VOCAB_LO, VOCAB_HI, args.shared_prefix).tolist()
            k = min(args.shared_prefix, L)
            body[:k] = prefixes[a][:k]
        reqs.append(dict(i=i, t=round(float(t[i]), 6), adapter=a, rank=ranks[rank_idx[i]],
                         prompt_len=L, output_len=int(ol[i]), prompt=body))
    stats = dict(
        n=n, duration_unit_rate=float(t[-1]),
        prompt_len=dict(mean=float(pl.mean()), p50=float(np.median(pl)), p99=float(np.percentile(pl, 99))),
        output_len=dict(mean=float(ol.mean()), p50=float(np.median(ol)), p99=float(np.percentile(ol, 99))),
        rank_counts={str(r): int((np.asarray([q["rank"] for q in reqs], dtype=object) == r).sum()) for r in ranks},
        adapter_counts={str(k): int((adapters == k).sum()) for k in range(min(args.adapters, 50))},
    )
    header = dict(kind="rankfuse-e2e-trace", version=1, args=vars(args), stats=stats)
    return header, reqs


def write(path: str | Path, header: dict, reqs: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps(header) + "\n")
        for r in reqs:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def load(path: str | Path) -> tuple[dict, list[dict]]:
    with open(path) as f:
        header = json.loads(f.readline())
        assert header.get("kind") == "rankfuse-e2e-trace", f"{path} is not a RankFuse e2e trace"
        return header, [json.loads(line) for line in f if line.strip()]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arrival", default="poisson")
    p.add_argument("--prompt-len", default="lognormal:256:0.8:16:2048")
    p.add_argument("--output-len", default="lognormal:128:0.7:8:512")
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct", help="for sharegpt lengths only")
    p.add_argument("--max-prompt", type=int, default=2048)
    p.add_argument("--max-output", type=int, default=1024)
    p.add_argument("--adapters", type=int, default=1)
    p.add_argument("--zipf", type=float, default=1.0)
    p.add_argument("--rank", default="uniform:8,16,32,64,128,256")
    p.add_argument("--shared-prefix", type=int, default=0)
    return p


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    header, reqs = generate(args)
    write(args.out, header, reqs)
    print(json.dumps(header["stats"]), file=sys.stderr)


if __name__ == "__main__":
    main()
