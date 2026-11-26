"""Start one serving system, replay traces against it at a sweep of rates, stop it.

    python bench/e2e/run_system.py --system rankfuse --model Qwen/Qwen2.5-0.5B-Instruct \
        --adapters-dir $RF_ADAPTERS/synthetic_r256/qwen2.5-0.5b --r-max 256 \
        --trace traces/q05_uniform.jsonl --trace traces/q05_skewed.jsonl \
        --rates 2 4 8 16 inf --out results/e2e_2026-11-2X/q05

Systems (all vLLM 0.32, one GPU, the same engine flags; see SYSTEMS below):

    base           no LoRA (enable_lora off): the ceiling
    stock-pad      stock vLLM, max_lora_rank = r_max, one registered adapter per rank tier
                   ("matryoshka-0-r{t}"), max_loras = number of tiers: mixed ranks the only way
                   stock vLLM can serve them, every slot padded to r_max       [system (a)]
    stock-full     stock vLLM, max_lora_rank = r_max, one r_max adapter, every request at r_max
                   (ignores the requested rank)                                [system (a')]
    stock-r<r>     stock vLLM, max_lora_rank = r, the rank-r copy only, every request at r: the
                   lower bound for a workload where everyone wants rank r      [system (b)]
    rankfuse       vLLM + RankFuse Stage 1 plugin (RANKFUSE_VLLM=1), one r_max adapter in one
                   slot, the rank from each request (vllm_xargs); RF1 kernel: every block padded
                   to the batch max, and to r_max under CUDA graphs            [system (c)]
    rankfuse-skip  the same, RANKFUSE_VLLM_KERNEL=rf2_index: rank-skipping split-K kernels over
                   a rank-sorted token order, graph-safe (rankfuse/integration/rank_skip.py)
                                                                               [system (c-skip)]
    rankfuse-seg   the same kernels over the batch order (rf2_segment), the control for the sort

The server is started through bench/e2e/serve_vllm.py on a free local port; its log goes to
<out>/<system>.server.log and the memory lines parsed from it to <out>/<system>.memory.json.
The client's per-rate files and summary.jsonl go to <out> (tag = <system>__<trace stem>).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import client as C  # noqa: E402
import memory as MEM  # noqa: E402


def system_spec(name: str, model: str, adapters_dir: str, r_max: int, tiers: list[int]) -> dict:
    """Server args, environment and client rank mode of a system."""
    ad = Path(adapters_dir)
    if name == "base":
        return dict(env={"RANKFUSE_VLLM": "0"}, args=[], rank_mode="base")
    if name == "stock-pad":
        mods = [f"matryoshka-0-r{t}={ad / f'matryoshka-0-r{t}'}" for t in tiers]
        return dict(env={"RANKFUSE_VLLM": "0"}, rank_mode="tier-copies",
                    args=["--enable-lora", "--max-lora-rank", str(r_max), "--max-loras", str(len(tiers)),
                          "--max-cpu-loras", str(len(tiers)), "--lora-modules", *mods])
    if name == "stock-full":
        return dict(env={"RANKFUSE_VLLM": "0"}, rank_mode="adapter",
                    args=["--enable-lora", "--max-lora-rank", str(r_max), "--max-loras", "1",
                          "--lora-modules", f"matryoshka-0={ad / 'matryoshka-0'}"])
    if name.startswith("stock-r"):
        r = int(name[len("stock-r"):])
        return dict(env={"RANKFUSE_VLLM": "0"}, rank_mode="adapter",
                    args=["--enable-lora", "--max-lora-rank", str(MEM.r_pad_vllm(r)), "--max-loras", "1",
                          "--lora-modules", f"matryoshka-0={ad / f'matryoshka-0-r{r}'}"])
    if name in ("rankfuse", "rankfuse-skip", "rankfuse-seg"):
        kernel = {"rankfuse": "rf1", "rankfuse-skip": "rf2_index", "rankfuse-seg": "rf2_segment"}[name]
        extra = [] if kernel == "rf1" else [
            "--worker-extension-cls", "rankfuse.integration.rank_skip.RankSkipWorkerExtension"]
        return dict(env={"RANKFUSE_VLLM": "1", "RANKFUSE_VLLM_KERNEL": kernel}, rank_mode="xargs",
                    args=["--enable-lora", "--max-lora-rank", str(r_max), "--max-loras", "1",
                          "--lora-modules", f"matryoshka-0={ad / 'matryoshka-0'}", *extra])
    raise ValueError(f"unknown system {name!r}")


RANK_CHECK_MIN = 0.003   # nats: smallest mean |d logprob| counted as "different"


async def _prompt_logprobs(session, base_url, model, prompt, xargs=None) -> list:
    """Teacher-forced logprob of every prompt token (prompt_logprobs; the first is None)."""
    body = dict(model=model, prompt=prompt, max_tokens=1, temperature=0.0, prompt_logprobs=1)
    if xargs is not None:
        body["vllm_xargs"] = xargs
    async with session.post(base_url + "/v1/completions", json=body) as resp:
        js = await resp.json()
    plp = js["choices"][0]["prompt_logprobs"]
    return [None if d is None else d[str(t)]["logprob"] for d, t in zip(plp, prompt)]


def _mad(xs, ys) -> float:
    v = [abs(x - y) for a, b in zip(xs, ys) for x, y in zip(a, b) if x is not None and y is not None]
    return sum(v) / len(v)


async def rank_check(base_url: str, rank_mode: str) -> dict:
    """Log-probability check that the adapter, and for RankFuse the requested rank, reach the model.

    Four 96-token random prompts are scored teacher-forced (prompt_logprobs) and compared by mean
    |d logprob| (nats):
      noise           the same request twice (rank 256, or the adapter)
      adapter_vs_base the whole adapter vs the base model (the adapter's own effect: the control)
      r8_vs_r256      RankFuse only: must exceed max(RANK_CHECK_MIN, 3 x noise)
      r256_vs_full    RankFuse only: rank r_max must equal the whole adapter (<= the same bound)
    For stock systems the result is reported, never fatal."""
    import random

    import aiohttp

    rng = random.Random(1234)
    prompts = [[rng.randrange(1000, 30000) for _ in range(96)] for _ in range(4)]
    async with aiohttp.ClientSession() as s:
        async def lps(model, xargs=None):
            return [await _prompt_logprobs(s, base_url, model, p, xargs) for p in prompts]
        base = await lps("base")
        if rank_mode == "xargs":
            r256 = await lps("matryoshka-0", {"matryoshka_rank": 256})
            r256b = await lps("matryoshka-0", {"matryoshka_rank": 256})
            r8 = await lps("matryoshka-0", {"matryoshka_rank": 8})
            full = await lps("matryoshka-0", {"matryoshka_rank": "full"})
            noise = _mad(r256, r256b)
            bound = max(RANK_CHECK_MIN, 3 * noise)
            out = dict(noise=noise, adapter_vs_base=_mad(full, base), r8_vs_base=_mad(r8, base),
                       r8_vs_r256=_mad(r8, r256), r256_vs_full=_mad(r256, full), bound=bound)
            out["ok"] = out["r8_vs_r256"] > bound and out["r256_vs_full"] <= bound
            return out
        name = "matryoshka-0" if rank_mode == "adapter" else "matryoshka-0-r256"
        ad = await lps(name)
        ad2 = await lps(name)
        return dict(noise=_mad(ad, ad2), adapter_vs_base=_mad(ad, base), ok=True)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(spec: dict, model: str, engine_args: list[str], port: int, log_path: Path) -> subprocess.Popen:
    cmd = [sys.executable, str(HERE / "serve_vllm.py"), "serve", model, "--served-model-name", "base",
           "--host", "127.0.0.1", "--port", str(port), *engine_args, *spec["args"]]
    env = os.environ | spec["env"]
    log = open(log_path, "w")
    log.write("# " + " ".join(cmd) + "\n# env: " + json.dumps(spec["env"]) + "\n")
    log.flush()
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def stop_server(proc: subprocess.Popen, timeout: float = 60) -> None:
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGINT)
    try:
        proc.wait(timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--adapters-dir", required=True)
    p.add_argument("--r-max", type=int, default=256)
    p.add_argument("--tiers", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    p.add_argument("--trace", action="append", default=[])
    p.add_argument("--check-only", action="store_true", help="start, run the rank check, stop")
    p.add_argument("--rates", type=float, nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--engine-args", default="--max-model-len 4096 --max-num-seqs 256 --gpu-memory-utilization 0.85 --seed 0",
                   help="vLLM flags shared by every system")
    p.add_argument("--startup-timeout", type=float, default=1800)
    p.add_argument("--client-args", default="", help="extra client.py flags, e.g. '--max-duration 90'")
    a = p.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = system_spec(a.system, a.model, a.adapters_dir, a.r_max, a.tiers)
    port = free_port()
    log_path = out / f"{a.system}.server.log"
    t0 = time.time()
    proc = start_server(spec, a.model, a.engine_args.split(), port, log_path)
    base_url = f"http://127.0.0.1:{port}"
    try:
        try:
            import asyncio
            asyncio.run(C.wait_ready(base_url, a.startup_timeout))
        except TimeoutError:
            raise SystemExit(f"{a.system}: server not ready, see {log_path}")
        if proc.poll() is not None:
            raise SystemExit(f"{a.system}: server exited, see {log_path}")
        startup = time.time() - t0
        mem = MEM.parse_vllm_log(log_path) | dict(startup_s=startup, system=a.system, engine_args=a.engine_args,
                                                    server_args=spec["args"], env=spec["env"])
        (out / f"{a.system}.memory.json").write_text(json.dumps(mem, indent=1))
        print(f"[{a.system}] ready in {startup:.0f} s on port {port}: {mem}", flush=True)
        if spec["env"]["RANKFUSE_VLLM"] == "1" and not (mem["rankfuse_plugin"] and mem["rankfuse_frontend"]):
            raise SystemExit(f"{a.system}: RankFuse plugin/front end not active, see {log_path}")
        if spec["env"].get("RANKFUSE_VLLM_KERNEL", "rf1") != "rf1" and not mem["rankfuse_rank_skip"]:
            raise SystemExit(f"{a.system}: rank-skip mode not active, see {log_path}")
        if spec["rank_mode"] != "base":
            mem["rank_check"] = asyncio.run(rank_check(base_url, spec["rank_mode"]))
            (out / f"{a.system}.memory.json").write_text(json.dumps(mem, indent=1))
            print(f"[{a.system}] rank check (mean |d logprob|, nats): {mem['rank_check']}", flush=True)
            if not mem["rank_check"]["ok"]:
                raise SystemExit(f"{a.system}: per-request rank has no effect over HTTP")
        if a.check_only:
            return
        for tr in a.trace:
            tag = f"{a.system}__{Path(tr).stem}"
            cargs = ["--base-url", base_url, "--trace", tr, "--rank-mode", spec["rank_mode"],
                     "--rates", *[str(r) for r in a.rates], "--out", str(out), "--tag", tag,
                     *a.client_args.split()]
            C.main(cargs)
    finally:
        stop_server(proc)
    print(f"[{a.system}] done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
