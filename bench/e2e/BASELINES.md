# SGLang and S-LoRA baselines: what each would need

Nothing in this file is installed. Versions are from memory as of mid-2026 and must be checked
against each project's install docs before anything is installed. Everything goes under
`~/scratch/rankfuse`.

## SGLang

* **Environment.** Needs a separate conda env, `env-sglang/`. SGLang pins its own torch,
  `sgl-kernel` and flashinfer builds, which won't coexist with our torch 2.13 / cu130 / vLLM 0.32
  env.
  * Install: `pip install "sglang[all]"` at a pinned recent release (0.5.x-series or newer).
  * That release targets a CUDA 12.x / 13 wheel set; use the one matching the node driver.
  * Size is about 10 GB with caches.
* **Launch.**

  ```
  python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --enable-lora \
    --lora-paths matryoshka-0-r8=<dir> ... matryoshka-0-r256=<dir> --max-lora-rank 256 \
    --max-loras-per-batch 6 --lora-backend <triton|csgmv> --port <p>
  ```

  Check the flag names on the installed version: `--lora-target-modules all` and
  `--max-loaded-loras` have changed across releases.
* **Our adapters.** They are PEFT directories, so they load as they are. The per-tier copies
  (`matryoshka-0-r{t}`) give mixed ranks, the same as stock vLLM.
  * Like vLLM, SGLang pads buffers to `max_lora_rank`, unless its newer unified/paged LoRA
    memory is used. Record which mode ran.
* **Client.** Use `client.py --rank-mode tier-copies` against `/v1/completions` (the
  OpenAI-compatible endpoint, which accepts token-id prompts and `ignore_eos`). Check that
  `stream_options.continuous_usage_stats` is honored. If it isn't, the client falls back to
  counting one token per chunk.
* **Metrics.** Add `--enable-metrics` for `/metrics`. `client.py` already maps
  `sglang:token_usage`, `sglang:num_running_reqs` and `sglang:num_queue_reqs`.
* **What we can't do.** SGLang can't serve prefix ranks from one copy without our kernels.
  Porting RankFuse into SGLang (part B generality) is separate work.

## S-LoRA (original research system, LightLLM-based)

* **Code.** github.com/S-LoRA/S-LoRA. It has been unmaintained since about 2024.
* **Environment.** Python 3.9–3.10, torch 2.0.x, triton 2.1, CUDA 11.8, plus its custom CUDA
  kernels (`setup.py install`, built for sm_80).
  * It needs its own env, `env-slora/`, with CUDA 11.8 from conda (`cudatoolkit-dev` or the
    `nvidia/label/cuda-11.8` channel).
  * Before investing in H100/H200 (sm_90), check whether the kernels compile for sm_90 at all.
    The A100s would be the natural GPU, but they are reserved, so this has to be decided.
* **Models.** It supports Llama-architecture models only. Qwen2 is not supported (no q/k/v
  bias, a different config). The S-LoRA comparison therefore needs a Llama model, for example
  Llama-3.1-8B-Instruct (gated: needs an HF token on the user's account) or Llama-2-7B.
  * Our harness and adapters already support `llama3-8b` (`scripts/make_adapters.py
    --model llama3-8b`).
* **Our adapters.** It loads PEFT LoRA directories (`--lora-dirs`). Its unified memory pool
  stores each adapter unpadded at its own rank, so the per-tier copies give mixed ranks without
  padding (the "S-LoRA-like" row of `memory.py fit`).
  * It may support only some target modules: check that it handles LoRA on all of
    q/k/v/o/gate/up/down. It may handle q/k/v/o only, in which case our adapters need
    `--targets q_proj k_proj v_proj o_proj` for that comparison, and so do the matching vLLM
    runs.
* **API.** Not OpenAI. It uses `POST /generate_stream` with
  `{"model_dir", "lora_dir", "inputs": <text>, "parameters": {"max_new_tokens", "ignore_eos", ...}}`.
  This needs a new `SLoRABackend` in `client.py` (about 40 lines).
  * Prompts are text, so random-token prompts need a decode/re-encode round trip. Record the
    real prompt length.
* **Effort.** The env is old and fragile. Plan about a day and keep a fallback: the
  "S-LoRA-like" memory row, and vLLM with `max_loras` large and `max_lora_rank` per tier, as a
  stand-in.

## What the main session needs to ask the user

1. May we create `env-sglang/` (about 10 GB) and `env-slora/` (about 6 GB)? Scratch is at about
   250/300 GB after the 7B download.
2. S-LoRA only runs Llama-architecture models. Use Llama-3.1-8B (gated, needs an HF token on the
   user's account) or Llama-2-7B? And may we run S-LoRA on an A100 if its kernels don't build
   for sm_90?
3. Which SGLang release should be pinned, to match the paper's submission date?
