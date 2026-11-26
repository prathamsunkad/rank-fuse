"""`vllm serve` with the RankFuse front end installed (per-request rank over HTTP).

    RANKFUSE_VLLM=1 python bench/e2e/serve_vllm.py serve <model> --enable-lora ...   # RankFuse
    RANKFUSE_VLLM=0 python bench/e2e/serve_vllm.py serve <model> ...                 # stock vLLM

Same arguments as the `vllm` CLI. With RANKFUSE_VLLM=1 it calls
rankfuse.integration.frontend.install() in this (API server) process first; the engine-core and
worker patches come from the existing `vllm.general_plugins` entry point as usual. With
RANKFUSE_VLLM unset or 0 it is exactly `vllm ...`, so stock baselines go through the same launcher.
Run with the default single API server (no --api-server-count > 1): the patches live in this
process only.
"""

import sys


def main() -> None:
    from rankfuse.integration import frontend

    installed = frontend.install()
    print(f"[serve_vllm] RankFuse front end {'installed' if installed else 'off (stock vLLM)'}", flush=True)
    from vllm.entrypoints.cli.main import main as vllm_main

    sys.argv = ["vllm"] + sys.argv[1:]
    vllm_main()


if __name__ == "__main__":
    main()
