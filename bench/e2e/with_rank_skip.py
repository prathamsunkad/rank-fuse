"""Run a script whose vLLM engines get the rank-skip worker extension (RANKFUSE_VLLM_KERNEL modes).

    RANKFUSE_VLLM=1 RANKFUSE_VLLM_KERNEL=rf2_index python bench/e2e/with_rank_skip.py \
        bench/vllm_correctness.py run rankfuse graphs out/rankfuse_graphs.json

Wraps vllm.LLM.__init__ to pass worker_extension_cls=rankfuse.integration.rank_skip.RankSkipWorkerExtension
(the existing offline scripts build LLM(...) without it), then runs the script as __main__.
"""

import runpy
import sys


def main():
    import vllm

    orig = vllm.LLM.__init__

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("worker_extension_cls", "rankfuse.integration.rank_skip.RankSkipWorkerExtension")
        orig(self, *args, **kwargs)

    vllm.LLM.__init__ = __init__
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
