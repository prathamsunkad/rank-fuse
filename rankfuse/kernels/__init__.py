"""RankFuse Triton kernels.

* rankfuse_mask.py   - v1. Rank-aware shrink + expand; columns beyond each token's rank are
                       masked, and rank tiles beyond a token tile's max rank exit early.
* rankfuse_ragged.py - v2 (not started). Host builds a list of only the valid
                       (token tile, rank tile) work items and launches exactly those. Sorting
                       tokens by rank inside the batch makes tiles uniform.

Each kernel module registers its wrapper with `rankfuse.interface.register`, so
tests/test_correctness.py and bench/run_sweep.py pick it up automatically.

Triton is only imported when installed (it is unavailable on Windows/CPU-only machines), so
`import rankfuse` keeps working there and the kernels are simply absent from the registry.
"""

import importlib.util

if importlib.util.find_spec("triton") is not None:
    from rankfuse.kernels import rankfuse_mask  # noqa: F401
