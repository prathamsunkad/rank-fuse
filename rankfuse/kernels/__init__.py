"""RankFuse Triton kernels.

* rankfuse_mask.py   - v1. Rank-aware shrink + expand; columns beyond each token's rank are
                       masked, and rank tiles beyond a token tile's max rank exit early.
* rankfuse_ragged.py - v2 (not started). Host builds a list of only the valid
                       (token tile, rank tile) work items and launches exactly those. Sorting
                       tokens by rank inside the batch makes tiles uniform.

Each kernel module registers its wrapper with `rankfuse.interface.register`, so
tests/test_correctness.py and bench/run_sweep.py pick it up automatically.

* rankfuse_mask.py   - RF1, the fusion control. Fuse shrink + mask + expand in Triton, loop to
                       the batch maximum rank, and mask columns >= each token's rank.
                       Keep v on chip. This is intentionally a fused padded/masked strategy;
                       compare B1 vs RF1 to measure fusion and intermediate-traffic savings.
* rankfuse_ragged.py - RF2, the rank-aware experiment. Sort token indices by rank, then let
                       each token tile loop only to its local maximum band. Each output element
                       is computed and written by one program, so programs do not race to
                       update the same result and need no atomic additions.
                       Compare RF1 vs RF2 to measure the benefit of skipping high-rank bands.

Next TODOs: accept a scheduler-prepared rank order so RF2 can reuse it across layers, autotune
tile sizes, and profile actual DRAM/L2 traffic. Rank sorting aims to improve prefix reuse; it
cannot guarantee that a weight is physically loaded exactly once because output tiling and
caches determine the real loads.


Triton is only imported when installed (it is unavailable on Windows/CPU-only machines), so
`import rankfuse` keeps working there and the kernels are simply absent from the registry.
"""

from importlib.util import find_spec

if find_spec("triton") is not None:
    from rankfuse.kernels import rankfuse_mask, rankfuse_ragged  # noqa: F401

