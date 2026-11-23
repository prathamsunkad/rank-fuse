"""RankFuse Triton kernels.

Each kernel module registers its wrappers with `rankfuse.interface.register`, so
tests/test_correctness.py and bench/run_sweep.py pick them up automatically.

Shared split-K base. rankfuse_fused.py holds the two kernels every current RF1 and RF2 variant
launches (`splitk_pass`, section 8 of its docstring): a split-K shrink into an fp32 workspace, then
one expand that sums the partials, masks each token to its rank, scales and writes or adds the
output. Launch settings come from `_config`: the default rules, overridden per (width, token range)
by rankfuse/kernels/tuned_a100.py on the GPU it was tuned on (RANKFUSE_TUNED=0 turns that off).
The variants differ only in how tokens are read and how wide each group runs:

* rankfuse_fused.py        - rf1_fused: RF1, every token padded to the batch's width and masked
                             back to its own rank. The fusion control against B1.
* rankfuse_rf2_segment.py  - rf2_segment: per-block rank buckets over the batch's own order, chosen
                             at run time in the kernel. The main RF2, on rank-sorted batches.
                             rf2_segment_fw is its full-width control; rf2_probe_read and
                             rf2_probe_branch split the cost of the rank read and branch.
* rankfuse_rf2_index.py    - rf2_index: the same buckets over a sorted index list, tokens left in
                             batch order, and rf2_index_fw. For the sorted-index assumption.
* rankfuse_rf2_bucketed.py - rf2_bucketed: one launch per rank bucket through that bucket's index,
                             on concurrent streams, from a per-step CPU plan; rf2_bucketed_tuned
                             with per-bucket launch settings; rf2_bucketed_seq without streams.
* rankfuse_rf2_ranges.py   - rf2_ranges: on rank-sorted rows, one RF1 call per bucket over that
                             bucket's contiguous rows, at its own width and launch settings;
                             rf2_ranges_seq without streams.

Earlier kernels, kept as ablations (not on the shared base):

* rankfuse_fused.py  - rf1_single and rf2_fused: the one-kernel path (sections 3-4), each program
                       computing v once and reusing it across its output columns. rf1_single is the
                       fusion ablation for rf1_fused's split-K path.
* rankfuse_mask.py   - rf1_masked: the first RF1, one kernel whose programs each own a single
                       output tile, so every column tile recomputes the shrink.
* rankfuse_ragged.py - rf2_ragged: the first RF2, rf1_masked's layout over a rank-sorted order,
                       each token tile stopping after its own last 16-row rank band.

Next TODOs: profile actual DRAM/L2 traffic, and check whether a serving engine (vLLM) can supply
rank-sorted batches or a rank-sorted index. Rank sorting aims to improve prefix reuse; it cannot
guarantee that a weight is physically loaded exactly once because output tiling and caches
determine the real loads.

Triton is only imported when installed (it is unavailable on Windows/CPU-only machines), so
`import rankfuse` keeps working there and the kernels are simply absent from the registry.
"""

from importlib.util import find_spec

if find_spec("triton") is not None:
    from rankfuse.kernels import rankfuse_fused, rankfuse_mask, rankfuse_ragged  # noqa: F401
    from rankfuse.kernels import rankfuse_rf2_bucketed, rankfuse_rf2_index, rankfuse_rf2_segment  # noqa: F401
    from rankfuse.kernels import rankfuse_rf2_ranges  # noqa: F401

