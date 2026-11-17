"""Launch settings tuned by bench/tune_blocks.py on NVIDIA A100 80GB PCIe (commit afc6600).

Keys: (width, token range), ranges from rankfuse_fused.TOKEN_RANGES. Missing entries use the
default rules. Set RANKFUSE_TUNED=0 to ignore this table."""

# width 16, range 1 ([256, 1024] tokens): 1.082x over default
# width 16, range 2 ([2048] tokens): 1.022x over default
# width 16, range 3 ([8192] tokens): default kept (best 1.014x)
# width 32, range 1 ([256, 1024] tokens): 1.058x over default
# width 32, range 2 ([2048] tokens): 1.023x over default
# width 32, range 3 ([8192] tokens): default kept (best 1.001x)
# width 64, range 1 ([256, 1024] tokens): 1.112x over default
# width 64, range 2 ([2048] tokens): default kept (best 1.004x)
# width 64, range 3 ([8192] tokens): 1.022x over default
# width 128, range 1 ([256, 1024] tokens): 1.153x over default
# width 128, range 2 ([2048] tokens): 1.464x over default
# width 128, range 3 ([8192] tokens): 1.611x over default
# width 256, range 1 ([256, 1024] tokens): 1.263x over default
# width 256, range 2 ([2048] tokens): 1.460x over default
# width 256, range 3 ([8192] tokens): 2.144x over default

GPU = 'NVIDIA A100 80GB PCIe'
TUNED = {
    (16, 1): {'bm': 16, 'shrink_warps': 4, 'loop_expand': False, 'bn': 128, 'expand_warps': 4, 'bh': 128, 'stages': 4},
    (16, 2): {'bm': 32, 'shrink_warps': 4, 'loop_expand': False, 'bn': 128, 'expand_warps': 8, 'bh': 128, 'stages': 3},
    (32, 1): {'bm': 32, 'shrink_warps': 4, 'loop_expand': True, 'bn': 256, 'expand_warps': 8, 'bh': 128, 'stages': 3},
    (32, 2): {'bm': 32, 'shrink_warps': 4, 'loop_expand': False, 'bn': 128, 'expand_warps': 8, 'bh': 64, 'stages': 4},
    (64, 1): {'bm': 16, 'shrink_warps': 4, 'loop_expand': True, 'bn': 256, 'expand_warps': 8, 'bh': 128, 'stages': 4},
    (64, 3): {'bm': 32, 'shrink_warps': 8, 'loop_expand': False, 'bn': 256, 'expand_warps': 8, 'bh': 64, 'stages': 3},
    (128, 1): {'bm': 16, 'shrink_warps': 4, 'loop_expand': True, 'bn': 256, 'expand_warps': 8, 'bh': 128, 'stages': 3},
    (128, 2): {'bm': 64, 'shrink_warps': 8, 'loop_expand': True, 'bn': 256, 'expand_warps': 8, 'bh': 64, 'stages': 4},
    (128, 3): {'bm': 64, 'shrink_warps': 8, 'loop_expand': True, 'bn': 256, 'expand_warps': 8, 'bh': 64, 'stages': 3},
    (256, 1): {'bm': 16, 'shrink_warps': 4, 'loop_expand': True, 'bn': 256, 'expand_warps': 8, 'bh': 64, 'stages': 3},
    (256, 2): {'bm': 64, 'shrink_warps': 8, 'loop_expand': True, 'bn': 256, 'expand_warps': 8, 'bh': 64, 'stages': 2},
    (256, 3): {'bm': 64, 'shrink_warps': 8, 'loop_expand': True, 'bn': 256, 'expand_warps': 8, 'bh': 128, 'stages': 3},
}
