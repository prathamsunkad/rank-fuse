"""Launch settings tuned by bench/tune_blocks.py on NVIDIA A100 80GB PCIe (commit f18f092).

Keys: (width, token range), ranges from rankfuse_fused.TOKEN_RANGES. Missing entries use the
default rules. Set RANKFUSE_TUNED=0 to ignore this table."""

# width 16, range 1 ([256, 1024] tokens): 1.090x over default
# width 16, range 2 ([2048] tokens): 1.023x over default
# width 16, range 3 ([8192] tokens): default kept (best 1.016x)
# width 32, range 1 ([256, 1024] tokens): 1.144x over default
# width 32, range 2 ([2048] tokens): default kept (best 1.015x)
# width 32, range 3 ([8192] tokens): default kept (best 1.000x)
# width 64, range 1 ([256, 1024] tokens): 1.146x over default
# width 64, range 2 ([2048] tokens): default kept (best 1.006x)
# width 64, range 3 ([8192] tokens): 1.031x over default
# width 128, range 1 ([256, 1024] tokens): 1.066x over default
# width 128, range 2 ([2048] tokens): 1.267x over default
# width 128, range 3 ([8192] tokens): 1.429x over default
# width 256, range 1 ([256, 1024] tokens): 1.162x over default
# width 256, range 2 ([2048] tokens): 1.248x over default
# width 256, range 3 ([8192] tokens): 1.335x over default

GPU = 'NVIDIA A100 80GB PCIe'
TUNED = {
    (16, 1): {'bm': 16, 'shrink_warps': 4, 'bn': 256, 'expand_warps': 4, 'bh': 128, 'stages': 4},
    (16, 2): {'bm': 32, 'shrink_warps': 4, 'bn': 256, 'expand_warps': 8, 'bh': 128, 'stages': 3},
    (32, 1): {'bm': 16, 'shrink_warps': 4, 'bn': 256, 'expand_warps': 8, 'bh': 128, 'stages': 4},
    (64, 1): {'bm': 32, 'shrink_warps': 8, 'bn': 256, 'expand_warps': 8, 'bh': 128, 'stages': 3},
    (64, 3): {'bm': 64, 'shrink_warps': 8, 'bn': 256, 'expand_warps': 8, 'bh': 64, 'stages': 3},
    (128, 1): {'bm': 32, 'shrink_warps': 4, 'bn': 128, 'expand_warps': 4, 'bh': 64, 'stages': 4},
    (128, 2): {'bm': 64, 'shrink_warps': 8, 'bn': 128, 'expand_warps': 4, 'bh': 64, 'stages': 4},
    (128, 3): {'bm': 64, 'shrink_warps': 4, 'bn': 128, 'expand_warps': 4, 'bh': 64, 'stages': 3},
    (256, 1): {'bm': 16, 'shrink_warps': 4, 'bn': 256, 'expand_warps': 8, 'bh': 64, 'stages': 3},
    (256, 2): {'bm': 32, 'shrink_warps': 8, 'bn': 256, 'expand_warps': 8, 'bh': 64, 'stages': 3},
    (256, 3): {'bm': 64, 'shrink_warps': 8, 'bn': 256, 'expand_warps': 4, 'bh': 128, 'stages': 3},
}
