#!/bin/bash
# ncu DRAM/L2 bytes at 32 and 128 decode tokens over 16 adapters (uniform popularity and ranks),
# 4096 x 4096 and Qwen2.5-0.5B down (4864 -> 896). Run from the repository root on an H200 node.
set -uo pipefail
for shape in 4096x4096 down; do
  for T in 32 128; do
    for impl in rf1_ma b3_multi b4_multi_copies b4_multi_shared; do
      bash results/h200_2026-11-03_multi/scripts/ncu_multi.sh "$impl" "$shape" 16 "$T" 1 || echo "FAILED $impl $shape $T"
    done
  done
done
