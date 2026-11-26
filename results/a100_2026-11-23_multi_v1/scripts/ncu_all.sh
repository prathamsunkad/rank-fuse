#!/bin/bash
# ncu DRAM/L2 bytes on the 4096 x 4096 layer, 16 adapters, 32 and 128 decode tokens (uniform
# popularity and ranks), r_max 64 / 128 / 256 (as results/h200_2026-11-03_multi/scripts/ncu_all.sh).
set -uo pipefail
module load cuda >/dev/null 2>&1 || true
ncu --version | tail -1
for R in 64 128 256; do
  for T in 32 128; do
    for impl in rf1_ma rf2_ma_splitk rf2_ma_bucketed b3_multi b4_multi_copies b4_multi_shared; do
      bash results/a100_2026-11-23_multi_v1/scripts/ncu_multi.sh "$R" "$impl" 4096x4096 16 "$T" 1 || echo "FAILED $impl r$R T$T"
    done
  done
done
