#!/bin/bash
# Nsight Compute DRAM / L2 bytes of one multi-adapter call (bench/run_multi_sweep.py --ncu). Run on
# an H200 node from the repository root:
#   bash results/h200_2026-11-03_multi/scripts/ncu_multi.sh rf1_ma 4096x4096 16 32 1
# Arguments: impl shape num_adapters requests tokens_per_request [dist popularity pages].
# Only the last of three calls is profiled; ncu defaults apply (caches flushed, clocks locked).
set -euo pipefail
source scripts/ice/env.sh
module load cuda >/dev/null 2>&1 || true
ncu --version | tail -1
OUTDIR=results/h200_2026-11-03_multi/source/ncu
mkdir -p "$OUTDIR"
NAME=$(IFS=-; echo "$*")
REP="$OUTDIR/$NAME"
ncu --profile-from-start off --set full -o "$REP" -f python bench/run_multi_sweep.py --ncu "$@"
if [ ! -f "${REP}.ncu-rep" ] && [ -f "${REP}.ncu-repz" ]; then
    zstd -q -d -f "${REP}.ncu-repz" -o "${REP}.ncu-rep"
fi
ncu --import "${REP}.ncu-rep" --page raw --csv --metrics \
gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors.sum,lts__t_sector_hit_rate.pct,\
launch__grid_size,launch__block_size,launch__registers_per_thread > "${REP}.csv"
rm -f "${REP}.ncu-rep" "${REP}.ncu-repz"
echo "wrote ${REP}.csv"
