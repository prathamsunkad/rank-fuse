#!/bin/bash
# Nsight Compute DRAM / L2 bytes of one multi-adapter call (bench/run_multi_sweep.py --ncu), as
# results/h200_2026-11-03_multi/scripts/ncu_multi.sh plus r_max. Run on an A100 node from the
# repository root (inside a100_job.sbatch MODE=ncu, which sets up the environment):
#   bash results/a100_2026-11-23_multi_v1/scripts/ncu_multi.sh RMAX impl shape num_adapters requests tokens_per_request
# Only the last of three calls is profiled; ncu defaults apply (caches flushed, clocks locked).
set -euo pipefail
module load cuda >/dev/null 2>&1 || true
RMAX=$1; shift
OUTDIR=results/a100_2026-11-23_multi_v1/source/ncu
mkdir -p "$OUTDIR"
NAME=$(IFS=-; echo "$*")-r$RMAX
REP="$OUTDIR/$NAME"
ncu --profile-from-start off --set full -o "$REP" -f python bench/run_multi_sweep.py --r-max "$RMAX" --ncu "$@"
if [ ! -f "${REP}.ncu-rep" ] && [ -f "${REP}.ncu-repz" ]; then
    zstd -q -d -f "${REP}.ncu-repz" -o "${REP}.ncu-rep"
fi
ncu --import "${REP}.ncu-rep" --page raw --csv --metrics \
gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors.sum,lts__t_sector_hit_rate.pct,\
launch__grid_size,launch__block_size,launch__registers_per_thread > "${REP}.csv"
rm -f "${REP}.ncu-rep" "${REP}.ncu-repz"
echo "wrote ${REP}.csv"
