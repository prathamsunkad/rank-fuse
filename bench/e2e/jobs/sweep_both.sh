#!/bin/bash
# Step 3 sweeps, both models, every system, all in ONE allocation (one GPU type per sweep):
#   srun --jobid=<id> --overlap -N1 -n1 bash bench/e2e/jobs/sweep_both.sh <results dir>
# Rates start with "inf" (max throughput), then rise until the client sees saturation
# (--stop-when-saturated: throughput < 0.8 x offered, > 5% failures, or < 20% within TTFT 2 s).
set -uo pipefail
WT=/storage/ice1/4/7/psunkad3/rankfuse/wt/e2e
OUT=$1; mkdir -p "$OUT"
export CLIENT_ARGS="--max-duration 60 --min-requests 300 --max-requests 2000 --warmup 128 --warmup-concurrency 32 --stop-when-saturated"
export N=3000 FIXED_RANKS="8 64 256"
SYS="base stock-pad stock-full rankfuse rankfuse-skip stock-r8 stock-r64"
RATES="inf 5 10 20 30 40 60 80 100" bash "$WT/bench/e2e/jobs/single_adapter.sh" q05 "$OUT/q05" $SYS
RATES="inf 1 2 3 4 6 8 12 16" bash "$WT/bench/e2e/jobs/single_adapter.sh" q7 "$OUT/q7" $SYS
