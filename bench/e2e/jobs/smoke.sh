#!/bin/bash
# Smoke test inside an allocation: (re)build the 7B r256 adapters if incomplete (the login node
# OOM-kills make_adapters for 7B), then a short 0.5B run of every system kind.
#   srun --jobid=<id> --overlap -N1 -n1 bash bench/e2e/jobs/smoke.sh <out>
set -uo pipefail
WT=/storage/ice1/4/7/psunkad3/rankfuse/wt/e2e
cd "$WT"
source /storage/ice1/4/7/psunkad3/rankfuse/rf_paths.sh
source scripts/ice/env.sh
export PYTHONPATH=$WT
OUT=$1; mkdir -p "$OUT"
A7=$RF_ADAPTERS/synthetic_r256/qwen2.5-7b
if [ ! -f "$A7/matryoshka-0-r256/adapter_config.json" ] || [ ! -f "$A7/matryoshka-0-r128/adapter_config.json" ]; then
  echo "== building 7B adapters $(date +%T)"
  python scripts/make_adapters.py --model qwen2.5-7b --r-max 256 --tiers 8 16 32 64 128 256 --out "$A7" > "$OUT/make_adapters_7b.log" 2>&1
  echo "exit $?"; tail -2 "$OUT/make_adapters_7b.log"; ls "$A7"
fi
export N=400 RATES="4 inf" FIXED_RANKS="8" \
  CLIENT_ARGS="--max-duration 30 --min-requests 100 --max-requests 200 --warmup 16"
bash bench/e2e/jobs/single_adapter.sh q05 "$OUT" ${SMOKE_SYSTEMS:-rankfuse-skip stock-pad rankfuse base stock-r8}
