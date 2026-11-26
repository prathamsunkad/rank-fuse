#!/bin/bash
# Single-adapter end-to-end sweep (Step 3): base, stock vLLM (padded per-tier, full r_max,
# fixed rank r) and vLLM + RankFuse, on one GPU. Run inside an allocation:
#
#   salloc --no-shell -J rf-e2e -p ice-gpu -N1 --cpus-per-task=16 --gres=gpu:h200:1 --mem=128G -t 08:00:00
#   srun --jobid=<id> --overlap -N1 -n1 bash bench/e2e/jobs/single_adapter.sh <q05|q7> <out> [systems...]
#
# Environment knobs: RATES ("2 4 8 inf"), FIXED_RANKS ("8 64 256"), N (requests per trace),
# CLIENT_ARGS (extra client.py flags), ENGINE_ARGS (vLLM flags shared by every system).
set -uo pipefail
WT=/storage/ice1/4/7/psunkad3/rankfuse/wt/e2e
cd "$WT"
source /storage/ice1/4/7/psunkad3/rankfuse/rf_paths.sh
source scripts/ice/env.sh
export PYTHONPATH=$WT
# Own caches: no sharing of compiled graphs / extensions with the other sessions' jobs.
export TORCH_EXTENSIONS_DIR=$RF_SCRATCH/cache/torch_extensions_e2e
export VLLM_CACHE_ROOT=$RF_SCRATCH/cache/vllm_e2e
python - <<'EOF' || exit 1
import rankfuse, sys
assert "/wt/e2e/" in rankfuse.__file__, rankfuse.__file__
from importlib.metadata import entry_points
eps = [e for e in entry_points(group="vllm.general_plugins") if e.name == "rankfuse"]
assert eps, "rankfuse vllm.general_plugins entry point not found"
print("rankfuse from", rankfuse.__file__, "| entry point", eps[0].value)
EOF

KEY=$1; OUT=$2; shift 2
case $KEY in
  q05) MODEL=Qwen/Qwen2.5-0.5B-Instruct; AD=$RF_ADAPTERS/synthetic_r256/qwen2.5-0.5b ;;
  q7)  MODEL=Qwen/Qwen2.5-7B-Instruct;   AD=$RF_ADAPTERS/synthetic_r256/qwen2.5-7b ;;
  *) echo "unknown model key $KEY"; exit 1 ;;
esac
SYSTEMS=${*:-"base stock-pad stock-full rankfuse rankfuse-skip stock-r8 stock-r64"}
RATES=${RATES:-"2 4 8 inf"}
FIXED_RANKS=${FIXED_RANKS:-"8 64 256"}
N=${N:-3000}
ENGINE_ARGS=${ENGINE_ARGS:-"--max-model-len 4096 --max-num-seqs 256 --gpu-memory-utilization 0.85 --seed 0"}
CLIENT_ARGS=${CLIENT_ARGS:-"--max-duration 120 --min-requests 300 --max-requests 2000 --warmup 96"}
TR=$OUT/traces
mkdir -p "$TR"

# Traces (unit rate; the client scales arrivals by the rate). Same seed -> same requests.
LEN="--prompt-len lognormal:256:0.8:16:2048 --output-len lognormal:128:0.7:8:512"
gen() { [ -f "$TR/$1.jsonl" ] || python bench/e2e/gen_trace.py --out "$TR/$1.jsonl" --n "$N" --seed 0 $LEN "${@:2}"; }
gen uniform --arrival poisson --rank uniform:8,16,32,64,128,256
gen skewed  --arrival poisson --rank skewed:1:8,16,32,64,128,256
for r in $FIXED_RANKS; do gen fixed$r --arrival poisson --rank fixed:$r; done

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$OUT/gpu.txt"
git -C "$WT" rev-parse HEAD > "$OUT/commit.txt"; git -C "$WT" status --short >> "$OUT/commit.txt"
python -c "import vllm, torch, triton; print('vllm', vllm.__version__, 'torch', torch.__version__, 'triton', triton.__version__)" > "$OUT/versions.txt"

fixed_traces() { for r in $FIXED_RANKS; do printf -- "--trace %s " "$TR/fixed$r.jsonl"; done; }
for sys in $SYSTEMS; do
  case $sys in
    base|stock-full) TRACES="--trace $TR/uniform.jsonl" ;;     # rank ignored by these systems
    stock-pad)       TRACES="--trace $TR/uniform.jsonl --trace $TR/skewed.jsonl" ;;
    rankfuse*)       TRACES="--trace $TR/uniform.jsonl --trace $TR/skewed.jsonl $(fixed_traces)" ;;
    stock-r*)        TRACES="--trace $TR/fixed${sys#stock-r}.jsonl" ;;
  esac
  echo "=== $sys $(date +%T)"
  python bench/e2e/run_system.py --system "$sys" --model "$MODEL" --adapters-dir "$AD" --r-max 256 \
      $TRACES --rates $RATES --out "$OUT" --engine-args "$ENGINE_ARGS" --client-args "$CLIENT_ARGS" \
      2>&1 | tee -a "$OUT/driver.log" || echo "FAILED: $sys" | tee -a "$OUT/driver.log"
done
python bench/e2e/summarize.py "$OUT" > "$OUT/summary.md" 2>&1 || true
