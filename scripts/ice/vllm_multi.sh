#!/bin/bash
# Multi-adapter RankFuse in vLLM 0.32: tests and correctness runs (one GPU; H100 / H200 / A100-40GB).
#   bash scripts/ice/vllm_multi.sh <out dir> <stage...>
# stages: unit  make  step1  step1rf2  layer  ranktime  paged  pagedsmall  prefix  compare
# Run inside an allocation:  srun --jobid=<id> --overlap -N1 -n1 bash scripts/ice/vllm_multi.sh ...
set -uo pipefail
WT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$WT"
source /storage/ice1/4/7/psunkad3/rankfuse/rf_paths.sh
source scripts/ice/env.sh
export PYTHONPATH=$WT
export TORCH_EXTENSIONS_DIR=$RF_SCRATCH/cache/torch_extensions_vllm_multi
export VLLM_CACHE_ROOT=$RF_SCRATCH/cache/vllm_multi
OUT=$1; shift
mkdir -p "$OUT"
python -c "import rankfuse, sys; f = rankfuse.__file__; print('rankfuse from', f); sys.exit(0 if '/wt/vllm-multi/' in f else 1)" || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run() {  # run <name> <env...> -- <args>
    local name=$1; shift
    echo "== $name $(date +%T)"
    env "$@" > "$OUT/$name.log" 2>&1
    local rc=$?
    tail -n 3 "$OUT/$name.log"
    [ $rc -eq 0 ] || echo "!! $name failed (rc $rc), see $OUT/$name.log"
}

RF="RANKFUSE_VLLM=1 RANKFUSE_MULTI=1"
for stage in "$@"; do
    case $stage in
    unit)
        run unit python -m pytest -x -q tests/test_vllm_multi.py tests/test_model_pages.py tests/test_paging_bridge.py tests/test_multi_graphs.py ;;
    make)
        run make python bench/vllm_multi_check.py make ;;
    step1)
        for g in graphs eager; do
            for b in all waves; do
                run stock_${g}_$b python bench/vllm_multi_check.py run stock $g $b "$OUT/stock_${g}_$b.json"
                run dense_${g}_$b $RF python bench/vllm_multi_check.py run dense $g $b "$OUT/dense_${g}_$b.json"
            done
        done ;;
    layer)
        run layer_dense $RF python bench/vllm_multi_layer_check.py "$OUT/layer_dense.out" ;;
    paged)  # pool holds everything: no eviction
        for g in graphs eager; do
            for b in all waves; do
                run paged_${g}_$b $RF RANKFUSE_PAGING=1 RANKFUSE_PAGE_FRAMES=200 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
                    python bench/vllm_multi_check.py run paged $g $b "$OUT/paged_${g}_$b.json"
            done
        done
        run layer_paged $RF RANKFUSE_PAGING=1 RANKFUSE_PAGE_FRAMES=200 python bench/vllm_multi_layer_check.py "$OUT/layer_paged.out" ;;
    pagedsmall)  # 20 frames for 12 adapters needing 1..16 pages each: eviction and reload under load
        for g in graphs eager; do
            for b in all waves; do
                run pagedsmall_${g}_$b $RF RANKFUSE_PAGING=1 RANKFUSE_PAGE_FRAMES=20 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
                    python bench/vllm_multi_check.py run paged $g $b "$OUT/pagedsmall_${g}_$b.json"
            done
        done
        run layer_pagedsmall $RF RANKFUSE_PAGING=1 RANKFUSE_PAGE_FRAMES=20 python bench/vllm_multi_layer_check.py "$OUT/layer_pagedsmall.out" ;;
    prefix)
        run prefix_dense $RF python bench/vllm_multi_prefix.py dense "$OUT/prefix_dense.json"
        run prefix_paged $RF RANKFUSE_PAGING=1 RANKFUSE_PAGE_FRAMES=40 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
            python bench/vllm_multi_prefix.py paged "$OUT/prefix_paged.json" ;;
    ranktime)
        run ranktime_05b python bench/vllm_multi_rank_time.py 0.5B 64 "$OUT/ranktime_05b_T64.json"
        run ranktime_7b python bench/vllm_multi_rank_time.py 7B 64 "$OUT/ranktime_7b_T64.json" ;;
    step1rf2)
        for g in graphs eager; do
            run dense_rf2_${g}_waves $RF RANKFUSE_MULTI_KERNEL=rf2_ma_splitk python bench/vllm_multi_check.py run dense $g waves "$OUT/densrf2_${g}_waves.json"
        done ;;
    compare)
        python bench/vllm_multi_check.py compare "$OUT" | tee "$OUT/compare.txt" ;;
    *) echo "unknown stage $stage" ;;
    esac
done
