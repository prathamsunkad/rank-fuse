"""Per-step sort / prepare cost on the mixed-batching steps, next to the kernel-only sweeps.

    python bench/mixed_prepare_cost.py single bench/configs/rf2_mixed_a100_r128.yaml --out .../single-mixed-r128-prepare.csv
    python bench/mixed_prepare_cost.py multi bench/configs/multi_mixed_a100_r128.yaml --out .../multi-mixed-r128-prepare.csv

bench/run_mixed_sweep.py and bench/run_multi_mixed_sweep.py call this after their sweep (same config,
output `<out stem>-prepare.csv`). The kernel sweeps keep every implementation's per-step metadata
(`prepare`) and the rank sort outside the timed graph; this measures that per-step work so it can be
charged per step and amortized over a model's LoRA calls (analysis/mixed_tables.py).

Two clocks:
  wall_us  host wall time of the call with torch.cuda.synchronize() before and after: host work, any
           host<->device copies and syncs, and GPU work. An upper bound on what a serving engine pays
           (it could overlap host work with the previous step's GPU work). Median of `reps`.
  gpu_us   CUDA-graph GPU time (run_sweep.time_graph, no L2 flush) for pieces that run on the GPU
           only and can be captured. Median of the repetitions.

Single adapter, one row per (workload, item), on the workload's own batch (random or sorted order):
  prepare:<impl>           impl.prepare(x, lora, token_ranks) as the sweep calls it (B3 / B4 weight
                           copies and pools are cached on the adapter after the first call, so this is
                           per-step metadata only). For rf2_ranges this is RangePlan from GPU ranks
                           (one device->host copy).
  sort_requests_host       sorting the step's requests by rank on the host (Python sorted on the
                           request ranks): what `request_order: sorted` assumes the scheduler does
                           before building the batch (a request reorder, as vLLM's reorder_batch).
  rangeplan_host           RangePlan from host ranks (an engine has them on the host).
  sorted_path_host         sort_requests_host + rangeplan_host: rf2_ranges' per-step cost when the
                           engine reorders requests (the case the kernel sweep times).
  argsort_gpu              stable torch.argsort of the token ranks on the GPU (gpu_us), per step.
  permute_per_call_gpu     if the engine cannot reorder requests: gather x into rank order and add
                           the delta back in batch order (x.index_select + out.index_add_), GPU time
                           per LoRA call (h_in = h_out = the workload's), on top of the kernel.
Multi adapter, one row per (workload, item):
  prepare:<impl>           impl.prepare(x, W, ta, tr) for the RankFuse kernels, B1-multi and
                           B4-multi-shared (pool cached on W).
  step_meta:b3_multi       B3-multi's per-step part only: (adapter, rank) pair ids, token -> slot,
                           vLLM's LoRAKernelMeta.prepare_tensors and the fp32 buffer. Its prepare also
                           builds padded weight copies for every pair; vLLM keeps those in persistent
                           slots, so they are not charged per step.
  step_meta:b4_multi_copies  likewise: pair ids, token -> entry and the delta buffer, without
                           rebuilding the pool (S-LoRA keeps adapters in its memory pool).
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def wall_us(fn, reps: int = 20, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(times)


def gpu_us(fn) -> float | None:
    """CUDA-graph GPU time, or None if the call cannot be captured."""
    from run_sweep import time_graph

    try:
        return statistics.median(time_graph(fn, (), {}, 5, 10, 20, False))
    except Exception:
        torch.cuda.synchronize()
        return None


def single(cfg: dict, out: Path, reps: int) -> None:
    import rankfuse  # noqa: F401
    from rankfuse.baselines import slora_bgmv_wide
    from rankfuse.interface import REGISTRY
    from rankfuse.kernels.rankfuse_rf2_ranges import RangePlan
    from rankfuse.workloads import make_batch, make_lora, sample_request_ranks
    from run_sweep import expand_grid

    slora_bgmv_wide.enable()
    impls = list(dict.fromkeys(cfg["impls"]))
    wl = expand_grid(cfg["grid"], cfg.get("shapes"), cfg.get("batches"))
    gname = torch.cuda.get_device_name()
    rows = []
    for shape, w in wl:
        lora, batch = make_lora(w, "cuda"), make_batch(w, "cuda")
        x, tr = batch.x, batch.token_ranks
        tr_cpu = tr.cpu()
        req_ranks = sample_request_ranks(w, torch.Generator().manual_seed(w.seed + 1))
        base = dict(gpu=gname, shape=shape, **w.as_row())

        def add(item, wall=None, gpu=None):
            rows.append(dict(base, item=item, wall_us=wall if wall is not None else "", gpu_us=gpu if gpu is not None else ""))

        for name in impls:
            impl = REGISTRY[name]
            if impl.prepare is not None:
                add(f"prepare:{name}", wall=wall_us(lambda: impl.prepare(x, lora, tr), reps))
        h_sort = wall_us(lambda: sorted(range(len(req_ranks)), key=req_ranks.__getitem__), reps)
        h_plan = wall_us(lambda: RangePlan(tr_cpu), reps)
        add("sort_requests_host", wall=h_sort)
        add("rangeplan_host", wall=h_plan)
        add("sorted_path_host", wall=h_sort + h_plan)
        g = gpu_us(lambda: torch.argsort(tr, stable=True))
        add("argsort_gpu", gpu=g, wall=None if g is not None else wall_us(lambda: torch.argsort(tr, stable=True), reps))
        perm = torch.argsort(tr, stable=True)
        y = torch.empty(x.shape[0], w.h_out, dtype=x.dtype, device="cuda")
        out_buf = torch.zeros_like(y)
        add("permute_per_call_gpu", gpu=gpu_us(lambda: (torch.index_select(x, 0, perm), out_buf.index_add_(0, perm, y))))
        print(f"  prepare-cost {w.lengths or 'decode'} D={w.num_requests} r{w.r_max} {w.request_order} s{w.seed} "
              f"T={x.shape[0]}: sorted path {h_sort + h_plan:.0f} us host", flush=True)
    write(rows, out)


def multi(cfg: dict, out: Path, reps: int) -> None:
    from rankfuse.baselines import slora_bgmv as sb
    from rankfuse.baselines.vllm_ops_multi import Step
    from rankfuse.mixed_multi import make_batch
    from rankfuse.multi import MULTI_REGISTRY, load_all, make_weights
    from run_multi_mixed_sweep import expand, weights_key

    load_all(include_slora=True)
    impls = cfg["impls"]
    wl = sorted(expand(cfg["grid"]), key=weights_key)
    gname = torch.cuda.get_device_name()
    rows, wk, W = [], None, None

    def pairs_of(ta, tr):
        ad, rk = ta.long().cpu(), tr.long().cpu()
        live = (rk > 0) & (ad >= 0)
        uniq, inv = torch.unique(torch.stack([ad[live], rk[live]], 1), dim=0, return_inverse=True)
        return ad, live, uniq, inv

    for w in wl:
        if weights_key(w) != wk:
            W = None
            torch.cuda.empty_cache()
            W, wk = make_weights(w.weights_workload(), "cuda"), weights_key(w)
        x, ta, tr = make_batch(w)
        base = dict(gpu=gname, **w.as_row())

        def add(item, wall):
            rows.append(dict(base, item=item, wall_us=wall, gpu_us=""))

        for name in impls:
            impl = MULTI_REGISTRY[name]
            if impl.unsupported(W) or impl.prepare is None:
                continue
            if name == "b3_multi":
                def b3_step():
                    ad, live, uniq, inv = pairs_of(ta, tr)
                    slot = torch.full((x.shape[0],), -1, dtype=torch.int32)
                    slot[live] = inv.to(torch.int32)
                    return Step(uniq.shape[0], slot.to(x.device), W.n_slices, W.r_max)

                add("step_meta:b3_multi", wall_us(b3_step, reps))
            elif name == "b4_multi_copies":
                width = next(n for n in (8, 16, 32, 64, 128, 256) if n >= W.r_max)

                def b4_step():
                    ad, live, uniq, inv = pairs_of(ta, tr)
                    entries = torch.full_like(ad, uniq.shape[0])
                    entries[live] = inv
                    return entries.to(x.device), torch.zeros(x.shape[0], width, dtype=x.dtype, device=x.device)

                sb._ext()
                add("step_meta:b4_multi_copies", wall_us(b4_step, reps))
            else:
                add(f"prepare:{name}", wall_us(lambda: impl.prepare(x, W, ta, tr), reps))
        print(f"  prepare-cost {w.lengths or 'decode'} D={w.num_decodes} N={w.num_adapters} r{w.r_max} s{w.seed} "
              f"T={x.shape[0]}", flush=True)
    write(rows, out)


def write(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}", flush=True)


def after_sweep(kind: str, config: str, out: str) -> None:
    """Called by the sweep runners once their sweep is done: `<out stem>-prepare.csv` (skipped if it exists)."""
    dest = Path(out).with_name(Path(out).stem + "-prepare.csv")
    if dest.exists():
        print(f"prepare cost: {dest} exists, skipped", flush=True)
        return
    print(f"prepare cost ({kind}) -> {dest}", flush=True)
    (single if kind == "single" else multi)(yaml.safe_load(Path(config).read_text()), dest, 20)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=["single", "multi"])
    p.add_argument("config")
    p.add_argument("--out", required=True)
    p.add_argument("--reps", type=int, default=20)
    a = p.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    (single if a.kind == "single" else multi)(cfg, Path(a.out), a.reps)


if __name__ == "__main__":
    main()
