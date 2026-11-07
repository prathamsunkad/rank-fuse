"""Where do small-batch timing errors come from? Times one call several ways on one GPU.

    python bench/flush_timing_check.py --out results/raw/rf2_pin/flush_timing.csv

    separate  run_sweep's method: call + flush replays, minus a flush median measured beforehand
    paired    the same, with the flush measured right before each repetition
    no_flush  the call alone, so weights can stay in L2 (no subtraction, warmer than a real forward)

Each method runs twice per workload, so its two medians give the error between repeats. flush rows
show the size and spread of what gets subtracted.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rankfuse  # noqa: E402,F401  (registers implementations)
from rankfuse.interface import REGISTRY  # noqa: E402
from rankfuse.workloads import Workload, make_batch, make_lora  # noqa: E402

REPLAYS = 20
REPS = 30


def capture(body):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(10):
            body()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    return graph


def per_replay_us(graph) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPLAYS):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3 / REPLAYS


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name()
    rows = []
    with out.open("w", newline="") as f:
        writer = None

        def emit(**row):
            nonlocal writer
            row = {"gpu": gpu, **row}
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            rows.append(row)
            print(row, flush=True)

        for mb in [256, 64]:
            flush = torch.empty(mb * 2**20, dtype=torch.uint8, device="cuda")
            flush_graph = capture(flush.zero_)
            for tokens in [1, 8, 32, 128, 512, 8192]:
                w = Workload(tokens, 1, r_max=64, dist="all_max", seed=0)
                lora, batch = make_lora(w, "cuda"), make_batch(w, "cuda")
                args = (batch.x, lora, batch.token_ranks)
                for name in ["rf1_fused", "rf2_segment_fw"]:
                    impl = REGISTRY[name]
                    kwargs = impl.prepare(*args)
                    alone = capture(lambda: impl(*args, **kwargs))

                    def with_flush():
                        flush.zero_()
                        impl(*args, **kwargs)

                    both = capture(with_flush)
                    torch.cuda.synchronize()
                    for attempt in range(2):
                        flushes = [per_replay_us(flush_graph) for _ in range(REPS)]
                        emit(flush_mb=mb, tokens=tokens, impl=name, attempt=attempt, method="flush", median_us=statistics.median(flushes),
                             min_us=min(flushes), max_us=max(flushes))
                        before = statistics.median(per_replay_us(flush_graph) for _ in range(5))
                        methods = {
                            "separate": [per_replay_us(both) - before for _ in range(REPS)],
                            "paired": [-per_replay_us(flush_graph) + per_replay_us(both) for _ in range(REPS)],
                            "no_flush": [per_replay_us(alone) for _ in range(REPS)],
                        }
                        for method, times in methods.items():
                            emit(flush_mb=mb, tokens=tokens, impl=name, attempt=attempt, method=method, median_us=statistics.median(times),
                                 min_us=min(times), max_us=max(times))


if __name__ == "__main__":
    main()
