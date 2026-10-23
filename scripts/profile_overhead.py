"""Profile wall time and CUDA kernel time for the decode overhead figure."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
from torch.profiler import DeviceType, ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rankfuse
from rankfuse.baselines.grouped import clear_graph_cache
from rankfuse.workloads import Workload, make_batch, make_lora


def main() -> None:
    rows = []
    for batch in (1, 32, 256):
        workload = Workload(batch, 1, h_in=4096, h_out=4096, r_max=64,
                            dist="uniform", dtype="fp16", seed=0)
        lora, batch_data = make_lora(workload, "cuda"), make_batch(workload, "cuda")
        for impl in ("b1_padded", "b1_padded_graphed", "b2_grouped", "b2_grouped_graphed"):
            clear_graph_cache()
            fn = rankfuse.REGISTRY[impl]
            for _ in range(30):
                fn(batch_data.x, lora, batch_data.token_ranks)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(batch_data.x, lora, batch_data.token_ranks)
            end.record()
            end.synchronize()
            event_us = start.elapsed_time(end) * 1e3
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                fn(batch_data.x, lora, batch_data.token_ranks)
                torch.cuda.synchronize()
            kernels = [
                event for event in prof.events()
                if event.device_type == DeviceType.CUDA
                and event.device_time_total > 0
                and not event.name.startswith(("Memcpy", "memcpy"))
            ]
            rows.append({
                "impl": impl,
                "num_requests": batch,
                "event_us": event_us,
                "kernel_sum_us": sum(event.device_time_total for event in kernels),
                "kernel_launches": len(kernels),
            })
    output = Path("results/profiler_overhead.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
