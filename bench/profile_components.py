"""Measure RF1/RF2 components on one H200 and write a CSV."""
from __future__ import annotations
import argparse, csv, statistics, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rankfuse
from rankfuse.interface import REGISTRY
from rankfuse.workloads import Workload, make_batch, make_lora

def event_time(fn, reps=9):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    values=[]
    for _ in range(reps):
        a,b=torch.cuda.Event(True),torch.cuda.Event(True)
        a.record(); fn(); b.record(); b.synchronize()
        values.append(a.elapsed_time(b)*1000)
    return statistics.median(values)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",required=True); p.add_argument("--ablations",action="store_true"); a=p.parse_args()
    w=Workload(32,1,h_in=4096,h_out=4096,r_max=64,dist="uniform",dtype="fp16",seed=0)
    lora,batch=make_lora(w,"cuda"); args=(batch.x,lora,batch.token_ranks)
    rf1=REGISTRY["rf1_fused"]; rf2=REGISTRY["rf2_splitk"]; meta=rf2.prepare(*args)
    rows=[]
    rows.append({"kind":"component","impl":"rf2_splitk","component":"sort","median_us":event_time(lambda: torch.argsort(batch.token_ranks,stable=True))})
    rows.append({"kind":"component","impl":"rf1_fused","component":"total","median_us":event_time(lambda: rf1(*args))})
    rows.append({"kind":"component","impl":"rf2_splitk","component":"total_prepared","median_us":event_time(lambda: rf2(*args,**meta))})
    if a.ablations:
        from rankfuse.kernels.profiling_variants import VARIANTS
        order=meta["order"]
        for name,fn in VARIANTS.items():
            if name == "rf2_fixed_bucket":
                call = lambda fn=fn: fn(batch.x,lora,batch.token_ranks,order,64)
            else:
                call = lambda fn=fn: fn(batch.x,lora,batch.token_ranks,order)
            rows.append({"kind":"ablation","impl":name,"component":name,"median_us":event_time(call)})
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    with open(a.out,"w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["kind","impl","component","median_us"]); writer.writeheader(); writer.writerows(rows)
if __name__=="__main__": main()
