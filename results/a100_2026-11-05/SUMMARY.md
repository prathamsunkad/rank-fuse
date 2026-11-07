# RF2 sweeps, A100 80GB PCIe, 5 Nov 2026

Commit 7871e06, add mode, CUDA-graph timing with the L2 flush timed before each repetition, 3 seeds,
4096 × 4096 fp16. Raw data: `results/raw/rf2_next/`. Full tables: `REPORT_overhead.md`,
`REPORT_skew.md`, `REPORT_ideal.md` (built with `analysis/rf2_eval.py`). Speedups are
time(baseline) ÷ time(kernel), so above 1× means the kernel is faster. RF1 repeat noise was within
about 2.6% at every size.

These runs predate the A100 tuned launch table (`rankfuse/kernels/tuned_a100.py`, commit 12bf0f6),
which changes the settings of `rf1_fused` and every split-K RF2 on this GPU. Conclusions that rest
on RF1's untuned r_max 256 speed (section 2) need a re-run with the table.

Assumptions: "sorted tokens" is `rf2_segment` on physically sorted batches; "sorted index" is
`rf2_index` on random-order batches. `rf2_bucketed` reads through per-bucket indices, so it doesn't
depend on order.

## 1. Overhead with nothing to skip
Extra time over `rf1_fused` on single-rank batches:

| Batch | `rf2_segment` | `rf2_index` |
|---|---|---|
| 1–128 tokens, any r_max | +0.5 to +1.6 µs (5–13%) | +0.9 to +2.2 µs (10–18%) |
| 8192 tokens, r_max 64 | +4.0 µs (2.5%) | +7.1 µs (4.4%) |
| 8192 tokens, r_max 128, all at r_max | +11.9 µs (3.9%) | +36.3 µs (12%) |
| 8192 tokens, r_max 256, all at r_max | +108 µs (13%) | +121 µs (15%) |

Nothing can be skipped below a few hundred tokens of mixed ranks, so RF2 should not be used there.

## 2. r_max 256 is dominated by RF1's tuning gap
At r_max 256 with every request at full width, B1 is 2.3× faster than RF1 at 8192 tokens (358 vs
812 µs). So RF2's 1.9–2.1× gains over RF1 at r_max 256 mostly recover RF1's weakness:

| r_max 256, 8192 × 1, sorted | vs RF1 | vs B1 |
|---|---|---|
| `rf2_segment`, one r_max request | 2.14× | 0.94× |
| `rf2_index`, one r_max request | 1.92× | 0.85× |
| `rf2_bucketed`, one r_max request | 3.96× | 1.74× |
| `rf2_bucketed_tuned`, one r_max request | 5.39× | 2.37× |
| `rf2_bucketed`, 90% low | 3.30× | 1.44× |
| `rf2_bucketed`, 50% low | 1.60× | 0.71× |

Only the bucketed kernels beat B1 at r_max 256, and only when most requests are low rank.

## 3. r_max 128 gives the cleanest wins
RF1 is slightly faster than B1 at r_max 128 (305 vs 324 µs at 8192 tokens), so wins here are not an
RF1 artifact. At 8192 × 1, sorted:

| Low-rank share | `rf2_segment` vs RF1 / B1 | `rf2_index` vs RF1 / B1 | `rf2_bucketed` vs RF1 / B1 |
|---|---|---|---|
| All but one | 1.68× / 1.78× | 1.61× / 1.71× | 2.03× / 2.16× |
| 90% | 1.48× / 1.57× | 1.42× / 1.51× | 1.78× / 1.89× |
| 50% | 1.26× / 1.34× | 1.18× / 1.25× | 1.42× / 1.51× |
| 25% | 1.15× / 1.21× | 1.06× / 1.12× | 1.23× / 1.30× |

At r_max 64 RF2 gains at most about 8% over RF1, but every RF kernel is about 1.7× faster than B1
there, from the shared kernel's general optimizations.

## 4. The two assumptions (r_max 256 skew mixes)
* Index cost: `rf2_index` is 4–11% slower than `rf2_segment` on the same sorted blocks, rising with
  batch size.
* The host sort (`prepare_us`) took a median of about 690 µs per call, measured in Python with a
  GPU sync. It is paid once per step, not per layer, but it is not small and was not measured on
  the GPU alone.
* Order matters only when high-rank requests are common. With one r_max request, `rf2_segment` in
  random order matched sorted (363 vs 379 µs at 8192 × 1), as predicted. At 90% low, random order
  took 799 µs against 428 sorted.

## 5. What limits `rf2_segment` and `rf2_index`
With one r_max 256 request at 8192 × 1 (ideal ≈ 139 µs, RF1 on an all-low batch):

| Kernel | Time | What differs from the row above |
|---|---|---|
| `rf2_segment`, sorted | 379 µs | |
| `rf2_bucketed`, sorted | 205 µs | Same 16-token blocks, but each launch compiled for one width, with no rank read or branch |
| `rf2_bucketed_tuned`, sorted | 151 µs | Larger blocks for narrow buckets (128 tokens) |

* Asymmetry 2, the run-time width choice in a five-width kernel, costs about 1.85× on narrow blocks
  here, much more than the 6–7% measured at full width. That is consistent with registers sized for
  the widest path limiting occupancy, but it hasn't been confirmed with a profiler.
* Asymmetry 1, block size tied to the batch width, costs about another 1.36×.
* `rf2_segment` captures 36–64% of the gap between RF1 and the ideal on the skew mixes.

## Takeaways
1. Skipping works: wins grow with r_max and with the low-rank share, and r_max 128 shows them
   against both RF1 and B1.
2. At r_max 256, RF1 needs tuning before RF2-vs-RF1 numbers mean much; B1 is the honest baseline.
3. The run-time-branch kernels leave most of the possible gain unrealized on narrow blocks.
   Per-width compilation (as in `rf2_bucketed`) is worth about 1.85× there, larger blocks another 1.36×.
4. The sorted-index assumption costs 4–11% over physically sorted tokens in-kernel, plus the sort.
