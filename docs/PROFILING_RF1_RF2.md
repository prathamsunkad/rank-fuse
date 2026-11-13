# RF1/RF2 component profiling

This report is generated from measurements under `results/h200_ice_<date>/`.  All
performance claims must cite a table or figure in that directory; this document does
not substitute estimates for missing measurements.

## Algorithms

The public contract stores `A_t` as `[r_max, h_in]`, `B` as `[r_max, h_out]`, and
one rank per flattened token ([interface.py](../rankfuse/interface.py#L1)).

### RF1

`rf1_fused` uses the split-K path by default ([rankfuse_fused.py](../rankfuse/kernels/rankfuse_fused.py#L833)).
The host chooses `r_limit` with `prepare_r_limit`; the kernel rounds it to `R_PAD`,
selects `BM`, and splits the `h_in` reduction into `split_k` slices.  Each shrink
program loads a contiguous block of `x` rows and contiguous rows of `A_t`, accumulates
`x @ A_t.T` with `tl.dot`, and stores an fp32 partial in workspace.  The expand
program reads all partials, applies the per-token rank mask and optional scale, then
loads `B` and writes or adds the output.  For small token counts it uses the strip
expand kernel; at the tiled threshold it uses one output tile per program
([rankfuse_fused.py](../rankfuse/kernels/rankfuse_fused.py#L700)).

### RF2

`rf2_splitk` is the two-kernel sorted implementation
([rankfuse_rf2_splitk.py](../rankfuse/kernels/rankfuse_rf2_splitk.py#L153)).  Its
`prepare_rank_order` call computes a stable `torch.argsort` and passes `order` outside
the timed call ([rankfuse/kernels/_common.py](../rankfuse/kernels/_common.py#L54)).
The shrink kernel loads `tok = order[pos]`, gathers `ranks[tok]`, computes the tile
maximum, selects the 16/32/64/128/256 bucket, and loads only that many rank rows.
The expand kernel repeats the index setup, reads workspace, masks each token to its
rank, and scatters output rows back through `tok`
([rankfuse_rf2_splitk.py](../rankfuse/kernels/rankfuse_rf2_splitk.py#L35).

When `order` is omitted, `rf2_splitk` computes `torch.argsort` inside the call
([rankfuse_rf2_splitk.py](../rankfuse/kernels/rankfuse_rf2_splitk.py#L162)); the
fused RF2 path has the same fallback ([rankfuse_fused.py](../rankfuse/kernels/rankfuse_fused.py#L812)).
The benchmark's `prepare` path supplies `order` before graph capture
([bench/run_sweep.py](../bench/run_sweep.py#L193)).

The deck's one-launch RF2 wording therefore describes the serving-level launch
envelope, not the current `rf2_splitk` implementation: the code launches shrink and
expand separately and rank sorting is a third component when not amortised.  The
registered names and source mapping used by the benchmark are recorded in the
baseline CSVs and the companion source manifest.

## Measurement protocol

The check job is `scripts/ice/check.sbatch`; all GPU work uses `ice-gpu`, H200, and
`scripts/ice/env.sh`.  CUDA-graph timing uses the existing graph replay harness
([bench/run_sweep.py](../bench/run_sweep.py#L100)); Nsight uses
`--profile-from-start off` and `scripts/ice/ncu_profile.sh`.  `PROFILE_SORT=1`
profiles only the stable argsort through the new driver mode.  The timing-only
ablation functions live in [profiling_variants.py](../rankfuse/kernels/profiling_variants.py).

## Results

Run:

```bash
python analysis/profile_plots.py results/h200_ice_<date>
```

The script writes CSV/Markdown tables and 200-dpi figures.  It intentionally skips
figures whose required measured columns are absent, so no unmeasured number can enter
the report.  Once the H200 jobs finish, add the generated tables and figures here,
then record a verdict for each hypothesis H1-H6 with a direct table/figure citation.
