### 10. Selector: rf2_ma_bucketed or rf1_ma (fitted on the same data)

Rule (`rankfuse.kernels.rankfuse_multi.use_bucketed`): run rf2_ma_bucketed when work_ratio <= 0.7,
1 <= T <= 8192 and h_in >= 2048; else rf1_ma. work_ratio = rank rows the bucketed kernels
multiply / rank rows rf1_ma multiplies (each token at its own bucket vs every token at the batch's bucket).
Searched over a small threshold grid on the 1050 main-sweep batches and evaluated on the same
batches: an optimistic, in-sample estimate.

| policy | speedup over always rf1_ma (gmean) |
|---|---|
| always rf2_ma_bucketed | 0.999x |
| selector (220/1050 batches pick bucketed; worst batch 0.76x) | 1.004x |
| speedup on the batches it picks | 1.018x |
| oracle (faster of the two per batch) | 1.022x |
