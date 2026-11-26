### 00. Rows per implementation and r_max

Timed rows in raw/ (seeds [0, 1, 2]); errors = rows whose call raised (time NaN, excluded from every table); shapes not run = skipped as unsupported (B4: rank width must be below h_in and h_out) or capped.

| r_max | impl | rows | errors | shapes not run | first error |
|---|---|---|---|---|---|
| 64 | rf1_ma | 5670 | 0 | - |  |
| 64 | rf2_ma_splitk | 5670 | 0 | - |  |
| 64 | rf2_ma_segment | 5670 | 0 | - |  |
| 64 | rf2_ma_bucketed | 5670 | 0 | - |  |
| 64 | b3_multi | 5670 | 0 | - |  |
| 64 | b4_multi_copies | 5670 | 0 | - |  |
| 64 | b4_multi_shared | 5670 | 0 | - |  |
| 64 | b1_multi | 5670 | 0 | - |  |
| 64 | stage1_rf1_fused | 1134 | 0 | - |  |
| 128 | rf1_ma | 5670 | 0 | - |  |
| 128 | rf2_ma_splitk | 5670 | 0 | - |  |
| 128 | rf2_ma_segment | 5670 | 0 | - |  |
| 128 | rf2_ma_bucketed | 5670 | 0 | - |  |
| 128 | b3_multi | 5670 | 0 | - |  |
| 128 | b4_multi_copies | 5040 | 0 | k_v |  |
| 128 | b4_multi_shared | 5040 | 0 | k_v |  |
| 128 | b1_multi | 5670 | 0 | - |  |
| 128 | stage1_rf1_fused | 1134 | 0 | - |  |
| 256 | rf1_ma | 5670 | 0 | - |  |
| 256 | rf2_ma_splitk | 5670 | 0 | - |  |
| 256 | rf2_ma_segment | 5670 | 0 | - |  |
| 256 | rf2_ma_bucketed | 5670 | 0 | - |  |
| 256 | b3_multi | 5670 | 0 | - |  |
| 256 | b4_multi_copies | 5040 | 0 | k_v |  |
| 256 | b4_multi_shared | 5040 | 0 | k_v |  |
| 256 | b1_multi | 5670 | 0 | - |  |
| 256 | stage1_rf1_fused | 1134 | 0 | - |  |
