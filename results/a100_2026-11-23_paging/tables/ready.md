### Time until a rank-r request on a cold adapter can run (NVIDIA A100 80GB PCIe, r_max 256) — rerun job 7284990 (source/rerun_overlap_ready)

ensure() call to copy_done (host wall clock; device copy time in brackets). prefix = ceil(r/16) whole-model pages; whole = all 16 pages (what a whole-adapter policy loads).

| model | rank | prefix pages | prefix ms | whole ms | speed-up |
|---|---:|---:|---:|---:|---:|
| 0.5B | 16 | 1 | 1.64 (1.62) | 14.30 (14.28) | x8.7 |
| 0.5B | 32 | 2 | 1.42 (1.40) | 10.98 (10.97) | x7.8 |
| 0.5B | 64 | 4 | 2.79 (2.77) | 10.97 (10.96) | x3.9 |
| 0.5B | 128 | 8 | 5.52 (5.50) | 10.98 (10.96) | x2.0 |
| 0.5B | 256 | 16 | 10.98 (10.96) | 10.97 (10.95) | x1.0 |
| 7B | 16 | 1 | 3.14 (3.12) | 49.42 (49.40) | x15.8 |
| 7B | 32 | 2 | 6.22 (6.20) | 49.41 (49.39) | x7.9 |
| 7B | 64 | 4 | 12.40 (12.38) | 49.41 (49.40) | x4.0 |
| 7B | 128 | 8 | 24.72 (24.71) | 49.39 (49.37) | x2.0 |
| 7B | 256 | 16 | 49.38 (49.37) | 49.39 (49.37) | x1.0 |

Host cost of ensure() when all pages are resident (0.5B): 12.0 us
Host cost of ensure() when all pages are resident (7B): 12.1 us
