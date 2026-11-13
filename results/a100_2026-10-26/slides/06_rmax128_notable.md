### r_max 128: the notable rows

NVIDIA A100-PCIE-40GB, 4096 x 4096, fp16, tiers 16/32/64/128, seed 0. Picked from table 05. Prefill = 16-token requests.

| phase | requests x tokens | tokens | mix | max rank | rank groups | B1 us | RF1 us | RF2 us | RF1 vs B1 | RF2 vs RF1 | RF2 vs B1 | what it shows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decode | 16 x 1 | 16 | uniform | 64 | 3 | 20.8 | 9.0 | 10.2 | 2.33x | 0.88x | 2.05x | small decode: RF1 wins, too few tokens for RF2 to skip |
| decode | 64 x 1 | 64 | uniform | 128 | 4 | 24.1 | 15.6 | 15.2 | 1.55x | 1.02x | 1.59x | decode, max rank 128: RF2 = RF1 |
| decode | 256 x 1 | 256 | skew_low | 128 | 4 | 30.0 | 27.0 | 23.5 | 1.11x | 1.15x | 1.28x | largest decode: RF2 starts to win |
| decode | 256 x 1 | 256 | uniform | 128 | 4 | 29.8 | 26.0 | 24.2 | 1.15x | 1.08x | 1.24x | largest decode: RF2 starts to win |
| decode | 256 x 1 | 256 | all_max | 128 | 1 | 29.9 | 26.9 | 28.3 | 1.11x | 0.95x | 1.06x | control: nothing to skip, RF2 = RF1 |
| prefill | 64 x 16 | 1024 | all_min | 16 | 1 | 41.2 | 20.7 | 22.3 | 1.99x | 0.93x | 1.85x | rank width 16: RF1 2x faster than B1 |
| prefill | 64 x 16 | 1024 | all_max | 128 | 1 | 45.7 | 50.6 | 52.2 | 0.90x | 0.97x | 0.87x | rank width 128: RF1 slower than B1 (tuning gap) |
| prefill | 64 x 16 | 1024 | skew_low | 128 | 4 | 44.7 | 50.3 | 38.3 | 0.89x | 1.31x | 1.17x | mixed: RF2 recovers it, beats B1 |
| prefill | 256 x 16 | 4096 | all_min | 16 | 1 | 85.2 | 63.1 | 65.4 | 1.35x | 0.96x | 1.30x | rank width 16: RF1 1.35x faster than B1 |
| prefill | 256 x 16 | 4096 | all_max | 128 | 1 | 113.8 | 141.8 | 148.9 | 0.80x | 0.95x | 0.76x | rank width 128: RF1 0.8x of B1; control for RF2 |
| prefill | 256 x 16 | 4096 | skew_low | 128 | 4 | 113.8 | 141.6 | 90.0 | 0.80x | 1.57x | 1.26x | best RF2 case: 1.57x over RF1, 1.26x over B1 |
| prefill | 256 x 16 | 4096 | uniform | 128 | 4 | 113.7 | 141.2 | 102.8 | 0.80x | 1.37x | 1.11x | RF2 1.37x over RF1 |
| prefill | 256 x 16 | 4096 | skew_high | 128 | 4 | 114.2 | 142.0 | 115.7 | 0.80x | 1.23x | 0.99x | mostly high ranks: RF2 gain shrinks, ties B1 |
