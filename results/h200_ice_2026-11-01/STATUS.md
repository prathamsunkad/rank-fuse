# Profiling status

Date: 2026-11-01

Resume command: bash scripts/ice/profile_all.sh

| Stage | Job ID | Dependency |
|---|---:|---|
| setup | 6765460 | none |
| check | 6765461 | afterok:6765460 |
| latency | 6765462 | afterok:6765461 |
| ablation | 6765463 | afterok:6765462 |
| ncu full | 6765464 | afterok:6765463 |
| ncu noflush | 6765465 | afterok:6765464 |
| analysis | 6765466 | afterok:6765465 |

Next command: `squeue -j 6765460,6765461,6765462,6765463,6765464,6765465,6765466` or read this file.

Resume chain: 6765718 6765719 6765720 6765721 6765722 6765723
[2026-11-01T20:34:01-05:00] START check job=6765718
[2026-11-01T20:40:10-05:00] DONE check job=6765718
[2026-11-03T11:04:42-05:00] START latency job=6765719
