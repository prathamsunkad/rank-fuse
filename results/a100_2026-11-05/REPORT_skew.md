# RF1 and RF2 after standardization

Above 1x means the first kernel named is faster. Wins and losses lie outside the noise floor.

## NVIDIA A100 80GB PCIe

### Noise floor (90th percentile of |rf1_fused pass 1 / pass 0 - 1|)

| tokens | noise |
|---|---|
| 129-1024 | 0.6% |
| 1025-4096 | 0.4% |
| >4096 | 0.3% |

### Parity: nothing to skip, rf1_fused vs full-width segment

| mode | r_max | dist | tokens | workloads | median | min | max | wins | ties | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| add | 256 | all_max | 129-1024 | 4 | 1.059x | 1.05x | 1.07x | 4 | 0 | 0 |
| add | 256 | all_max | 1025-4096 | 2 | 1.126x | 1.13x | 1.13x | 2 | 0 | 0 |
| add | 256 | all_max | >4096 | 6 | 1.135x | 1.13x | 1.14x | 6 | 0 | 0 |
| add | 256 | all_min | 129-1024 | 4 | 1.058x | 1.04x | 1.07x | 4 | 0 | 0 |
| add | 256 | all_min | 1025-4096 | 2 | 1.033x | 1.03x | 1.03x | 2 | 0 | 0 |
| add | 256 | all_min | >4096 | 6 | 1.017x | 1.02x | 1.02x | 6 | 0 | 0 |

### Ideal mixes. Cost of the RF2 path: rf1_fused vs full-width segment

| shape | r_max | request_order | tokens | workloads | median | min | max | wins | ties | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096x4096 | 256 | random | 129-1024 | 6 | 1.058x | 1.05x | 1.07x | 6 | 0 | 0 |
| 4096x4096 | 256 | random | 1025-4096 | 3 | 1.128x | 1.13x | 1.13x | 3 | 0 | 0 |
| 4096x4096 | 256 | random | >4096 | 9 | 1.140x | 1.13x | 1.14x | 9 | 0 | 0 |
| 4096x4096 | 256 | sorted | 129-1024 | 6 | 1.058x | 1.05x | 1.06x | 6 | 0 | 0 |
| 4096x4096 | 256 | sorted | 1025-4096 | 3 | 1.128x | 1.13x | 1.13x | 3 | 0 | 0 |
| 4096x4096 | 256 | sorted | >4096 | 9 | 1.140x | 1.13x | 1.14x | 9 | 0 | 0 |

### Ideal mixes. Skipping alone: segment vs its full-width control

| shape | r_max | request_order | tokens | workloads | median | min | max | wins | ties | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096x4096 | 256 | random | 129-1024 | 6 | 1.416x | 1.09x | 1.59x | 6 | 0 | 0 |
| 4096x4096 | 256 | random | 1025-4096 | 3 | 1.756x | 1.09x | 2.10x | 3 | 0 | 0 |
| 4096x4096 | 256 | random | >4096 | 9 | 2.341x | 1.16x | 2.55x | 9 | 0 | 0 |
| 4096x4096 | 256 | sorted | 129-1024 | 6 | 1.496x | 1.40x | 1.58x | 6 | 0 | 0 |
| 4096x4096 | 256 | sorted | 1025-4096 | 3 | 2.069x | 1.98x | 2.08x | 3 | 0 | 0 |
| 4096x4096 | 256 | sorted | >4096 | 9 | 2.245x | 2.10x | 2.44x | 9 | 0 | 0 |

### Ideal mixes. Net: segment vs rf1_fused

| shape | r_max | request_order | tokens | workloads | median | min | max | wins | ties | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096x4096 | 256 | random | 129-1024 | 6 | 1.345x | 1.03x | 1.49x | 6 | 0 | 0 |
| 4096x4096 | 256 | random | 1025-4096 | 3 | 1.557x | 0.97x | 1.86x | 2 | 0 | 1 |
| 4096x4096 | 256 | random | >4096 | 9 | 2.059x | 1.02x | 2.24x | 9 | 0 | 0 |
| 4096x4096 | 256 | sorted | 129-1024 | 6 | 1.412x | 1.33x | 1.49x | 6 | 0 | 0 |
| 4096x4096 | 256 | sorted | 1025-4096 | 3 | 1.833x | 1.76x | 1.85x | 3 | 0 | 0 |
| 4096x4096 | 256 | sorted | >4096 | 9 | 1.971x | 1.85x | 2.14x | 9 | 0 | 0 |

### Ideal mixes. Net: segment vs B1

| shape | r_max | request_order | tokens | workloads | median | min | max | wins | ties | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096x4096 | 256 | random | 129-1024 | 6 | 0.933x | 0.73x | 1.20x | 2 | 0 | 4 |
| 4096x4096 | 256 | random | 1025-4096 | 3 | 0.811x | 0.50x | 0.96x | 0 | 0 | 3 |
| 4096x4096 | 256 | random | >4096 | 9 | 0.907x | 0.45x | 0.99x | 0 | 0 | 9 |
| 4096x4096 | 256 | sorted | 129-1024 | 6 | 1.060x | 0.92x | 1.20x | 3 | 0 | 3 |
| 4096x4096 | 256 | sorted | 1025-4096 | 3 | 0.953x | 0.92x | 0.96x | 0 | 0 | 3 |
| 4096x4096 | 256 | sorted | >4096 | 9 | 0.867x | 0.81x | 0.94x | 0 | 0 | 9 |

### Ideal mixes. Skipping alone: sorted-index vs its full-width control

| shape | r_max | request_order | tokens | workloads | median | min | max | wins | ties | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096x4096 | 256 | random | 129-1024 | 6 | 1.465x | 1.37x | 1.55x | 6 | 0 | 0 |
| 4096x4096 | 256 | random | 1025-4096 | 3 | 1.938x | 1.87x | 1.95x | 3 | 0 | 0 |
| 4096x4096 | 256 | random | >4096 | 9 | 2.081x | 1.96x | 2.23x | 9 | 0 | 0 |
| 4096x4096 | 256 | sorted | 129-1024 | 6 | 1.467x | 1.38x | 1.55x | 6 | 0 | 0 |
| 4096x4096 | 256 | sorted | 1025-4096 | 3 | 1.933x | 1.86x | 1.95x | 3 | 0 | 0 |
| 4096x4096 | 256 | sorted | >4096 | 9 | 2.074x | 1.96x | 2.23x | 9 | 0 | 0 |

### Ideal mixes. Index cost: segment vs sorted-index (read the sorted rows: same blocks)

| shape | r_max | request_order | tokens | workloads | median | min | max | wins | ties | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096x4096 | 256 | random | 129-1024 | 6 | 1.012x | 0.73x | 1.05x | 4 | 0 | 2 |
| 4096x4096 | 256 | random | 1025-4096 | 3 | 0.921x | 0.60x | 1.11x | 1 | 0 | 2 |
| 4096x4096 | 256 | random | >4096 | 9 | 1.148x | 0.59x | 1.17x | 7 | 0 | 2 |
| 4096x4096 | 256 | sorted | 129-1024 | 6 | 1.042x | 1.04x | 1.04x | 6 | 0 | 0 |
| 4096x4096 | 256 | sorted | 1025-4096 | 3 | 1.090x | 1.09x | 1.09x | 3 | 0 | 0 |
| 4096x4096 | 256 | sorted | >4096 | 9 | 1.098x | 1.09x | 1.11x | 9 | 0 | 0 |

### Ideal mixes. Net: sorted-index vs rf1_fused

| shape | r_max | request_order | tokens | workloads | median | min | max | wins | ties | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096x4096 | 256 | random | 129-1024 | 6 | 1.356x | 1.27x | 1.43x | 6 | 0 | 0 |
| 4096x4096 | 256 | random | 1025-4096 | 3 | 1.678x | 1.62x | 1.69x | 3 | 0 | 0 |
| 4096x4096 | 256 | random | >4096 | 9 | 1.802x | 1.70x | 1.93x | 9 | 0 | 0 |
| 4096x4096 | 256 | sorted | 129-1024 | 6 | 1.358x | 1.27x | 1.43x | 6 | 0 | 0 |
| 4096x4096 | 256 | sorted | 1025-4096 | 3 | 1.678x | 1.62x | 1.69x | 3 | 0 | 0 |
| 4096x4096 | 256 | sorted | >4096 | 9 | 1.796x | 1.69x | 1.93x | 9 | 0 | 0 |

### Ideal mixes. Net: sorted-index vs B1

| shape | r_max | request_order | tokens | workloads | median | min | max | wins | ties | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096x4096 | 256 | random | 129-1024 | 6 | 1.018x | 0.88x | 1.15x | 3 | 0 | 3 |
| 4096x4096 | 256 | random | 1025-4096 | 3 | 0.868x | 0.84x | 0.88x | 0 | 0 | 3 |
| 4096x4096 | 256 | random | >4096 | 9 | 0.791x | 0.75x | 0.85x | 0 | 0 | 9 |
| 4096x4096 | 256 | sorted | 129-1024 | 6 | 1.022x | 0.89x | 1.14x | 3 | 0 | 3 |
| 4096x4096 | 256 | sorted | 1025-4096 | 3 | 0.871x | 0.84x | 0.88x | 0 | 0 | 3 |
| 4096x4096 | 256 | sorted | >4096 | 9 | 0.790x | 0.75x | 0.85x | 0 | 0 | 9 |

### Ideal mixes. Bucketed vs segment

| shape | r_max | request_order | tokens | workloads | median | min | max | wins | ties | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096x4096 | 256 | random | 129-1024 | 6 | 1.357x | 1.30x | 1.76x | 6 | 0 | 0 |
| 4096x4096 | 256 | random | 1025-4096 | 3 | 1.727x | 1.47x | 2.51x | 3 | 0 | 0 |
| 4096x4096 | 256 | random | >4096 | 9 | 1.731x | 1.65x | 3.24x | 9 | 0 | 0 |
| 4096x4096 | 256 | sorted | 129-1024 | 6 | 1.302x | 1.23x | 1.34x | 6 | 0 | 0 |
| 4096x4096 | 256 | sorted | 1025-4096 | 3 | 1.456x | 1.38x | 1.49x | 3 | 0 | 0 |
| 4096x4096 | 256 | sorted | >4096 | 9 | 1.798x | 1.67x | 1.84x | 9 | 0 | 0 |

### Ideal mixes. Bucketed: extra tuning

| shape | r_max | request_order | tokens | workloads | median | min | max | wins | ties | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096x4096 | 256 | random | 129-1024 | 6 | 1.237x | 1.17x | 1.45x | 6 | 0 | 0 |
| 4096x4096 | 256 | random | 1025-4096 | 3 | 1.359x | 1.17x | 1.40x | 3 | 0 | 0 |
| 4096x4096 | 256 | random | >4096 | 9 | 1.142x | 1.11x | 1.36x | 9 | 0 | 0 |
| 4096x4096 | 256 | sorted | 129-1024 | 6 | 1.240x | 1.17x | 1.46x | 6 | 0 | 0 |
| 4096x4096 | 256 | sorted | 1025-4096 | 3 | 1.370x | 1.17x | 1.41x | 3 | 0 | 0 |
| 4096x4096 | 256 | sorted | >4096 | 9 | 1.140x | 1.12x | 1.36x | 9 | 0 | 0 |

### Sampled mixes. Cost of the RF2 path: rf1_fused vs full-width segment

| shape | r_max | request_order | tokens | workloads | median | min | max | wins | ties | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096x4096 | 256 | random | 129-1024 | 6 | 1.057x | 1.05x | 1.06x | 6 | 0 | 0 |
| 4096x4096 | 256 | random | 1025-4096 | 3 | 1.127x | 1.13x | 1.13x | 3 | 0 | 0 |
| 4096x4096 | 256 | random | >4096 | 9 | 1.043x | 1.02x | 1.14x | 9 | 0 | 0 |
| 4096x4096 | 256 | sorted | 129-1024 | 6 | 1.061x | 1.05x | 1.06x | 6 | 0 | 0 |
| 4096x4096 | 256 | sorted | 1025-4096 | 3 | 1.128x | 1.13x | 1.13x | 3 | 0 | 0 |
| 4096x4096 | 256 | sorted | >4096 | 9 | 1.044x | 1.02x | 1.14x | 9 | 0 | 0 |

### Sampled mixes. Skipping alone: segment vs its full-width control

| shape | r_max | request_order | tokens | workloads | median | min | max | wins | ties | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096x4096 | 256 | random | 129-1024 | 6 | 1.351x | 1.18x | 1.52x | 6 | 0 | 0 |
| 4096x4096 | 256 | random | 1025-4096 | 3 | 1.282x | 1.23x | 1.58x | 3 | 0 | 0 |
| 4096x4096 | 256 | random | >4096 | 9 | 1.555x | 1.00x | 2.21x | 8 | 1 | 0 |
| 4096x4096 | 256 | sorted | 129-1024 | 6 | 1.492x | 1.40x | 1.58x | 6 | 0 | 0 |
| 4096x4096 | 256 | sorted | 1025-4096 | 3 | 2.025x | 2.01x | 2.06x | 3 | 0 | 0 |
| 4096x4096 | 256 | sorted | >4096 | 9 | 1.623x | 1.00x | 2.41x | 8 | 1 | 0 |

### Sampled mixes. Net: segment vs rf1_fused

| shape | r_max | request_order | tokens | workloads | median | min | max | wins | ties | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096x4096 | 256 | random | 129-1024 | 6 | 1.277x | 1.12x | 1.43x | 6 | 0 | 0 |
| 4096x4096 | 256 | random | 1025-4096 | 3 | 1.139x | 1.09x | 1.40x | 3 | 0 | 0 |
| 4096x4096 | 256 | random | >4096 | 9 | 1.368x | 0.99x | 1.94x | 8 | 0 | 1 |
| 4096x4096 | 256 | sorted | 129-1024 | 6 | 1.411x | 1.33x | 1.48x | 6 | 0 | 0 |
| 4096x4096 | 256 | sorted | 1025-4096 | 3 | 1.796x | 1.78x | 1.83x | 3 | 0 | 0 |
| 4096x4096 | 256 | sorted | >4096 | 9 | 1.554x | 0.99x | 2.12x | 8 | 0 | 1 |

### Sampled mixes. Net: segment vs B1

| shape | r_max | request_order | tokens | workloads | median | min | max | wins | ties | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096x4096 | 256 | random | 129-1024 | 6 | 0.957x | 0.78x | 1.15x | 3 | 0 | 3 |
| 4096x4096 | 256 | random | 1025-4096 | 3 | 0.592x | 0.57x | 0.73x | 0 | 0 | 3 |
| 4096x4096 | 256 | random | >4096 | 9 | 1.751x | 0.54x | 1.79x | 5 | 0 | 4 |
| 4096x4096 | 256 | sorted | 129-1024 | 6 | 1.059x | 0.93x | 1.19x | 3 | 0 | 3 |
| 4096x4096 | 256 | sorted | 1025-4096 | 3 | 0.932x | 0.93x | 0.95x | 0 | 0 | 3 |
| 4096x4096 | 256 | sorted | >4096 | 9 | 1.600x | 0.87x | 1.78x | 5 | 0 | 4 |

### Sampled mixes. Skipping alone: sorted-index vs its full-width control

| shape | r_max | request_order | tokens | workloads | median | min | max | wins | ties | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096x4096 | 256 | random | 129-1024 | 6 | 1.465x | 1.38x | 1.54x | 6 | 0 | 0 |
| 4096x4096 | 256 | random | 1025-4096 | 3 | 1.904x | 1.90x | 1.93x | 3 | 0 | 0 |
| 4096x4096 | 256 | random | >4096 | 9 | 1.679x | 1.00x | 2.18x | 8 | 1 | 0 |
| 4096x4096 | 256 | sorted | 129-1024 | 6 | 1.466x | 1.38x | 1.55x | 6 | 0 | 0 |
| 4096x4096 | 256 | sorted | 1025-4096 | 3 | 1.905x | 1.90x | 1.93x | 3 | 0 | 0 |
| 4096x4096 | 256 | sorted | >4096 | 9 | 1.678x | 1.00x | 2.20x | 8 | 1 | 0 |

### Sampled mixes. Index cost: segment vs sorted-index (read the sorted rows: same blocks)

| shape | r_max | request_order | tokens | workloads | median | min | max | wins | ties | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096x4096 | 256 | random | 129-1024 | 6 | 0.913x | 0.88x | 1.01x | 1 | 1 | 4 |
| 4096x4096 | 256 | random | 1025-4096 | 3 | 0.694x | 0.67x | 0.84x | 0 | 0 | 3 |
| 4096x4096 | 256 | random | >4096 | 9 | 1.035x | 0.70x | 1.12x | 6 | 0 | 3 |
| 4096x4096 | 256 | sorted | 129-1024 | 6 | 1.043x | 1.04x | 1.05x | 6 | 0 | 0 |
| 4096x4096 | 256 | sorted | 1025-4096 | 3 | 1.085x | 1.08x | 1.09x | 3 | 0 | 0 |
| 4096x4096 | 256 | sorted | >4096 | 9 | 1.038x | 1.01x | 1.11x | 9 | 0 | 0 |

### Sampled mixes. Net: sorted-index vs rf1_fused

| shape | r_max | request_order | tokens | workloads | median | min | max | wins | ties | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096x4096 | 256 | random | 129-1024 | 6 | 1.352x | 1.27x | 1.42x | 6 | 0 | 0 |
| 4096x4096 | 256 | random | 1025-4096 | 3 | 1.642x | 1.63x | 1.66x | 3 | 0 | 0 |
| 4096x4096 | 256 | random | >4096 | 9 | 1.503x | 0.97x | 1.88x | 8 | 0 | 1 |
| 4096x4096 | 256 | sorted | 129-1024 | 6 | 1.358x | 1.27x | 1.42x | 6 | 0 | 0 |
| 4096x4096 | 256 | sorted | 1025-4096 | 3 | 1.651x | 1.64x | 1.67x | 3 | 0 | 0 |
| 4096x4096 | 256 | sorted | >4096 | 9 | 1.497x | 0.97x | 1.91x | 8 | 0 | 1 |

### Sampled mixes. Net: sorted-index vs B1

| shape | r_max | request_order | tokens | workloads | median | min | max | wins | ties | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096x4096 | 256 | random | 129-1024 | 6 | 1.015x | 0.88x | 1.14x | 3 | 0 | 3 |
| 4096x4096 | 256 | random | 1025-4096 | 3 | 0.853x | 0.85x | 0.86x | 0 | 0 | 3 |
| 4096x4096 | 256 | random | >4096 | 9 | 1.549x | 0.78x | 1.76x | 5 | 0 | 4 |
| 4096x4096 | 256 | sorted | 129-1024 | 6 | 1.019x | 0.89x | 1.14x | 3 | 0 | 3 |
| 4096x4096 | 256 | sorted | 1025-4096 | 3 | 0.859x | 0.86x | 0.87x | 0 | 0 | 3 |
| 4096x4096 | 256 | sorted | >4096 | 9 | 1.545x | 0.79x | 1.76x | 5 | 0 | 4 |

### Sampled mixes. Bucketed vs segment

| shape | r_max | request_order | tokens | workloads | median | min | max | wins | ties | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096x4096 | 256 | random | 129-1024 | 6 | 1.353x | 1.18x | 1.49x | 6 | 0 | 0 |
| 4096x4096 | 256 | random | 1025-4096 | 3 | 2.155x | 1.84x | 2.26x | 3 | 0 | 0 |
| 4096x4096 | 256 | random | >4096 | 9 | 1.165x | 1.00x | 2.86x | 9 | 0 | 0 |
| 4096x4096 | 256 | sorted | 129-1024 | 6 | 1.215x | 1.16x | 1.28x | 6 | 0 | 0 |
| 4096x4096 | 256 | sorted | 1025-4096 | 3 | 1.386x | 1.38x | 1.41x | 3 | 0 | 0 |
| 4096x4096 | 256 | sorted | >4096 | 9 | 1.253x | 1.00x | 1.84x | 9 | 0 | 0 |

### Sampled mixes. Bucketed: extra tuning

| shape | r_max | request_order | tokens | workloads | median | min | max | wins | ties | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096x4096 | 256 | random | 129-1024 | 6 | 1.304x | 1.16x | 1.42x | 6 | 0 | 0 |
| 4096x4096 | 256 | random | 1025-4096 | 3 | 1.351x | 1.34x | 1.38x | 3 | 0 | 0 |
| 4096x4096 | 256 | random | >4096 | 9 | 1.048x | 0.98x | 1.31x | 6 | 0 | 3 |
| 4096x4096 | 256 | sorted | 129-1024 | 6 | 1.270x | 1.15x | 1.42x | 6 | 0 | 0 |
| 4096x4096 | 256 | sorted | 1025-4096 | 3 | 1.339x | 1.34x | 1.38x | 3 | 0 | 0 |
| 4096x4096 | 256 | sorted | >4096 | 9 | 1.046x | 0.98x | 1.32x | 6 | 0 | 3 |

### Ideal mixes: share of the full-width-to-ideal gap captured (linear estimate)

| r_max | mix | order | kernel | batches | median share captured |
|---|---|---|---|---|---|
| 256 | low_frac_100 | random | bucketed | 6 | 89% |
| 256 | low_frac_100 | random | segment | 6 | 69% |
| 256 | low_frac_100 | sorted | bucketed | 6 | 88% |
| 256 | low_frac_100 | sorted | segment | 6 | 67% |
| 256 | low_frac_90 | random | bucketed | 6 | 92% |
| 256 | low_frac_90 | random | segment | 6 | 16% |
| 256 | low_frac_90 | sorted | bucketed | 6 | 92% |
| 256 | low_frac_90 | sorted | segment | 6 | 70% |
| 256 | low_frac_99 | random | bucketed | 6 | 89% |
| 256 | low_frac_99 | random | segment | 6 | 59% |
| 256 | low_frac_99 | sorted | bucketed | 6 | 89% |
| 256 | low_frac_99 | sorted | segment | 6 | 68% |

### Overhead with nothing to skip, in us (one-rank batches)

| r_max | mix | tokens | batch | rf1_fused us | rf2_probe_read vs rf1 | rf2_probe_branch vs rf1 | rf2_segment_fw vs rf1 | rf2_index_fw vs rf1 |
|---|---|---|---|---|---|---|---|---|
| 256 | all_max | 256 | 256x1 | 47.8 |  |  | +3.07 (+6.4%) | +4.12 (+8.6%) |
| 256 | all_max | 256 | 256x1 | 47.6 |  |  | +3.12 (+6.6%) | +4.20 (+8.8%) |
| 256 | all_max | 512 | 512x1 | 68.6 |  |  | +3.76 (+5.5%) | +5.35 (+7.8%) |
| 256 | all_max | 512 | 512x1 | 68.7 |  |  | +3.69 (+5.4%) | +5.58 (+8.1%) |
| 256 | all_max | 2048 | 2048x1 | 198.5 |  |  | +24.57 (+12.4%) | +29.75 (+15.0%) |
| 256 | all_max | 2048 | 2048x1 | 198.7 |  |  | +24.93 (+12.5%) | +29.87 (+15.0%) |
| 256 | all_max | 8192 | 8192x1 | 813.1 |  |  | +114.20 (+14.0%) | +128.70 (+15.8%) |
| 256 | all_max | 8192 | 8192x1 | 819.1 |  |  | +110.36 (+13.5%) | +125.18 (+15.3%) |
| 256 | all_max | 8192 | 32x256 | 819.2 |  |  | +112.90 (+13.8%) | +128.23 (+15.7%) |
| 256 | all_max | 8192 | 32x256 | 811.9 |  |  | +112.67 (+13.9%) | +124.80 (+15.4%) |
| 256 | all_max | 8192 | 16x512 | 819.7 |  |  | +109.80 (+13.4%) | +129.25 (+15.8%) |
| 256 | all_max | 8192 | 16x512 | 819.5 |  |  | +107.88 (+13.2%) | +128.28 (+15.7%) |
| 256 | all_min | 256 | 256x1 | 13.0 |  |  | +1.05 (+8.1%) | +1.38 (+10.7%) |
| 256 | all_min | 256 | 256x1 | 13.0 |  |  | +0.95 (+7.3%) | +1.33 (+10.2%) |
| 256 | all_min | 512 | 512x1 | 18.3 |  |  | +0.84 (+4.6%) | +1.20 (+6.6%) |
| 256 | all_min | 512 | 512x1 | 18.1 |  |  | +0.87 (+4.8%) | +1.28 (+7.1%) |
| 256 | all_min | 2048 | 2048x1 | 44.5 |  |  | +1.56 (+3.5%) | +2.15 (+4.8%) |
| 256 | all_min | 2048 | 2048x1 | 44.4 |  |  | +1.48 (+3.3%) | +2.20 (+5.0%) |
| 256 | all_min | 8192 | 8192x1 | 139.5 |  |  | +2.00 (+1.4%) | +3.53 (+2.5%) |
| 256 | all_min | 8192 | 8192x1 | 139.1 |  |  | +2.18 (+1.6%) | +4.07 (+2.9%) |
| 256 | all_min | 8192 | 32x256 | 138.8 |  |  | +2.41 (+1.7%) | +3.81 (+2.7%) |
| 256 | all_min | 8192 | 32x256 | 139.1 |  |  | +2.36 (+1.7%) | +4.20 (+3.0%) |
| 256 | all_min | 8192 | 16x512 | 139.1 |  |  | +2.27 (+1.6%) | +3.66 (+2.6%) |
| 256 | all_min | 8192 | 16x512 | 139.1 |  |  | +2.41 (+1.7%) | +3.92 (+2.8%) |

### Two scheduler assumptions: speedup of RF2 (above 1x is faster)

| r_max | mix | batch | tokens | rf1_fused us | sorted tokens vs RF1 | sorted index vs RF1 | sorted tokens vs B1 | sorted index vs B1 | sort us (host, per step) |
|---|---|---|---|---|---|---|---|---|---|
| 256 | low_frac_100 | 256x1 | 256 | 47.7 | 1.49x | 1.42x | 1.20x | 1.15x | 56 |
| 256 | low_frac_100 | 512x1 | 512 | 68.5 | 1.36x | 1.30x | 0.94x | 0.90x | 64 |
| 256 | low_frac_100 | 2048x1 | 2048 | 198.3 | 1.84x | 1.68x | 0.95x | 0.87x | 23 |
| 256 | low_frac_100 | 8192x1 | 8192 | 813.0 | 2.15x | 1.92x | 0.94x | 0.85x | 66 |
| 256 | low_frac_100 | 32x256 | 8192 | 811.6 | 2.04x | 1.85x | 0.90x | 0.81x | 58 |
| 256 | low_frac_100 | 16x512 | 8192 | 813.4 | 1.97x | 1.80x | 0.87x | 0.79x | 42 |
| 256 | low_frac_90 | 256x1 | 256 | 47.8 | 1.47x | 1.41x | 1.18x | 1.13x | 57 |
| 256 | low_frac_90 | 512x1 | 512 | 68.7 | 1.33x | 1.27x | 0.92x | 0.88x | 46 |
| 256 | low_frac_90 | 2048x1 | 2048 | 198.3 | 1.76x | 1.62x | 0.91x | 0.84x | 14 |
| 256 | low_frac_90 | 8192x1 | 8192 | 813.5 | 1.90x | 1.72x | 0.83x | 0.76x | 90 |
| 256 | low_frac_90 | 32x256 | 8192 | 804.2 | 1.89x | 1.73x | 0.84x | 0.76x | 77 |
| 256 | low_frac_90 | 16x512 | 8192 | 813.0 | 1.85x | 1.70x | 0.81x | 0.75x | 46 |
| 256 | low_frac_99 | 256x1 | 256 | 47.6 | 1.49x | 1.42x | 1.20x | 1.15x | 49 |
| 256 | low_frac_99 | 512x1 | 512 | 68.6 | 1.36x | 1.30x | 0.94x | 0.90x | 54 |
| 256 | low_frac_99 | 2048x1 | 2048 | 198.3 | 1.85x | 1.69x | 0.96x | 0.88x | 9 |
| 256 | low_frac_99 | 8192x1 | 8192 | 813.2 | 2.12x | 1.90x | 0.93x | 0.83x | 61 |
| 256 | low_frac_99 | 32x256 | 8192 | 804.6 | 2.03x | 1.84x | 0.90x | 0.81x | 76 |
| 256 | low_frac_99 | 16x512 | 8192 | 812.7 | 1.97x | 1.80x | 0.87x | 0.79x | 43 |
| 256 | skew_low | 256x1 | 256 | 47.6 | 1.47x | 1.40x | 1.18x | 1.13x | 63 |
| 256 | skew_low | 512x1 | 512 | 68.7 | 1.34x | 1.28x | 0.93x | 0.89x | 52 |
| 256 | skew_low | 2048x1 | 2048 | 198.2 | 1.80x | 1.64x | 0.93x | 0.85x | 25 |
| 256 | skew_low | 8192x1 | 8192 | 813.1 | 1.99x | 1.79x | 0.88x | 0.79x | 64 |
| 256 | skew_low | 32x256 | 8192 | 305.6 | 1.53x | 1.47x | 1.62x | 1.56x | -6 |
| 256 | skew_low | 16x512 | 8192 | 161.1 | 1.03x | 1.01x | 1.72x | 1.70x | 25 |
| 256 | zipf_1 | 256x1 | 256 | 47.8 | 1.47x | 1.41x | 1.18x | 1.13x | 59 |
| 256 | zipf_1 | 512x1 | 512 | 68.5 | 1.33x | 1.27x | 0.93x | 0.89x | 30 |
| 256 | zipf_1 | 2048x1 | 2048 | 198.1 | 1.78x | 1.63x | 0.92x | 0.85x | 4 |
| 256 | zipf_1 | 8192x1 | 8192 | 812.8 | 1.96x | 1.77x | 0.86x | 0.78x | 44 |
| 256 | zipf_1 | 32x256 | 8192 | 800.2 | 1.99x | 1.80x | 0.88x | 0.80x | 44 |
| 256 | zipf_1 | 16x512 | 8192 | 304.8 | 1.51x | 1.46x | 1.60x | 1.55x | 51 |
| 256 | zipf_2 | 256x1 | 256 | 47.8 | 1.49x | 1.42x | 1.20x | 1.14x | 61 |
| 256 | zipf_2 | 512x1 | 512 | 68.6 | 1.36x | 1.30x | 0.94x | 0.90x | 46 |
| 256 | zipf_2 | 2048x1 | 2048 | 198.2 | 1.82x | 1.66x | 0.95x | 0.86x | 46 |
| 256 | zipf_2 | 8192x1 | 8192 | 813.2 | 2.12x | 1.88x | 0.93x | 0.83x | 35 |
| 256 | zipf_2 | 32x256 | 8192 | 161.2 | 1.06x | 1.05x | 1.77x | 1.75x | 25 |
| 256 | zipf_2 | 16x512 | 8192 | 140.1 | 0.99x | 0.97x | 1.90x | 1.88x | 48 |

### Does ordering requests help? rf2_segment, sorted vs random (bench/sort_cost.py has the sort's cost)

| r_max | tokens | workloads | median | min | max | sorted faster | ties | random faster |
|---|---|---|---|---|---|---|---|---|
| 256 | 129-1024 | 12 | 1.093x | 0.99x | 1.43x | 10 | 2 | 0 |
| 256 | 1025-4096 | 6 | 1.441x | 0.99x | 1.82x | 5 | 0 | 1 |
| 256 | >4096 | 18 | 0.959x | 0.90x | 1.87x | 5 | 1 | 12 |
