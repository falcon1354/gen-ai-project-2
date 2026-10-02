# Results (Section 4)

## Table 1 – Data

| | Train | Dev | Test |
|---|---:|---:|---:|
| Pairs | 56355 | 8421 | 15878 |
| Mean source length (tokens) | 41.47 | 42.01 | 42.17 |
| Max source length (tokens) | 230 | 194 | 305 |
| Mean target length (tokens) | 17.05 | 17.21 | 17.34 |
| Max target length (tokens) | 76 | 51 | 55 |
| Pairs dropped as too long (>512 src or >128 tgt) | 0 | 0 | 0 |

## Table 2 – Model and training

| Item | Value |
|---|---|
| Trainable parameters | 7578624 |
| Epochs trained / best epoch | 20 / 19 |
| Best dev loss | 1.4948123068941328 |
| Training time | — |
| GPU | EDIT ME (e.g. Tesla T4) |

Notes:
- training_history.json missing (loss curve needs a re-train with updated train.py, or paste Colab logs).

## Table 3 – Official metrics

| Split | Decoding | Logical form (%) | Execution (%) | Parse failures (%) |
|---|---|---:|---:|---:|
| dev | greedy | 40.16 | 47.31 | 0.9381308633178957 |
| dev | beam (4) | 40.48 | 48.0 | 0.96 |
| test | beam (4) | 38.78 | 46.35 | 0.9132132510391737 |

## Table 4 – Component accuracy (dev, beam)

| Component | Accuracy (%) |
|---|---:|
| sel column correct | 68.07 |
| agg correct | 88.11 |
| WHERE clause correct | 54.13 |

## Figures checklist

| # | Figure | File |
|---|---|---|
| 1 | Positional-encoding heat-map | `positional_encoding_heatmap.png` |
| 2 | Train/dev loss per epoch | `loss_curves.png` |
| 3 | Learning-rate schedule | `lr_schedule.png` |
| 4 | Cross-attention map | `cross_attention_heatmap.png` |
| 5 | Front-end screenshot | `frontend_screenshot.png` (add manually) |
