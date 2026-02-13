---
name: bc-tft-model
description: Implement and maintain Bot Cripto Temporal Fusion Transformer pipeline with pytorch-forecasting quantile outputs. Use when working on TFT dataset, training, evaluation, and persistence.
---

# bc-tft-model

1. Keep TFT logic in `src/bot_cripto/models/tft.py`.
2. Build `TimeSeriesDataSet` with explicit encoder/prediction lengths.
3. Train with quantile loss and export quantile outputs used by signal contract.
4. Save both model weights and dataset parameters for reproducible inference.
5. Validate with `tests/test_tft.py` and avoid API drift vs `BasePredictor`.
