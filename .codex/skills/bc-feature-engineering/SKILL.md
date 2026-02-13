---
name: bc-feature-engineering
description: Build and evolve Bot Cripto feature engineering pipeline, including indicators and processed datasets with tests. Use when adding or changing model input features.
---

# bc-feature-engineering

1. Keep transformations in `src/bot_cripto/features/engineering.py`.
2. Maintain deterministic, vectorized features:
   - returns/log-returns
   - RSI, MACD, ATR
   - rolling volatility
   - relative volume
   - time-based cyclic/context features
3. Drop or impute NaNs consistently after rolling windows.
4. Save processed datasets to `data/processed/*_features.parquet`.
5. Update `tests/test_features.py` whenever feature schema changes.
