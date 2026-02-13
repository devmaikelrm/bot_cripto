---
name: bc-data-ingestion-storage
description: Implement and maintain Bot Cripto market data ingestion with CCXT, validation, and raw parquet storage. Use when building or fixing fetch pipelines and ingestion reliability.
---

# bc-data-ingestion-storage

1. Use `src/bot_cripto/data/ingestion.py` as ingestion entrypoint.
2. Fetch OHLCV in pages with `since` timestamps and retry on transient exchange errors.
3. Validate continuity by timeframe and warn on gaps.
4. Persist raw data in parquet under `data/raw/{symbol}_{timeframe}.parquet`.
5. Merge incrementally and deduplicate by timestamp.
6. Keep CLI command `bot-cripto fetch` operational.
