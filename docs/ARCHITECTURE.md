# Architecture

## Pipeline

```
fetch -> validate -> raw parquet -> features -> processed parquet
      -> realtime stream snapshots (optional)
      -> train trend/return/risk -> versioned models
      -> ensemble -> regime engine -> risk engine -> decision engine
      -> paper/live executor -> telegram
```

## Layers

- `src/bot_cripto/data`: exchange ingestion and persistence.
- `src/bot_cripto/data/streaming.py`: realtime microstructure snapshots (cryptofeed/poll fallback).
- `src/bot_cripto/features`: indicators and feature transforms.
- `src/bot_cripto/models`: model contract, baseline, TFT.
- `src/bot_cripto/jobs`: train and inference orchestration.
- `src/bot_cripto/models/ensemble.py`: weighted ensemble merger.
- `src/bot_cripto/regime`: rule-based market regime detector.
- `src/bot_cripto/risk`: position sizing and drawdown constraints.
- `src/bot_cripto/backtesting`: walk-forward backtesting.
- `src/bot_cripto/monitoring`: drift detection.
- `src/bot_cripto/decision`: rule-based signal decision.
- `src/bot_cripto/execution`: paper and live execution adapters.
- `src/bot_cripto/notifications`: Telegram messaging.

## Artifact Strategy

- Raw data: `data/raw/{symbol}_{tf}.parquet`
- Realtime stream: `data/raw/stream/{symbol}_stream.parquet`
- Features: `data/processed/{symbol}_{tf}_features.parquet`
- Models: `models/{trend|return|risk|baseline}/{symbol}/{timestamp_commit}/`
  - Includes model artifacts + `metadata.json` (train metrics, calibration metrics, git commit, timestamp)
- Signal output: `logs/signal.json`
