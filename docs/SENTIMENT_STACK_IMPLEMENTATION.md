# Sentiment Stack Implementation Plan

Date: 2026-02-19  
Scope: complete real-time sentiment from roadmap section 5 with safe staged rollout.

## Objective

Upgrade sentiment from lexicon-only partial mode to a robust hybrid stack:

1. Native source collection (X + Telegram + optional external endpoint + CryptoPanic + local).
2. NLP scoring (`finBERT` by default) with automatic fallback.
3. Stable integration in inference without pipeline breakage.

## Target Architecture

1. Source adapters:
- `src/bot_cripto/data/sentiment_x.py`
- `src/bot_cripto/data/sentiment_telegram.py`

2. NLP scorer:
- `src/bot_cripto/data/sentiment_nlp.py`
- lazy-load transformer pipeline (`ProsusAI/finbert`)
- fallback to lexicon if model/import/runtime unavailable

3. Signal orchestration:
- `src/bot_cripto/data/quant_signals.py`
- `SOCIAL_SENTIMENT_SOURCE=auto` order:
  - `nlp -> api -> x -> telegram -> cryptopanic -> local -> fear_greed`

4. CLI validation:
- `bot-cripto fetch-sentiment --source nlp`
- `bot-cripto fetch-sentiment-nlp`

## Delivery Phases

1. Phase 1 (implemented now)
- Native X/Telegram text extraction.
- NLP scorer module with safe fallback.
- Quant-signal routing supports `nlp`.
- Config/env/documentation updates.

2. Phase 2 (implemented)
- Weighted source blend:
  - `sent_combined = 0.5*x + 0.3*news + 0.2*telegram`
- Automatic reweight when one or more sources are missing.
- Sentiment velocity and EMA smoothing (`SOCIAL_SENTIMENT_EMA_ALPHA`).

3. Phase 3 (next)
- Real-time ingestion with `cryptofeed` for microstructure synchronization.
- Persist synchronized stream snapshots for training/inference parity.

4. Phase 4 (next)
- Retrain and walk-forward validation with updated sentiment features.
- Tune context adjustment thresholds and gating.

## Operational Notes

1. `SOCIAL_SENTIMENT_SOURCE=auto` for production.
2. For explicit testing:
- `bot-cripto fetch-sentiment --symbol BTC/USDT --source x`
- `bot-cripto fetch-sentiment --symbol BTC/USDT --source telegram`
- `bot-cripto fetch-sentiment-nlp --symbol BTC/USDT`

3. If NLP model dependencies are missing, system stays operational and falls back without crashing inference.
