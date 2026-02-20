#!/usr/bin/env bash
set -euo pipefail

# Full Phase 2 SOTA run:
# - enforces no skipped/error models
# - writes JSON + Markdown OOS report under logs/

SYMBOL="${SYMBOL:-BTC/USDT}"
TIMEFRAME="${TIMEFRAME:-5m}"
MODELS="${MODELS:-baseline,tft,nbeats,itransformer,patchtst}"
TRAIN_FRAC="${TRAIN_FRAC:-0.7}"

echo "[phase2] Installing optional SOTA deps (forecast)..."
pip install -e ".[forecast]"

echo "[phase2] Running SOTA train+OOS benchmark"
bot-cripto phase2-sota-run \
  --symbol "${SYMBOL}" \
  --timeframe "${TIMEFRAME}" \
  --models "${MODELS}" \
  --train-frac "${TRAIN_FRAC}" \
  --strict-complete

echo "[phase2] Done. Check logs/phase2_sota_*.json and logs/phase2_sota_*.md"
