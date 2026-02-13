#!/usr/bin/env bash
set -euo pipefail

SYMBOL="${SYMBOL:-BTC/USDT}"
DAYS="${DAYS:-30}"
FOLDS="${FOLDS:-4}"
HISTORY_FILE="${HISTORY_FILE:-./logs/performance_history.json}"

if ! command -v bot-cripto >/dev/null 2>&1; then
  echo "bot-cripto command not found. Activate venv first."
  echo "  source .venv/bin/activate"
  exit 1
fi

echo "[1/7] fetch ${SYMBOL} (${DAYS} days)"
bot-cripto fetch --symbol "${SYMBOL}" --days "${DAYS}"

echo "[2/7] features ${SYMBOL}"
bot-cripto features --symbol "${SYMBOL}"

echo "[3/7] train-trend ${SYMBOL}"
bot-cripto train-trend --symbol "${SYMBOL}"

echo "[4/7] train-return ${SYMBOL}"
bot-cripto train-return --symbol "${SYMBOL}"

echo "[5/7] train-risk ${SYMBOL}"
bot-cripto train-risk --symbol "${SYMBOL}"

echo "[6/7] run-inference ${SYMBOL}"
bot-cripto run-inference --symbol "${SYMBOL}"

echo "[7/7] backtest ${SYMBOL} folds=${FOLDS}"
bot-cripto backtest --symbol "${SYMBOL}" --folds "${FOLDS}"

if [[ -f "${HISTORY_FILE}" ]]; then
  echo "[drift] detect-drift using ${HISTORY_FILE}"
  bot-cripto detect-drift --history-file "${HISTORY_FILE}"
else
  echo "[drift] skipped (history file not found: ${HISTORY_FILE})"
fi

echo "Smoke run completed successfully."
