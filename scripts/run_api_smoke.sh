#!/usr/bin/env bash
set -euo pipefail

SYMBOL="${SYMBOL:-BTC/USDT}"
TIMEFRAME="${TIMEFRAME:-5m}"

bot-cripto api-smoke --symbol "${SYMBOL}" --timeframe "${TIMEFRAME}"
