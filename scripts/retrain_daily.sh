#!/usr/bin/env bash
set -euo pipefail

# Daily retrain for configured symbols/timeframes.
# Intended to run via systemd timer (e.g. once per day at 02:00).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs
lock_file="${LOCK_FILE:-$ROOT_DIR/logs/bot_cripto.lock}"
exec 9>"$lock_file"
flock 9

if [[ ! -x ".venv/bin/bot-cripto" ]]; then
  echo "Missing .venv/bin/bot-cripto. Run setup first."
  exit 1
fi

symbols_csv="${RETRAIN_SYMBOLS:-${SYMBOLS:-BTC/USDT}}"
timeframes_csv="${RETRAIN_TIMEFRAMES:-${TIMEFRAME:-5m}}"
fetch_days="${RETRAIN_FETCH_DAYS:-30}"

IFS=',' read -r -a symbols <<< "$symbols_csv"
IFS=',' read -r -a timeframes <<< "$timeframes_csv"

echo "retrain_start ts=$(date -u +%Y-%m-%dT%H:%M:%SZ) symbols=${symbols_csv} timeframes=${timeframes_csv} fetch_days=${fetch_days}"

# Download Macro Data (SPY, DXY) since 2017
.venv/bin/bot-cripto fetch-macro --days 3300

for sym in "${symbols[@]}"; do
  sym="$(echo "$sym" | xargs)"
  [[ -z "$sym" ]] && continue
  for tf in "${timeframes[@]}"; do
    tf="$(echo "$tf" | xargs)"
    [[ -z "$tf" ]] && continue
    echo "retrain_pair ts=$(date -u +%Y-%m-%dT%H:%M:%SZ) symbol=${sym} timeframe=${tf}"

    if [[ "${fetch_days}" != "0" ]]; then
      .venv/bin/bot-cripto fetch --days "$fetch_days" --symbol "$sym" --timeframe "$tf"
    fi

    .venv/bin/bot-cripto features --symbol "$sym" --timeframe "$tf"
    .venv/bin/bot-cripto train-trend --symbol "$sym" --timeframe "$tf"
    .venv/bin/bot-cripto train-return --symbol "$sym" --timeframe "$tf"
    .venv/bin/bot-cripto train-risk --symbol "$sym" --timeframe "$tf"
    # Meta-model must run AFTER primary models so it can use their outputs as features
    .venv/bin/bot-cripto train-meta --symbol "$sym" --timeframe "$tf"
  done
done

echo "retrain_done ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
