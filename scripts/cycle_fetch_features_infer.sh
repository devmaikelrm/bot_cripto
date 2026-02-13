#!/usr/bin/env bash
set -euo pipefail

# Periodic cycle: fetch small recent window -> features -> inference.
# Intended to run every 5m via systemd timer.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs
lock_file="${LOCK_FILE:-$ROOT_DIR/logs/bot_cripto.lock}"
exec 9>"$lock_file"
if ! flock -n 9; then
  echo "cycle_skip ts=$(date -u +%Y-%m-%dT%H:%M:%SZ) reason=lock_busy lock_file=${lock_file}"
  exit 0
fi

if [[ ! -x ".venv/bin/bot-cripto" ]]; then
  echo "Missing .venv/bin/bot-cripto. Run setup first."
  exit 1
fi

symbols_csv="${SYMBOLS:-BTC/USDT}"
timeframes_csv="${CYCLE_TIMEFRAMES:-${TIMEFRAME:-5m}}"
fetch_days="${FETCH_UPDATE_DAYS:-3}"

IFS=',' read -r -a symbols <<< "$symbols_csv"
IFS=',' read -r -a timeframes <<< "$timeframes_csv"

echo "cycle_start ts=$(date -u +%Y-%m-%dT%H:%M:%SZ) symbols=${symbols_csv} timeframes=${timeframes_csv} fetch_days=${fetch_days}"

for sym in "${symbols[@]}"; do
  sym="$(echo "$sym" | xargs)"
  [[ -z "$sym" ]] && continue
  for tf in "${timeframes[@]}"; do
    tf="$(echo "$tf" | xargs)"
    [[ -z "$tf" ]] && continue
    echo "cycle_pair ts=$(date -u +%Y-%m-%dT%H:%M:%SZ) symbol=${sym} timeframe=${tf}"

    # Update raw parquet (merge is handled by ingestion.save_data()).
    .venv/bin/bot-cripto fetch --days "$fetch_days" --symbol "$sym" --timeframe "$tf"

    # Rebuild features for the whole parquet (simple, robust; can be optimized later).
    .venv/bin/bot-cripto features --symbol "$sym" --timeframe "$tf"

    # Run inference using latest models for this symbol/timeframe.
    .venv/bin/bot-cripto run-inference --symbol "$sym" --timeframe "$tf" >/dev/null
  done
done

echo "cycle_done ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
