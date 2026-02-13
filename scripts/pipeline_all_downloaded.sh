#!/usr/bin/env bash
set -euo pipefail

# Runs features + train(trend/return/risk) + inference for every raw parquet file.
# Intended for VPS automation.
#
# Usage:
#   cd ~/bot-cripto
#   bash scripts/pipeline_all_downloaded.sh
#
# Logs:
#   logs/pipeline_<symbol>_<tf>.log

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ".venv/bin/bot-cripto" ]]; then
  echo "Missing .venv/bin/bot-cripto. Create venv and pip install -e . first."
  exit 1
fi

raw_dir="data/raw"
proc_dir="data/processed"
mkdir -p logs "$proc_dir"

shopt -s nullglob
raw_files=("$raw_dir"/*.parquet)
if [[ ${#raw_files[@]} -eq 0 ]]; then
  echo "No raw parquet files in $raw_dir"
  exit 2
fi

run_one() {
  local raw_path="$1"
  local base
  base="$(basename "$raw_path" .parquet)"   # e.g. BTC_USDT_5m

  local tf="${base##*_}"                   # 5m
  local sym_part="${base%_*}"              # BTC_USDT
  local base_asset="${sym_part%%_*}"       # BTC
  local quote_asset="${sym_part#*_}"       # USDT
  local symbol="${base_asset}/${quote_asset}"

  local safe="${base_asset}_${quote_asset}_${tf}"
  local log_file="logs/pipeline_${safe}.log"

  echo "=== PIPELINE ${symbol} ${tf} ===" | tee "$log_file"
  echo "ts=$(date -u +%Y-%m-%dT%H:%M:%SZ) raw=$raw_path" | tee -a "$log_file"

  set +e
  .venv/bin/bot-cripto features --symbol "$symbol" --timeframe "$tf" | tee -a "$log_file"
  local rc_feat=${PIPESTATUS[0]}
  if [[ $rc_feat -ne 0 ]]; then
    echo "ERROR features rc=$rc_feat" | tee -a "$log_file"
    set -e
    return 0
  fi

  .venv/bin/bot-cripto train-trend --symbol "$symbol" --timeframe "$tf" | tee -a "$log_file"
  local rc_trend=${PIPESTATUS[0]}
  .venv/bin/bot-cripto train-return --symbol "$symbol" --timeframe "$tf" | tee -a "$log_file"
  local rc_ret=${PIPESTATUS[0]}
  .venv/bin/bot-cripto train-risk --symbol "$symbol" --timeframe "$tf" | tee -a "$log_file"
  local rc_risk=${PIPESTATUS[0]}
  .venv/bin/bot-cripto run-inference --symbol "$symbol" --timeframe "$tf" | tee -a "$log_file"
  local rc_inf=${PIPESTATUS[0]}
  set -e

  if [[ $rc_trend -ne 0 || $rc_ret -ne 0 || $rc_risk -ne 0 || $rc_inf -ne 0 ]]; then
    echo "ERROR train/infer rc_trend=$rc_trend rc_return=$rc_ret rc_risk=$rc_risk rc_infer=$rc_inf" | tee -a "$log_file"
    return 0
  fi

  echo "DONE ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$log_file"
}

for f in "${raw_files[@]}"; do
  run_one "$f"
done

echo "ALL_DONE ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
