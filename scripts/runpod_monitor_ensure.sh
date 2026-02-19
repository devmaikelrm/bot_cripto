#!/usr/bin/env bash
set -euo pipefail

cd /workspace
mkdir -p logs

if pgrep -f "python telegram_monitor.py" >/dev/null 2>&1; then
  exit 0
fi

nohup python telegram_monitor.py --interval 120 --total-epochs 50 \
  > logs/telegram_monitor.log 2>&1 < /dev/null &

