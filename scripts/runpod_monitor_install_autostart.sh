#!/usr/bin/env bash
set -euo pipefail

cd /workspace
mkdir -p scripts logs

chmod +x /workspace/scripts/runpod_monitor_ensure.sh || true
chmod +x /workspace/scripts/runpod_monitor_guard.sh || true

# Start one guard process now.
pkill -f "runpod_monitor_guard.sh" >/dev/null 2>&1 || true
nohup /bin/bash /workspace/scripts/runpod_monitor_guard.sh \
  > /workspace/logs/telegram_monitor_guard.log 2>&1 < /dev/null &

# If root opens a shell after restart, it will re-arm the guard automatically.
HOOK='[ -x /workspace/scripts/runpod_monitor_ensure.sh ] && /bin/bash /workspace/scripts/runpod_monitor_ensure.sh >/dev/null 2>&1 || true'
grep -F "$HOOK" /root/.bashrc >/dev/null 2>&1 || echo "$HOOK" >> /root/.bashrc
grep -F "$HOOK" /root/.profile >/dev/null 2>&1 || echo "$HOOK" >> /root/.profile

echo "monitor autostart installed"

