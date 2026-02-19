#!/usr/bin/env bash
set -euo pipefail

cd /workspace
mkdir -p logs

while true; do
  /bin/bash /workspace/scripts/runpod_monitor_ensure.sh || true
  sleep 30
done

