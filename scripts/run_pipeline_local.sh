#!/usr/bin/env bash
set -euo pipefail

bot-cripto fetch --days 30
bot-cripto features
bot-cripto train-trend
bot-cripto train-return
bot-cripto train-risk
bot-cripto run-inference

echo "Local pipeline finished"
