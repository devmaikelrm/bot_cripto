#!/usr/bin/env bash
set -euo pipefail

# Install Bot Cripto systemd user services/timers (no sudo required).
# Uses project in $HOME/bot-cripto.
#
# Note: for timers/services to survive logout, enable lingering once:
#   sudo loginctl enable-linger "$USER"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

user_unit_dir="${HOME}/.config/systemd/user"
mkdir -p "$user_unit_dir"

cp systemd/user/bot-cripto-inference.service "$user_unit_dir/"
cp systemd/user/bot-cripto-inference.timer "$user_unit_dir/"
cp systemd/user/bot-cripto-retrain.service "$user_unit_dir/"
cp systemd/user/bot-cripto-retrain.timer "$user_unit_dir/"
cp systemd/user/bot-cripto-telegram-control.service "$user_unit_dir/"

systemctl --user daemon-reload
systemctl --user enable --now bot-cripto-inference.timer
systemctl --user enable --now bot-cripto-retrain.timer
systemctl --user enable --now bot-cripto-telegram-control.service

echo "Installed systemd user units."
echo "Check timers: systemctl --user list-timers | grep bot-cripto"
echo "Logs: ~/bot-cripto/logs/cycle.log, retrain.log, telegram_control.log"

