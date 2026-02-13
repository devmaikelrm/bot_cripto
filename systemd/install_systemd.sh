#!/usr/bin/env bash
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Run as root."
  exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-/opt/bot-cripto}"
SERVICE_USER="${SERVICE_USER:-botcripto}"
SERVICE_GROUP="${SERVICE_GROUP:-botcripto}"
ENV_DIR="/etc/bot-cripto"
ENV_FILE="${ENV_DIR}/bot-cripto.env"
LOG_DIR="/var/log/bot-cripto"

if ! id -u "${SERVICE_USER}" >/dev/null 2>&1; then
  useradd --system --create-home --shell /usr/sbin/nologin "${SERVICE_USER}"
fi

mkdir -p "${ENV_DIR}" "${LOG_DIR}"
chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${LOG_DIR}"

if [[ ! -f "${ENV_FILE}" ]]; then
  if [[ -f "${PROJECT_DIR}/.env" ]]; then
    cp "${PROJECT_DIR}/.env" "${ENV_FILE}"
  elif [[ -f "${PROJECT_DIR}/.env.example" ]]; then
    cp "${PROJECT_DIR}/.env.example" "${ENV_FILE}"
  fi
fi

cp "${PROJECT_DIR}/systemd/bot-cripto-inference.service" /etc/systemd/system/
cp "${PROJECT_DIR}/systemd/bot-cripto-inference.timer" /etc/systemd/system/
cp "${PROJECT_DIR}/systemd/bot-cripto-retrain.service" /etc/systemd/system/
cp "${PROJECT_DIR}/systemd/bot-cripto-retrain.timer" /etc/systemd/system/
cp "${PROJECT_DIR}/systemd/bot-cripto.logrotate" /etc/logrotate.d/bot-cripto

systemctl daemon-reload
systemctl enable --now bot-cripto-inference.timer
systemctl enable --now bot-cripto-retrain.timer

echo "Installed systemd timers for Bot Cripto"
echo "Edit env file: ${ENV_FILE}"
echo "Check timers: systemctl list-timers | grep bot-cripto"
