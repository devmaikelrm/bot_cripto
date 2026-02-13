#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/opt/bot-cripto}"
SERVICE_USER="${SERVICE_USER:-botcripto}"
SERVICE_GROUP="${SERVICE_GROUP:-botcripto}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "Project directory not found: ${PROJECT_DIR}"
  exit 1
fi

cd "${PROJECT_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python binary not found: ${PYTHON_BIN}"
  exit 1
fi

if [[ ! -d ".venv" ]]; then
  "${PYTHON_BIN}" -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

if [[ ! -f ".env" && -f ".env.example" ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

mkdir -p data/raw data/processed models logs

if [[ $EUID -eq 0 ]]; then
  PROJECT_DIR="${PROJECT_DIR}" SERVICE_USER="${SERVICE_USER}" SERVICE_GROUP="${SERVICE_GROUP}" \
    bash "${PROJECT_DIR}/systemd/install_systemd.sh"
else
  echo "Run the following as root to install systemd units:"
  echo "  sudo PROJECT_DIR=${PROJECT_DIR} SERVICE_USER=${SERVICE_USER} SERVICE_GROUP=${SERVICE_GROUP} bash ${PROJECT_DIR}/systemd/install_systemd.sh"
fi

echo "Local setup complete."
echo "Edit config at: ${PROJECT_DIR}/.env"
echo "If systemd installed, env used by services: /etc/bot-cripto/bot-cripto.env"
