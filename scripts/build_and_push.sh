#!/usr/bin/env bash
set -euo pipefail

REGISTRY="${1:-}"
TAG="${2:-latest}"

if [[ -z "${REGISTRY}" ]]; then
  echo "Usage: $0 <registry> [tag]"
  exit 1
fi

docker build -f docker/Dockerfile.train -t "${REGISTRY}/bot-cripto-train:${TAG}" .
docker build -f docker/Dockerfile.infer -t "${REGISTRY}/bot-cripto-infer:${TAG}" .

docker push "${REGISTRY}/bot-cripto-train:${TAG}"
docker push "${REGISTRY}/bot-cripto-infer:${TAG}"

echo "Images pushed"
