#!/usr/bin/env bash
set -euo pipefail

kubectl apply -f k8s/00-namespace-rbac.yaml
kubectl apply -f k8s/01-pvc.yaml
kubectl apply -f k8s/02-configmap.yaml
kubectl apply -f k8s/04-jobs.yaml
kubectl apply -f k8s/05-cronjobs.yaml

echo "Skipped: k8s/03-secret.example.yaml"
echo "Create your real Telegram secret before running jobs."
echo "All Kubernetes manifests applied"
