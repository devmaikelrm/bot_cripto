---
name: bc-k3s-manifests
description: Design and maintain Bot Cripto k3s manifests including namespace isolation, RBAC, storage, jobs, and cronjobs. Use for Kubernetes deployment and scheduling changes.
---

# bc-k3s-manifests

1. Keep manifests under `k8s/` in ordered files.
2. Include:
   - namespace and service account
   - namespace-scoped RBAC
   - PVCs for data/models/logs
   - configmap and secret integration
   - jobs and cronjobs
3. Use conservative resources and explicit mounts.
4. Route training jobs to nodes with `role=ml` when available.
5. Keep scripts (`scripts/k8s_apply_all.sh`) aligned with manifest names.
