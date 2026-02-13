---
name: bc-qa-documentation
description: Execute Bot Cripto QA workflows and maintain release-grade documentation. Use when validating regressions, improving troubleshooting, or finalizing project docs.
---

# bc-qa-documentation

1. Run quality gates (ruff, black, mypy, pytest).
2. Verify local pipeline script succeeds end-to-end.
3. Validate docker image builds and command contracts.
4. Validate k8s manifests render and apply cleanly.
5. Keep `README.md`, `docs/ARCHITECTURE.md`, and `docs/SECURITY.md` synchronized with actual code.
