---
name: bc-pipeline-smoke-linux
description: Execute and validate Bot Cripto end-to-end Linux-native pipeline smoke runs. Use for quick regression checks after model, feature, or inference changes.
---

# bc-pipeline-smoke-linux

1. Use `scripts/smoke_linux.sh` as the canonical smoke workflow.
2. Expected stages:
   - fetch
   - features
   - train-trend
   - train-return
   - train-risk
   - run-inference
   - backtest
   - optional drift check
3. Support overrides via env:
   - `SYMBOL`
   - `DAYS`
   - `FOLDS`
   - `HISTORY_FILE`
4. Ensure the venv is active before running smoke tests.
5. Treat smoke failure as release blocker until triaged.
