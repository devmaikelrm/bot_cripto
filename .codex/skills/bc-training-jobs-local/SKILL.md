---
name: bc-training-jobs-local
description: Implement local Bot Cripto training jobs for trend, return, and risk models with artifact versioning and metrics. Use when changing training orchestration.
---

# bc-training-jobs-local

1. Maintain job modules under `src/bot_cripto/jobs/`.
2. Keep separate entrypoints for `train_trend`, `train_return`, `train_risk`.
3. Store artifacts under `models/{job}/{symbol}/{timestamp_commit}/`.
4. Log training metrics and keep metadata traceable.
5. Expose CLI wrappers for each job.
