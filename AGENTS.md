# AGENTS.md instructions for Bot_cripto

## Skills

### Available project skills

- bc-repo-standards: Repo scaffolding, linting, typing, code quality baseline. (file: ./.codex/skills/bc-repo-standards/SKILL.md)
- bc-data-ingestion-storage: CCXT ingestion, validation, raw storage strategy. (file: ./.codex/skills/bc-data-ingestion-storage/SKILL.md)
- bc-feature-engineering: Technical indicators and feature dataset construction. (file: ./.codex/skills/bc-feature-engineering/SKILL.md)
- bc-model-api-baseline: Base model contract and baseline implementation. (file: ./.codex/skills/bc-model-api-baseline/SKILL.md)
- bc-tft-model: TFT integration with pytorch-forecasting and quantile outputs. (file: ./.codex/skills/bc-tft-model/SKILL.md)
- bc-training-jobs-local: Local training jobs per target model with artifact versioning. (file: ./.codex/skills/bc-training-jobs-local/SKILL.md)
- bc-inference-decision-paper: Inference merge logic, decision engine and paper execution. (file: ./.codex/skills/bc-inference-decision-paper/SKILL.md)
- bc-telegram-notifications: Telegram notifications and pipeline wiring. (file: ./.codex/skills/bc-telegram-notifications/SKILL.md)
- bc-dockerization: Dockerfiles and reproducible runtime images. (file: ./.codex/skills/bc-dockerization/SKILL.md)
- bc-k3s-manifests: k3s manifests: namespace, RBAC, PVC, jobs, cronjobs. (file: ./.codex/skills/bc-k3s-manifests/SKILL.md)
- bc-qa-documentation: QA gates, troubleshooting, final docs and operational scripts. (file: ./.codex/skills/bc-qa-documentation/SKILL.md)
- bc-senior-audit-review: Senior critical review and prioritized remediation report. (file: ./.codex/skills/bc-senior-audit-review/SKILL.md)
- bc-linux-native-ops: Linux native operations with systemd/timers/logrotate (no Kubernetes). (file: ./.codex/skills/bc-linux-native-ops/SKILL.md)
- bc-ensemble-regime-risk: Ensemble + regime + risk stack tuning for inference quality and safety. (file: ./.codex/skills/bc-ensemble-regime-risk/SKILL.md)
- bc-backtesting-drift: Walk-forward backtesting and performance drift detection workflows. (file: ./.codex/skills/bc-backtesting-drift/SKILL.md)
- bc-pipeline-smoke-linux: End-to-end Linux smoke run workflow and regression checks. (file: ./.codex/skills/bc-pipeline-smoke-linux/SKILL.md)
- bc-senior-linux-review: Senior Linux-native production-readiness review with prioritized remediation. (file: ./.codex/skills/bc-senior-linux-review/SKILL.md)
- bc-functional-roadmap: Functional gap analysis and prioritized implementation roadmap for what to build next. (file: ./.codex/skills/bc-functional-roadmap/SKILL.md)

## Trigger rules

- If a user names one of these skills explicitly (for example `$bc-k3s-manifests`) use it in that turn.
- If the task clearly matches one of the skill descriptions, use that skill.
- If multiple skills apply, use the minimum set and state execution order.
