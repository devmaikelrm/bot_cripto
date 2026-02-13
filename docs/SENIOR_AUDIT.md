# Senior Audit Report (Skill 12)

Date: 2026-02-12
Scope: Full repository audit (architecture, security, edge cases, performance, config consistency, tests)

## High Severity

1. TFT pipeline is currently broken and fails test/train path.
- Evidence: `src/bot_cripto/models/tft.py:165` (`trainer.fit(self.model, ...)`) fails with current stack (`TypeError: model must be LightningModule`).
- Impact: The "baseline -> TFT" migration path is blocked; production retraining with TFT is non-functional.
- Action:
  - Align to one supported stack (either `lightning.pytorch` + current forecasting version, or compatible pinned versions).
  - Add integration test that executes one full mini-train + predict cycle inside CI.

2. Kubernetes apply script deploys placeholder secret by default.
- Evidence: `scripts/k8s_apply_all.sh:7` applies `k8s/03-secret.example.yaml`.
- Impact: Can overwrite/create invalid credentials (`replace-me`) in cluster; breaks alerts and risks unsafe operational behavior.
- Action:
  - Remove example secret from automated apply flow.
  - Require explicit creation command from operator (`kubectl create secret ...`).

3. Runtime decision thresholds are inconsistent with documented env contract.
- Evidence:
  - `src/bot_cripto/decision/engine.py:49-51` uses `strategy_*` fields.
  - `.env.example:31-34` defines `RISK_MAX`, `PROB_MIN`, `MIN_EXPECTED_RETURN`.
  - `src/bot_cripto/core/config.py:48-51` duplicates threshold set (`strategy_*`) while `risk_max/prob_min/min_expected_return` also exist in `:68-70`.
- Impact: Operators changing `.env` decision values may see no effect; silent config drift.
- Action:
  - Use one canonical set in code and docs.
  - Prefer `risk_max/prob_min/min_expected_return` in decision engine.

## Medium Severity

1. RBAC is over-privileged for this workload.
- Evidence: `k8s/00-namespace-rbac.yaml:19` includes `secrets` read permissions.
- Impact: If pod compromised, secret enumeration risk increases.
- Action:
  - Remove `secrets` from Role resources unless runtime k8s API access is required.
  - Since env injection is handled by kubelet, most jobs need no API access.

2. Image tags use `latest` in Jobs/CronJobs.
- Evidence: `k8s/04-jobs.yaml:15,51,87,123,159`, `k8s/05-cronjobs.yaml:20,59`.
- Impact: Non-deterministic deployments, rollback difficulty.
- Action:
  - Pin immutable tags or digests (`:vX.Y.Z` or `@sha256:...`).

3. StorageClass is hardcoded and not configurable at deploy-time.
- Evidence: `k8s/01-pvc.yaml:11,23,35` fixed to `local-path`.
- Impact: Breaks portability across clusters with different StorageClass names.
- Action:
  - Template or patch via kustomize/helm/envsubst.

4. Training jobs for trend/return/risk currently train identical model objective.
- Evidence:
  - `src/bot_cripto/jobs/train_trend.py:17`
  - `src/bot_cripto/jobs/train_return.py:17`
  - `src/bot_cripto/jobs/train_risk.py:17`
  all call same baseline train on `target_col="close"`.
- Impact: Separation by responsibility exists only by folder name, not objective specialization.
- Action:
  - Split target engineering and objective per job (classification for trend, regression for return, risk target for drawdown/vol).

5. Full strict quality gate is red (ruff/mypy).
- Evidence:
  - Ruff: 199 issues across `src/` and `tests`.
  - Mypy: 21 errors (`src/bot_cripto/models/tft.py`, `src/bot_cripto/jobs/inference.py`, `src/bot_cripto/features/engineering.py`, etc.).
- Impact: Regression risk and lower maintainability.
- Action:
  - Stabilize typing on core runtime path first (`jobs/`, `models/`, `decision/`).
  - Then normalize formatting/lint debt.

## Low Severity

1. Encoding artifacts reduce readability in comments/docs.
- Evidence: mojibake in several files (for example `src/bot_cripto/core/config.py:1-5`).
- Impact: Not functional, but harms maintainability.
- Action: Normalize to UTF-8 and avoid mixed encoding edits.

2. Signal version metadata is not traceable to real artifact.
- Evidence: `src/bot_cripto/jobs/inference.py:71-72` uses `runtime/latest` literals.
- Impact: Auditability and incident analysis are weaker.
- Action: Emit actual model directory/version and commit hash.

## Prioritized Remediation Plan

1. Fix TFT compatibility and add CI smoke train/predict test.
2. Stop applying example secret automatically.
3. Unify decision threshold config contract and wire env keys actually used in runtime.
4. Reduce RBAC scope and pin immutable images.
5. Make StorageClass configurable and separate model objectives by job.
6. Clean ruff/mypy debt and encoding issues.

## Safe Diff Suggestions (non-breaking)

1. `scripts/k8s_apply_all.sh`
```diff
- kubectl apply -f k8s/03-secret.example.yaml
+ echo "Skipping example secret. Create real secret explicitly before deploy."
```

2. `src/bot_cripto/decision/engine.py`
```diff
- prob_thresh = self.settings.strategy_prob_threshold
- min_return = self.settings.strategy_min_return
- max_risk = self.settings.strategy_max_risk_score
+ prob_thresh = self.settings.prob_min
+ min_return = self.settings.min_expected_return
+ max_risk = self.settings.risk_max
```

3. `k8s/00-namespace-rbac.yaml`
```diff
- resources: ["pods", "pods/log", "configmaps", "secrets", "persistentvolumeclaims"]
+ resources: ["pods", "pods/log", "configmaps", "persistentvolumeclaims"]
```

4. `k8s/04-jobs.yaml`, `k8s/05-cronjobs.yaml`
```diff
- image: your-registry/bot-cripto-train:latest
+ image: your-registry/bot-cripto-train:v0.1.0
```

5. `src/bot_cripto/jobs/inference.py`
```diff
- "git_commit": "runtime",
- "model_version": "latest",
+ "git_commit": resolved_commit,
+ "model_version": resolved_model_version,
```

## Validation Snapshot

- Tests: 34 passed / 1 failed (`tests/test_tft.py::test_train_smoke`).
- Lint: failing (199 issues).
- Type check: failing (21 errors).

## Remediation Update (2026-02-12)

Completed:
- P1 TFT compatibility fixed in `src/bot_cripto/models/tft.py` (trainer import alignment + checkpoint load + predict bug fix).
- P2 secret apply hardening in `scripts/k8s_apply_all.sh` (example secret no longer auto-applied).
- P3 decision config consistency fixed in `src/bot_cripto/decision/engine.py` and aligned test fixture in `tests/test_decision.py`.

Validation after remediation:
- `pytest`: 35 passed.
- `ruff`: still failing with formatting/lint debt.
- `mypy`: still failing with typing debt.
