# Improvements Applied (Linux Native)

Date: 2026-02-12  
Scope: functional roadmap + senior audit hardening (no Kubernetes dependency for runtime)

## Priority Scoring

Formula used:

`priority = user_value*0.30 + risk_reduction*0.30 + dependency_blocker*0.20 + observability_gain*0.10 - complexity*0.10`

## Implementation Queue and Status

1. Persistent risk state across runs  
Status: Completed  
Score: 3.9  
Why: drawdown limits were reset every execution.  
Files:
- `src/bot_cripto/risk/state_store.py`
- `src/bot_cripto/jobs/inference.py`
- `src/bot_cripto/execution/paper.py`
Acceptance:
- Risk state survives process restarts via JSON files in `logs/`.
- Tests: `tests/test_risk_state_store.py`.

2. Separate objective training for trend/return/risk  
Status: Completed  
Score: 3.8  
Why: separate jobs were training the same objective.  
Files:
- `src/bot_cripto/models/baseline.py`
- `src/bot_cripto/jobs/train_trend.py`
- `src/bot_cripto/jobs/train_return.py`
- `src/bot_cripto/jobs/train_risk.py`
Acceptance:
- Training jobs now instantiate objective-specific baseline models.
- Objective metadata persists with model artifacts.

3. Realistic execution costs in paper and walk-forward  
Status: Completed  
Score: 4.0  
Why: no spread/slippage caused optimistic PnL.  
Files:
- `src/bot_cripto/execution/paper.py`
- `src/bot_cripto/backtesting/walk_forward.py`
- `src/bot_cripto/core/config.py`
Acceptance:
- Cost model includes fees + spread + slippage.
- Backtest report now exposes gross and net returns.
- Test update: `tests/test_backtesting.py`.

4. Persistent performance history for drift monitoring  
Status: Completed  
Score: 3.7  
Why: drift detection needed durable history source.  
Files:
- `src/bot_cripto/monitoring/performance_store.py`
- `src/bot_cripto/cli.py`
- `src/bot_cripto/jobs/inference.py`
- `src/bot_cripto/execution/paper.py`
Acceptance:
- Supports structured history (`[{ts, metric}]`) and legacy `list[float]`.
- Tests: `tests/test_performance_store.py`.

5. Live executor guardrails (still safe by default)  
Status: Completed  
Score: 3.6  
Why: live mode needed explicit protections before exchange wiring.  
Files:
- `src/bot_cripto/execution/live.py`
- `src/bot_cripto/core/config.py`
Acceptance:
- Requires `LIVE_MODE=true`.
- Requires `LIVE_CONFIRM_TOKEN=I_UNDERSTAND_LIVE_TRADING`.
- Blocks when daily loss threshold is exceeded.

6. Operational notifications for blocked/active signals  
Status: Completed  
Score: 3.4  
Why: operators need immediate context for inference outputs.  
Files:
- `src/bot_cripto/jobs/inference.py`
Acceptance:
- Sends Telegram messages for `NO_TRADE` and actionable signals when token/chat is configured.

## Milestones

M1 Critical foundation (done):
- persistent risk/performance state
- objective-specific training
- execution/backtest costs

M2 Quality and reliability (done):
- live guardrails
- signal/blocked notifications

M3 Advanced optimization (next):
- calibration layer for `prob_up` and `confidence`
- performance-driven auto-retrain trigger with cooldown windows
- full exchange adapter with order idempotency and reconciliation

## Rollback Strategy

If any change degrades behavior:
1. Disable live path with `LIVE_MODE=false`.
2. Set `SPREAD_BPS=0` and `SLIPPAGE_BPS=0` to recover previous PnL assumptions.
3. Remove objective specialization by training all jobs with `BaselineModel(objective="multi")`.
4. Ignore structured performance files and run drift with explicit static history input.

