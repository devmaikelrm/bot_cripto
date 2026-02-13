---
name: bc-backtesting-drift
description: Run and improve Bot Cripto walk-forward backtesting and performance drift detection. Use when validating temporal robustness, benchmarking model changes, or triggering retraining.
---

# bc-backtesting-drift

1. Keep walk-forward logic in `src/bot_cripto/backtesting/walk_forward.py`.
2. Keep drift detection logic in `src/bot_cripto/monitoring/drift.py`.
3. Use CLI commands:
   - `bot-cripto backtest --folds 4`
   - `bot-cripto detect-drift --history-file ./logs/performance_history.json`
4. Maintain deterministic backtest behavior and temporal splits (no leakage).
5. Update tests when changing behavior:
   - `tests/test_backtesting.py`
   - `tests/test_drift.py`
6. Document metric assumptions in README when thresholds change.
