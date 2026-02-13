---
name: bc-ensemble-regime-risk
description: Implement and tune Bot Cripto inference stack that combines ensemble predictions, regime filtering, and risk controls. Use when modifying signal quality, trade gating, and position sizing logic.
---

# bc-ensemble-regime-risk

1. Keep ensemble logic in `src/bot_cripto/models/ensemble.py`.
2. Keep regime detection in `src/bot_cripto/regime/engine.py`.
3. Keep risk controls in `src/bot_cripto/risk/engine.py`.
4. Wire all three in `src/bot_cripto/jobs/inference.py` before final decision output.
5. Respect env-backed controls from `src/bot_cripto/core/config.py`:
   - `REGIME_ADX_TREND_MIN`
   - `REGIME_ATR_HIGH_VOL_PCT`
   - `RISK_PER_TRADE`
   - `MAX_DAILY_DRAWDOWN`
   - `MAX_WEEKLY_DRAWDOWN`
   - `MAX_POSITION_SIZE`
6. Keep `signal.json` contract stable while adding fields.
7. Validate with tests:
   - `tests/test_ensemble.py`
   - `tests/test_regime.py`
   - `tests/test_risk_engine.py`
