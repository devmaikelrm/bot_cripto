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

## 2026-02-19 Precision Roadmap Applied

Status: Completed (phase 1 from `BOT_CRIPTO_MASTER_GUIDE.md`, section 5)

Implemented:
- Real-time context signals in `src/bot_cripto/data/quant_signals.py`:
  - `orderbook_imbalance` (Binance depth snapshot)
  - `social_sentiment` (optional local JSON source)
  - macro context from SP500 (`^GSPC`) and DXY (`DX-Y.NYB`):
    - `sp500_ret_1d`, `dxy_ret_1d`
    - `corr_btc_sp500`, `corr_btc_dxy`
    - `macro_risk_off_score`
- Inference context integration in `src/bot_cripto/jobs/inference.py`:
  - volatility mode override (`CRISIS_HIGH_VOL`) based on realized volatility
  - context-based probability/return/risk adjustment before final decision
  - macro/orderbook buy-gating with configurable thresholds
  - stable signal contract preserved, with added context fields
- New env-backed controls in `src/bot_cripto/core/config.py` and `.env.example`:
  - `MACRO_BLOCK_THRESHOLD`
  - `ORDERBOOK_SELL_WALL_THRESHOLD`
  - `SOCIAL_SENTIMENT_BULL_MIN`
  - `SOCIAL_SENTIMENT_BEAR_MAX`
  - `CONTEXT_PROB_ADJUST_MAX`

Tests:
- `tests/test_inference_context.py` (new)
- Existing smoke: `tests/test_ensemble.py`, `tests/test_risk_engine.py`

## 2026-02-19 Precision Roadmap Applied (phase 2)

Status: Completed

Implemented:
- Social sentiment source routing with fallback order:
  - configurable API endpoint (`SOCIAL_SENTIMENT_ENDPOINT`)
  - CryptoPanic headlines (`CRYPTOPANIC_API_KEY`)
  - local JSON sentiment file
  - neutral/Fear&Greed fallback
- Macro-event crisis windows in inference:
  - `MACRO_EVENT_CRISIS_ENABLED`
  - `MACRO_EVENT_CRISIS_WINDOWS_UTC` (UTC ranges, comma-separated)
  - `MACRO_EVENT_CRISIS_WEEKDAYS` (0=Mon ... 6=Sun)
  - when active, `effective_regime` is forced to `CRISIS_HIGH_VOL`
- Extended signal payload auditing fields:
  - `macro_event_mode`

Tests:
- `tests/test_inference_context.py` updated with macro-window checks.
- `tests/test_config.py` updated for macro-window parsing.

## 2026-02-19 Precision Roadmap Applied (phase 3 - sentiment foundation)

Status: In Progress (phase 1 delivered)

Implemented:
- Native source adapters ready:
  - `src/bot_cripto/data/sentiment_x.py`
  - `src/bot_cripto/data/sentiment_telegram.py`
- NLP scorer foundation:
  - `src/bot_cripto/data/sentiment_nlp.py`
  - lazy `transformers` load with fallback (no hard failure in inference)
- Quant routing upgraded in `src/bot_cripto/data/quant_signals.py`:
  - `SOCIAL_SENTIMENT_SOURCE` now supports `nlp`
  - `auto` order: `nlp -> api -> x -> telegram -> cryptopanic -> local -> fear_greed`
- Validation CLI:
  - `bot-cripto fetch-sentiment-nlp --symbol BTC/USDT`

Config/docs:
- `.env.example`, `README.md`, `BOT_CRIPTO_MASTER_GUIDE.md`
- new implementation document: `docs/SENTIMENT_STACK_IMPLEMENTATION.md`

Tests:
- `tests/test_quant_signals.py` extended to cover NLP source routing.

## 2026-02-19 Precision Roadmap Applied (phase 4 - sentiment blend)

Status: Completed

Implemented:
- Weighted blend in `src/bot_cripto/data/quant_signals.py`:
  - `x/news/telegram` weights configurable
  - automatic reweighting when missing sources
- Temporal stabilization:
  - EMA smoothing and velocity tracking persisted per symbol
  - bundle fields exposed to inference payload:
    - `social_sentiment_raw`
    - `social_sentiment_velocity`
    - `social_sentiment_x`
    - `social_sentiment_news`
    - `social_sentiment_telegram`
- Inference integration in `src/bot_cripto/jobs/inference.py` now reads and stores the sentiment bundle.

Config:
- `SOCIAL_SENTIMENT_WEIGHT_X`
- `SOCIAL_SENTIMENT_WEIGHT_NEWS`
- `SOCIAL_SENTIMENT_WEIGHT_TELEGRAM`
- `SOCIAL_SENTIMENT_EMA_ALPHA`

## Rollback Strategy

If any change degrades behavior:
1. Disable live path with `LIVE_MODE=false`.
2. Set `SPREAD_BPS=0` and `SLIPPAGE_BPS=0` to recover previous PnL assumptions.
3. Remove objective specialization by training all jobs with `BaselineModel(objective="multi")`.
4. Ignore structured performance files and run drift with explicit static history input.
