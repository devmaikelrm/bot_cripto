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

## 2026-02-19 Precision Roadmap Applied (phase 5 - realtime stream ingestion)

Status: Completed

Implemented:
- New module `src/bot_cripto/data/streaming.py`:
  - realtime capture via `cryptofeed`
  - automatic fallback to REST polling when unavailable
  - snapshot persistence in `data/raw/stream/{symbol}_stream.parquet`
  - retention policy with file lock protection
- New CLI command:
  - `bot-cripto stream-capture --symbol BTC/USDT --duration 120 --source cryptofeed`
- Optional dependency group:
  - `pip install -e ".[stream]"`

Config:
- `STREAM_SNAPSHOT_INTERVAL_SECONDS`
- `STREAM_ORDERBOOK_DEPTH`
- `STREAM_RETENTION_DAYS`

Tests:
- `tests/test_streaming.py`

## 2026-02-19 Sentiment roadmap refinement

Status: Completed

Implemented:
- Added implementation-focused document:
  - `docs/SENTIMENT_STACK_IMPLEMENTABLE_PLAN.md`
- Keeps `docs/sentiment-stack-mejorado.md` as vision reference and defines concrete next steps aligned to current repo architecture.

## 2026-02-19 Sentiment reliability weighting

Status: Completed

Implemented:
- Source reliability weighting in `src/bot_cripto/data/quant_signals.py`.
- Reliability-aware bundle fields:
  - `social_sentiment_reliability_x`
  - `social_sentiment_reliability_news`
  - `social_sentiment_reliability_telegram`
- Inference wiring persists reliability fields in quant signals output.
- Config controls:
  - `SOCIAL_SENTIMENT_RELIABILITY_ENABLED`
  - `SOCIAL_SENTIMENT_RELIABILITY_MIN_WEIGHT`
  - `SOCIAL_SENTIMENT_RELIABILITY_WINDOW`

Tests:
- `tests/test_quant_signals.py` extended for reliability-weight behavior.
- `tests/test_config.py` defaults updated.

## 2026-02-19 Sentiment anomaly detection

Status: Completed

Implemented:
- Robust anomaly scoring in `src/bot_cripto/data/quant_signals.py`:
  - `social_sentiment_anomaly` in `[0,1]`
  - `social_sentiment_zscore` signed
  - MAD-first z-score with std fallback
- Inference wiring stores anomaly fields through quant payload.
- Config controls:
  - `SOCIAL_SENTIMENT_ANOMALY_WINDOW`
  - `SOCIAL_SENTIMENT_ANOMALY_Z_CLIP`

Tests:
- `tests/test_quant_signals.py` includes anomaly spike detection coverage.

Additional integration:
- context adjustment now includes anomaly penalty in `src/bot_cripto/jobs/inference.py`:
  - reduces effective context score under extreme sentiment spikes
  - increases adjusted risk score when anomaly is high

## 2026-02-19 A/B sentiment backtest command

Status: Completed (initial version)

Implemented:
- New CLI command:
  - `bot-cripto backtest-ab-sentiment --symbol BTC/USDT --timeframe 5m`
- Compares baseline vs context/sentiment-adjusted signal generation under the same realistic execution-cost model.
- Outputs JSON with baseline metrics, with-sentiment metrics, and deltas.

## 2026-02-19 Triple-barrier labeling foundation

Status: Completed

Implemented:
- New labeling module:
  - `src/bot_cripto/labels/triple_barrier.py`
  - triple-barrier labels (`tb_label`) + first touch + return at touch
  - simple purged temporal split helper (`purged_train_test_split`)
- New CLI command:
  - `bot-cripto build-triple-barrier-labels --symbol BTC/USDT --timeframe 5m ...`
- New tests:
  - `tests/test_triple_barrier.py`

Training wiring:
- `train-return` prefers labeled dataset (`*_features_tb.parquet`) and consumes `tb_ret` when present.
- `train-trend` prefers labeled dataset and consumes `tb_label` via baseline trend objective when present; falls back to TFT if TB labels are absent.

## 2026-02-19 Purged temporal CV (phase 1 continuation)

Status: Completed

Implemented:
- New anti-leakage backtesting module:
  - `src/bot_cripto/backtesting/purged_cv.py`
  - contiguous purged K-Fold splits with configurable purge/embargo
  - fold-level and aggregate robustness metrics
- New CLI command:
  - `bot-cripto backtest-purged-cv --splits 5 --purge-size 5 --embargo-size 5`
- Backtesting package exports updated:
  - `src/bot_cripto/backtesting/__init__.py`

Tests:
- `tests/test_backtesting_purged_cv.py`

## 2026-02-19 CPCV-lite robustness validation

Status: Completed

Implemented:
- Combinatorial purged CV helpers:
  - `build_cpcv_splits(...)`
  - `run_cpcv_backtest(...)`
  - file: `src/bot_cripto/backtesting/purged_cv.py`
- New CLI command:
  - `bot-cripto backtest-cpcv --groups 6 --test-groups 2 --purge-size 5 --embargo-size 5`
- Added report distribution fields:
  - `accuracy_mean`
  - `total_net_return_mean`
  - `total_net_return_p5`

Tests:
- `tests/test_backtesting_purged_cv.py` extended for CPCV split count/isolation and CPCV end-to-end run.

## 2026-02-19 Meta-labeling foundation (phase 3 start)

Status: Completed (foundation)

Implemented:
- Hardened meta-model:
  - `src/bot_cripto/models/meta.py`
  - explicit feature schema, success probability output, configurable threshold
  - persisted `meta_config.json` with `min_prob_success`
- New training job:
  - `src/bot_cripto/jobs/train_meta.py`
  - builds out-of-sample meta labels from primary trend predictions
  - uses TB labels when available (`tb_label`), fallback to next-bar realized direction
  - performs temporal holdout validation and auto-tunes `min_prob_success` by best F1
- New CLI command:
  - `bot-cripto train-meta --symbol BTC/USDT --timeframe 5m`
- Inference integration updated:
  - `src/bot_cripto/jobs/inference.py`
  - loads latest versioned meta model (`models/meta/<symbol>/<timeframe>/<version>`)
  - payload adds:
    - `meta_prob_success`
    - `meta_blocked`

Config:
- `META_MODEL_ENABLED` (default `true`)
- `META_MODEL_MIN_PROB_SUCCESS` (default `0.55`)
- `META_MODEL_HOLDOUT_RATIO`
- `META_MODEL_THRESHOLD_MIN`
- `META_MODEL_THRESHOLD_MAX`
- `META_MODEL_THRESHOLD_STEP`
- `META_MODEL_MIN_POSITIVE_PREDICTIONS`

Tests:
- `tests/test_meta_model.py`
- `tests/test_config.py` defaults extended

## 2026-02-19 Risk hardening: CVaR + circuit breaker

Status: Completed

Implemented:
- Extended risk config controls:
  - `RISK_CVAR_ENABLED`
  - `RISK_CVAR_ALPHA`
  - `RISK_CVAR_MIN_SAMPLES`
  - `RISK_CVAR_LIMIT`
  - `RISK_CIRCUIT_BREAKER_MINUTES`
  - `RISK_COOLDOWN_MINUTES`
  - `RISK_ENABLE_KELLY`
  - `RISK_KELLY_FRACTION`
- `src/bot_cripto/risk/engine.py`:
  - historical CVaR(ES) guard over recent returns
  - persistent circuit breaker window when CVaR breaches threshold
  - active circuit breaker blocks new entries until expiry
- `src/bot_cripto/risk/state_store.py`:
  - persist `circuit_breaker_until`
- `src/bot_cripto/jobs/inference.py`:
  - passes `performance_history.json` returns to risk evaluation
  - wires all risk limit settings into `RiskLimits`
  - adds `circuit_breaker_until` into inference payload

Tests:
- `tests/test_risk_engine.py` extended with CVaR breach and active breaker cases
- `tests/test_config.py` defaults extended

## 2026-02-19 Model benchmark block (phase 2 enablement)

Status: Completed

Implemented:
- New benchmark orchestrator:
  - `src/bot_cripto/backtesting/model_benchmark.py`
  - compares multiple model families under same walk-forward config
  - robust status handling: `ok|skipped|error`
  - benchmark summary with winner and deltas vs TFT
- Optional NeuralForecast adapters:
  - `src/bot_cripto/models/neuralforecast_adapter.py`
  - supports `itransformer` and `patchtst` when `neuralforecast` is available
- New CLI command:
  - `bot-cripto benchmark-models --models baseline,tft,nbeats,itransformer,patchtst`
  - outputs ranked JSON results and persists artifact in `logs/benchmark_*.json`

Notes:
- If `neuralforecast` is missing, `itransformer`/`patchtst` are skipped with explicit reason.

Tests:
- `tests/test_model_benchmark.py`

## 2026-02-19 Sentiment coverage completion: RSS news connector

Status: Completed

Implemented:
- New RSS sentiment adapter:
  - `src/bot_cripto/data/sentiment_rss.py`
  - configurable feed list and item limit
  - symbol-aware filtering (`BTC`, `$BTC`, `#BTC`) before scoring
- Quant sentiment routing updated:
  - `src/bot_cripto/data/quant_signals.py`
  - new source `rss` supported in direct mode and `auto` fallback chain
  - news fallback order now: `api -> cryptopanic -> rss -> local`
- CLI source help updated:
  - `bot-cripto fetch-sentiment --source rss`

Config:
- `SOCIAL_SENTIMENT_NEWS_RSS_ENABLED`
- `SOCIAL_SENTIMENT_NEWS_RSS_URLS`
- `SOCIAL_SENTIMENT_NEWS_RSS_MAX_ITEMS`

Tests:
- `tests/test_quant_signals.py` extended for `rss` source and news fallback order.
- `tests/test_sentiment_rss.py`

## 2026-02-19 Threshold recalibration workflow

Status: Completed

Implemented:
- New threshold tuner:
  - `src/bot_cripto/backtesting/threshold_tuner.py`
  - evaluates `PROB_MIN` and `MIN_EXPECTED_RETURN` grids against realized returns
  - ranks candidates by objective combining net return + Sharpe
- New CLI command:
  - `bot-cripto tune-thresholds --symbol BTC/USDT --timeframe 5m`
  - outputs recommended thresholds plus top candidates
- Automatic apply + rollback for `.env`:
  - `--apply-env` on `tune-thresholds` writes `PROB_MIN` and `MIN_EXPECTED_RETURN`
  - creates timestamped backup: `.env.bak.<timestamp>`
  - rollback command: `bot-cripto rollback-thresholds-env [--backup-file ...]`
  - file: `src/bot_cripto/ops/env_tools.py`

Tests:
- `tests/test_threshold_tuner.py`
- `tests/test_env_tools.py`

## 2026-02-19 Compass master checklist + concept drift trigger

Status: Completed

Implemented:
- Execution checklist document to track all Compass phases and pending items:
  - `docs/COMPASS_CHECKLIST_EXECUTION.md`
- Concept drift online detector module:
  - `src/bot_cripto/adaptive/concept_drift.py`
  - ADWIN/PageHinkley via `river` when available
  - robust fallback detector when `river` is unavailable
- Integrated concept-drift trigger into retrain recommendation:
  - `src/bot_cripto/adaptive/online_learner.py`
  - `bot-cripto check-retrain` now evaluates an extra `concept_drift` trigger

Optional dependency:
- `pip install -e ".[online]"` for `river` backend.

Tests:
- `tests/test_concept_drift.py`

## 2026-02-19 Auto-retrain execution path

Status: Completed

Implemented:
- Retrain orchestration helper:
  - `src/bot_cripto/adaptive/retrain_orchestrator.py`
  - standard sequence: `trend -> return -> risk -> meta` (configurable)
- New CLI command:
  - `bot-cripto auto-retrain --symbol BTC/USDT --timeframe 5m --dry-run`
  - evaluates triggers and executes retrain jobs when needed
  - marks retrain timestamp when execution completes successfully
- Checklist tracking updated:
  - `docs/COMPASS_CHECKLIST_EXECUTION.md`

Tests:
- `tests/test_retrain_orchestrator.py`

## 2026-02-19 Champion-Challenger MVP

Status: Completed

Implemented:
- Champion-Challenger evaluator:
  - `src/bot_cripto/adaptive/champion_challenger.py`
  - compares champion vs challenger on a paper/offline evaluation window
  - promotion rule based on relative net-return improvement + min trades
- New CLI command:
  - `bot-cripto champion-challenger-check --model-name trend --symbol BTC/USDT --timeframe 5m`
  - optional `--promote` persists challenger as champion pointer (`champion.txt`)
- Model version listing helper:
  - `src/bot_cripto/jobs/common.py` (`model_version_dirs`)

Config:
- `CC_EVAL_WINDOW`
- `CC_PROMOTION_MARGIN`
- `CC_MIN_TRADES`

Tests:
- `tests/test_champion_challenger.py`

## 2026-02-20 Compass Fase 1 KPI consolidado

Status: Completed

Implemented:
- CPCV Sharpe distribution in backtesting core:
  - `src/bot_cripto/backtesting/purged_cv.py`
  - adds `sharpe` per combination and report-level `sharpe_mean`, `sharpe_p5`
- New Phase 1 KPI consolidation module:
  - `src/bot_cripto/backtesting/phase1_kpi.py`
  - computes:
    - `wf_efficiency = oos_sharpe / is_sharpe`
    - `cpcv_sharpe_mean`
    - `cpcv_sharpe_p5`
  - includes phase pass/fail gates from Compass thresholds
- New CLI command:
  - `bot-cripto phase1-kpi-report --symbol BTC/USDT --timeframe 5m`
  - writes reproducible artifact: `logs/phase1_kpi_<symbol>_<timeframe>_<timestamp>.json`

Tests:
- `tests/test_phase1_kpi.py`
- `tests/test_backtesting_purged_cv.py` (extended for CPCV Sharpe fields)

## 2026-02-20 Compass Fase 2 runner SOTA (automation-ready)

Status: Completed (automation and artifacts). GPU execution pending per environment.

Implemented:
- New Phase 2 orchestration module:
  - `src/bot_cripto/backtesting/phase2_sota.py`
  - trains requested model families, saves artifacts, and computes OOS table:
    - MSE / MAE
    - directional accuracy
    - total net return
    - Sharpe
    - stability proxy
  - computes winner and deltas vs TFT.
- New CLI command:
  - `bot-cripto phase2-sota-run --symbol BTC/USDT --timeframe 5m --models baseline,tft,nbeats,itransformer,patchtst --strict-complete`
  - `--strict-complete` exits non-zero if any model is `skipped/error` (enforces "sin skips").
- New runnable script:
  - `scripts/run_phase2_sota.sh`
  - installs optional forecast deps and launches full Phase 2 run.

Artifacts:
- JSON summary: `logs/phase2_sota_*.json`
- Markdown OOS table: `logs/phase2_sota_*.md`

Tests:
- `tests/test_phase2_sota.py`

## 2026-02-20 Fase 4 HRP MVP + Fase 5 telemetry

Status: Completed (MVP scope)

Implemented:
- HRP allocator MVP:
  - `src/bot_cripto/risk/hrp.py`
  - new CLI command:
    - `bot-cripto hrp-allocate --symbols BTC/USDT,ETH/USDT,SOL/USDT --timeframe 5m --lookback 1000`
  - artifact output:
    - `logs/hrp_allocation_<timeframe>_<timestamp>.json`
- Adaptive telemetry to Watchtower:
  - `src/bot_cripto/monitoring/watchtower_store.py`
    - new table `adaptive_events`
    - new writer `log_adaptive_event(...)`
  - `src/bot_cripto/adaptive/telemetry.py`
  - telemetry wired in CLI flows:
    - `check-retrain`
    - `auto-retrain`
    - `champion-challenger-check`
- Dashboard integration:
  - `src/bot_cripto/ui/dashboard.py`
  - new panel `Adaptation Telemetry` showing latest `adaptive_events`.

Tests:
- `tests/test_hrp.py`
- `tests/test_watchtower_store.py` (extended with `adaptive_events`)

## 2026-02-20 Fase 4 blend allocator + dynamic correlation proxy

Status: Completed

Implemented:
- Blended allocator module:
  - `src/bot_cripto/risk/allocation_blend.py`
  - components:
    - HRP weights
    - Kelly-like weights (long-only edge/variance proxy)
    - views weights (JSON views in `[-1,1]`)
  - dynamic-correlation adjustment:
    - mean absolute correlation monitor
    - shrink toward equal weights when correlation exceeds threshold (proxy DCC behavior)
- New CLI command:
  - `bot-cripto blend-allocate --symbols BTC/USDT,ETH/USDT,SOL/USDT --timeframe 5m --lookback 1000`
  - optional:
    - `--views-file ./views.json`
    - `--w-hrp --w-kelly --w-views`
    - `--corr-threshold --corr-max-shrink`
- Artifact output:
  - `logs/blend_allocation_<timeframe>_<timestamp>.json`

Tests:
- `tests/test_allocation_blend.py`

## 2026-02-20 Fase 3 completion: meta features + CPCV + history tracking

Status: Completed (implementation scope)

Implemented:
- Enriched meta feature stack:
  - `src/bot_cripto/models/meta.py`
  - added derived/context features (regime one-hot, sentiment/macro/orderbook, volatility/ADX, interaction terms)
  - backward-compatible feature alignment via `MetaModel.ensure_feature_columns(...)`.
- Internal CPCV validation for meta-model:
  - `src/bot_cripto/backtesting/meta_cpcv.py`
  - reports fold metrics and aggregate `f1_mean/f1_p5/precision_mean/recall_mean/accuracy_mean`.
- `train-meta` upgraded:
  - `src/bot_cripto/jobs/train_meta.py`
  - computes enriched dataset, runs CPCV validation, stores:
    - model `metadata.metrics` with CPCV summary
    - artifact report `meta_cpcv_report.json` in model version folder
- Historical tracking for degradation/recalibration:
  - `src/bot_cripto/monitoring/meta_metrics_store.py`
  - append-only `logs/meta_metrics_history.json`
  - each run stores `val_f1_delta_prev` plus validation/CPCV metrics.
- New CLI command:
  - `bot-cripto meta-metrics-report --symbol BTC/USDT --timeframe 5m --window 10`

Tests:
- `tests/test_meta_cpcv.py`
- `tests/test_meta_metrics_store.py`

## 2026-02-20 API scaling connectors (Bybit, GNews, Reddit, CoinGecko, Coinpaprika)

Status: Completed (integration scope)

Implemented:
- Market ingestion provider expansion:
  - `src/bot_cripto/data/adapters.py`
  - new provider `bybit` via CCXT (`DATA_PROVIDER=bybit`).
- Sentiment source expansion:
  - `src/bot_cripto/data/sentiment_gnews.py`
  - `src/bot_cripto/data/sentiment_reddit.py`
  - integrated in `src/bot_cripto/data/quant_signals.py`:
    - new sources `gnews` and `reddit`
    - source routing `auto|blend|...|gnews|reddit`.
- News fallback hardening:
  - `quant_signals._fetch_social_sentiment_news` now uses:
    - `endpoint -> gnews -> cryptopanic -> rss -> local`.
- Global market context fallback:
  - `quant_signals.fetch_global_market_context`:
    - primary `CoinGecko /global`
    - fallback `Coinpaprika /global`
  - `fetch_fear_and_greed` falls back to this context proxy when FNG API fails.
- Config & env:
  - `src/bot_cripto/core/config.py`
  - `.env.example`
  - added:
    - `GNEWS_API_KEY`, `GNEWS_MAX_RESULTS`
    - `REDDIT_USER_AGENT`, `REDDIT_MAX_RESULTS`
    - `COINGECKO_API_KEY`, `COINPAPRIKA_API_KEY`

Tests:
- `tests/test_data_adapters.py` (bybit adapter)
- `tests/test_quant_signals.py` (gnews/reddit sources + fallback order)
- `tests/test_config.py` (new defaults)

## 2026-02-20 API smoke validation command

Status: Completed

Implemented:
- Unified connector smoke command:
  - `bot-cripto api-smoke --symbol BTC/USDT --timeframe 5m`
  - file: `src/bot_cripto/ops/api_smoke.py`
  - checks:
    - market providers (`binance`, `bybit`)
    - sentiment sources (`x`, `telegram`, `gnews`, `reddit`, `cryptopanic`, `rss`, `nlp`)
    - context providers (`fear_greed`, `coingecko`, `coinpaprika`)
  - report artifact: `logs/api_smoke_*.json`
- CLI wiring:
  - `src/bot_cripto/cli.py`
- helper script:
  - `scripts/run_api_smoke.sh`

Tests:
- `tests/test_api_smoke.py`

## 2026-02-20 Windows runtime hardening (Python 3.12 default)

Status: Completed

Implemented:
- Windows bootstrap script:
  - `scripts/bootstrap_local_windows.ps1`
  - creates `.venv312`, installs project in editable mode with dev deps.
- Windows command wrapper:
  - `scripts/bot.ps1`
  - runs CLI through `.venv312` by default.
- Windows smoke shortcut:
  - `scripts/run_api_smoke.ps1`
  - executes `api-smoke` through `.venv312`.
- README updated with Windows-recommended flow to avoid Python 3.14/ccxt incompatibility.

## 2026-02-20 Layered architecture scaffolding (crypto + forex)

Status: Completed (scaffolding + routing)

Implemented:
- Market-domain classifier:
  - `src/bot_cripto/core/market.py`
- Data Layer:
  - `src/bot_cripto/data/crypto_feeds.py`
  - `src/bot_cripto/data/forex_feeds.py`
- Feature Layer:
  - `src/bot_cripto/features/layer.py`
- Model Layer:
  - `src/bot_cripto/models/crypto/predictor.py`
  - `src/bot_cripto/models/forex/predictor.py`
- Portfolio Layer:
  - `src/bot_cripto/portfolio/crypto_risk.py`
  - `src/bot_cripto/portfolio/forex_risk.py`
- Allocation Layer:
  - `src/bot_cripto/allocation/capital_allocator.py`
- Global Regime Layer:
  - `src/bot_cripto/regime/global_regime.py`
- Execution Layer:
  - `src/bot_cripto/execution/execution_router.py`
- CLI visibility:
  - `bot-cripto architecture-status --symbols BTC/USDT,EUR/USD`

Tests:
- `tests/test_market_layers.py`

## Rollback Strategy

If any change degrades behavior:
1. Disable live path with `LIVE_MODE=false`.
2. Set `SPREAD_BPS=0` and `SLIPPAGE_BPS=0` to recover previous PnL assumptions.
3. Remove objective specialization by training all jobs with `BaselineModel(objective="multi")`.
4. Ignore structured performance files and run drift with explicit static history input.
