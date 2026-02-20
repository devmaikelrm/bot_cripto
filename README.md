# Bot Cripto

Modular crypto market prediction system for `BTC/USDT` (`5m`, horizon `5` candles) with local Linux-native workflows.

## Output contract

`signal.json` produced by inference:

```json
{
  "ts": "2026-02-12T17:30:00Z",
  "symbol": "BTC/USDT",
  "timeframe": "5m",
  "horizon_steps": 5,
  "prob_up": 0.67,
  "expected_return": 0.008,
  "p10": -0.004,
  "p50": 0.006,
  "p90": 0.013,
  "risk_score": 0.22,
  "decision": "LONG",
  "confidence": 0.73,
  "reason": "BUY SIGNAL: Prob 67.0% >= 60.0%, Ret 0.80% >= 0.2%",
  "regime": "TREND",
  "position_size": 0.42,
  "risk_allowed": true,
  "version": {
    "git_commit": "abc1234",
    "model_version": "20260212T180000Z_abc1234,20260212T180100Z_abc1234,20260212T180200Z_abc1234"
  }
}
```

## Architecture

`data -> features -> models -> decision -> execution -> notifications`

Layered architecture (multi-market):
- `DATA LAYER`: `data/crypto_feeds.py`, `data/forex_feeds.py`
- `FEATURE LAYER`: `features/layer.py`
- `MODEL LAYER`: `models/crypto/`, `models/forex/`
- `PORTFOLIO LAYER`: `portfolio/crypto_risk.py`, `portfolio/forex_risk.py`
- `ALLOCATION LAYER`: `allocation/capital_allocator.py`
- `GLOBAL REGIME LAYER`: `regime/global_regime.py`
- `EXECUTION LAYER`: `execution/execution_router.py`

- Data: provider adapters (`binance` via CCXT, `yfinance` for forex like `EUR/USD`) + parquet storage.
- Features: RSI, MACD, ATR, rolling volatility, volume features.
- Models: `BasePredictor`, `BaselineModel`, `TFTPredictor`.
- Ensemble: `WeightedEnsemble` for trend/return/risk merge.
- Regime Engine: ADX/ATR filter (`TREND`, `RANGE`, `HIGH_VOL`).
- Risk Engine: dynamic position size + daily/weekly drawdown caps.
- Training: separate jobs `trend`, `return`, `risk`.
- Inference: merge outputs and generate final decision.
- Execution: paper mode implemented, live mode guarded.
- Notifications: Telegram with basic rate limiting.

## Local quickstart

Python soportado: `3.10`, `3.11`, `3.12` (el proyecto bloquea `3.13+`).

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env       # Windows: copy .env.example .env
```

Windows recomendado (evita Python 3.14 para compatibilidad CCXT):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_local_windows.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\bot.ps1 info
```

## Reproducible Clone on Another PC

```bash
git clone https://github.com/devmaikelrm/bot_cripto.git
cd bot_cripto
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev]"
cp .env.example .env       # Windows: copy .env.example .env
pytest -q
```

If `pytest` passes, the environment is ready for training/inference runs.

Run pipeline:

```bash
bot-cripto fetch --days 30
bot-cripto features
bot-cripto train-trend
bot-cripto train-return
bot-cripto train-risk
bot-cripto train-meta
bot-cripto meta-metrics-report --symbol BTC/USDT --timeframe 5m --window 10
bot-cripto run-inference
```

Realtime stream capture:

```bash
pip install -e ".[stream]"
bot-cripto stream-capture --symbol BTC/USDT --duration 120 --source cryptofeed
```

Optional neuralforecast models (iTransformer/PatchTST):

```bash
pip install -e ".[forecast]"
```

Optional online drift detectors (ADWIN/PageHinkley):

```bash
pip install -e ".[online]"
```

Backtest and drift:

```bash
bot-cripto backtest --folds 4
bot-cripto backtest-purged-cv --splits 5 --purge-size 5 --embargo-size 5
bot-cripto backtest-cpcv --groups 6 --test-groups 2 --purge-size 5 --embargo-size 5
bot-cripto phase1-kpi-report --symbol BTC/USDT --timeframe 5m
bot-cripto benchmark-models --models baseline,tft,nbeats,itransformer,patchtst
bot-cripto phase2-sota-run --symbol BTC/USDT --timeframe 5m --models baseline,tft,nbeats,itransformer,patchtst --strict-complete
bot-cripto hrp-allocate --symbols BTC/USDT,ETH/USDT,SOL/USDT --timeframe 5m --lookback 1000
bot-cripto blend-allocate --symbols BTC/USDT,ETH/USDT,SOL/USDT --timeframe 5m --lookback 1000
bot-cripto tune-thresholds --symbol BTC/USDT --timeframe 5m
bot-cripto tune-thresholds --symbol BTC/USDT --timeframe 5m --apply-env
bot-cripto rollback-thresholds-env
bot-cripto detect-drift --history-file ./logs/performance_history.json
bot-cripto auto-retrain --symbol BTC/USDT --timeframe 5m --dry-run
bot-cripto champion-challenger-check --model-name trend --symbol BTC/USDT --timeframe 5m
```

`benchmark-models` also writes an artifact in `logs/benchmark_<symbol>_<timeframe>_<timestamp>.json`
including winner and deltas vs TFT.

`phase2-sota-run` trains each requested model family, saves trained artifacts under
`models/sota_<model>/...`, and writes a final OOS report table in:
- `logs/phase2_sota_<symbol>_<timeframe>_<timestamp>.json`
- `logs/phase2_sota_<symbol>_<timeframe>_<timestamp>.md`

A/B sentiment backtest:

```bash
bot-cripto backtest-ab-sentiment --symbol BTC/USDT --timeframe 5m
```

Sentiment checks:

```bash
bot-cripto fetch-sentiment --symbol BTC/USDT --source x
bot-cripto fetch-sentiment --symbol BTC/USDT --source telegram
bot-cripto fetch-sentiment --symbol BTC/USDT --source gnews
bot-cripto fetch-sentiment --symbol BTC/USDT --source reddit
bot-cripto fetch-sentiment --symbol BTC/USDT --source rss
bot-cripto fetch-sentiment --symbol BTC/USDT --source blend
bot-cripto fetch-sentiment-nlp --symbol BTC/USDT
bot-cripto api-smoke --symbol BTC/USDT --timeframe 5m
bot-cripto architecture-status --symbols BTC/USDT,EUR/USD
```

Windows shortcut for smoke:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_api_smoke.ps1
```

Triple-barrier labeling:

```bash
bot-cripto build-triple-barrier-labels --symbol BTC/USDT --timeframe 5m --pt-mult 2.0 --sl-mult 2.0 --horizon-bars 20
```

Training integration:

- `train-return` now prefers `*_features_tb.parquet` and uses `tb_ret` when available.
- `train-trend` now prefers `*_features_tb.parquet` and uses `tb_label` (baseline trend objective) when available; falls back to TFT if TB labels are missing.
- `backtest-purged-cv` adds anti-leakage temporal CV (purge + embargo) to validate robustness beyond simple walk-forward.
- `backtest-cpcv` adds combinatorial purged CV (CPCV-lite) with distribution metrics (`mean/p5` net return and `mean/p5` Sharpe).
- `phase1-kpi-report` consolidates Compass Phase 1 KPIs (`wf_efficiency`, `cpcv_sharpe_mean`, `cpcv_sharpe_p5`) and writes `logs/phase1_kpi_*.json`.
- `phase2-sota-run` enforces full Phase 2 runs (`--strict-complete`) and generates the final OOS comparison table for SOTA families.
- `hrp-allocate` computes a Phase 4 HRP allocation MVP from aligned multi-asset returns and writes `logs/hrp_allocation_*.json`.
- `blend-allocate` computes Phase 4 blended allocation (`HRP + Kelly proxy + Views`) and applies dynamic-correlation shrink (proxy DCC), writing `logs/blend_allocation_*.json`.
- `train-meta` now generates enriched meta features, runs internal CPCV validation, and writes `meta_cpcv_report.json` in the model artifact.
- `meta-metrics-report` summarizes `logs/meta_metrics_history.json` to monitor meta-model degradation/recalibration trends.

Watchtower dashboard:

```bash
pip install -e ".[ui]"
bot-cripto dashboard --host 0.0.0.0 --port 8501
```

Dashboard now includes **Adaptation Telemetry** (`adaptive_events`) for retrain checks, auto-retrain actions, and champion/challenger promotion decisions.

Linux smoke run:

```bash
source .venv/bin/activate
bash scripts/smoke_linux.sh
```

Optional params:

```bash
SYMBOL="BTC/USDT" DAYS=30 FOLDS=4 bash scripts/smoke_linux.sh
```

Main configurable controls in `.env`:

- `REGIME_ADX_TREND_MIN`
- `REGIME_ATR_HIGH_VOL_PCT`
- `MACRO_EVENT_CRISIS_ENABLED`
- `MACRO_EVENT_CRISIS_WINDOWS_UTC`
- `MACRO_EVENT_CRISIS_WEEKDAYS`
- `MACRO_BLOCK_THRESHOLD`
- `ORDERBOOK_SELL_WALL_THRESHOLD`
- `SOCIAL_SENTIMENT_BULL_MIN`
- `SOCIAL_SENTIMENT_BEAR_MAX`
- `CONTEXT_PROB_ADJUST_MAX`
- `SOCIAL_SENTIMENT_SOURCE`
- `SOCIAL_SENTIMENT_ENDPOINT`
- `CRYPTOPANIC_API_KEY`
- `GNEWS_API_KEY`
- `GNEWS_MAX_RESULTS`
- `REDDIT_USER_AGENT`
- `REDDIT_MAX_RESULTS`
- `COINGECKO_API_KEY`
- `COINPAPRIKA_API_KEY`
- `SOCIAL_SENTIMENT_NLP_ENABLED`
- `SOCIAL_SENTIMENT_NLP_MODEL_ID`
- `SOCIAL_SENTIMENT_NLP_MAX_TEXTS`
- `SOCIAL_SENTIMENT_NEWS_RSS_ENABLED`
- `SOCIAL_SENTIMENT_NEWS_RSS_URLS`
- `SOCIAL_SENTIMENT_NEWS_RSS_MAX_ITEMS`
- `SOCIAL_SENTIMENT_WEIGHT_X`
- `SOCIAL_SENTIMENT_WEIGHT_NEWS`
- `SOCIAL_SENTIMENT_WEIGHT_TELEGRAM`
- `SOCIAL_SENTIMENT_EMA_ALPHA`
- `SOCIAL_SENTIMENT_RELIABILITY_ENABLED`
- `SOCIAL_SENTIMENT_RELIABILITY_MIN_WEIGHT`
- `SOCIAL_SENTIMENT_RELIABILITY_WINDOW`
- `SOCIAL_SENTIMENT_ANOMALY_WINDOW`
- `SOCIAL_SENTIMENT_ANOMALY_Z_CLIP`
- `META_MODEL_ENABLED`
- `META_MODEL_MIN_PROB_SUCCESS`
- `META_MODEL_HOLDOUT_RATIO`
- `META_MODEL_THRESHOLD_MIN`
- `META_MODEL_THRESHOLD_MAX`
- `META_MODEL_THRESHOLD_STEP`
- `META_MODEL_MIN_POSITIVE_PREDICTIONS`
- `X_BEARER_TOKEN`
- `X_QUERY_TEMPLATE`
- `X_MAX_RESULTS`
- `TELEGRAM_SENTIMENT_CHAT_IDS`
- `TELEGRAM_SENTIMENT_LOOKBACK_LIMIT`
- `RISK_PER_TRADE`
- `MAX_DAILY_DRAWDOWN`
- `MAX_WEEKLY_DRAWDOWN`
- `MAX_POSITION_SIZE`
- `RISK_COOLDOWN_MINUTES`
- `RISK_ENABLE_KELLY`
- `RISK_KELLY_FRACTION`
- `RISK_CVAR_ENABLED`
- `RISK_CVAR_ALPHA`
- `RISK_CVAR_MIN_SAMPLES`
- `RISK_CVAR_LIMIT`
- `RISK_CIRCUIT_BREAKER_MINUTES`
- `CC_EVAL_WINDOW`
- `CC_PROMOTION_MARGIN`
- `CC_MIN_TRADES`
- `RISK_SCORE_BLOCK_THRESHOLD`
- `RISK_POSITION_SIZE_MULTIPLIER`
- `MODEL_RISK_VOL_REF`
- `MODEL_RISK_SPREAD_REF`
- `ENABLE_PROBABILITY_CALIBRATION`
- `PROBABILITY_CALIBRATION_METHOD`
- `TFT_CALIBRATION_MAX_SAMPLES`
- `TFT_CALIBRATION_HOLDOUT_RATIO`
- `SPREAD_BPS`
- `SLIPPAGE_BPS`
- `INITIAL_EQUITY`
- `STOP_LOSS_BUFFER`
- `TAKE_PROFIT_BUFFER`
- `HARD_STOP_MAX_LOSS`
- `LIVE_CONFIRM_TOKEN`
- `LIVE_MAX_DAILY_LOSS`
- `DATA_PROVIDER`
- `WATCHTOWER_DB_PATH`
- `DASHBOARD_REFRESH_SECONDS`
- `STREAM_SNAPSHOT_INTERVAL_SECONDS`
- `STREAM_ORDERBOOK_DEPTH`
- `STREAM_RETENTION_DAYS`
- `DASHBOARD_TARGET_START`

Operational state files:

- `logs/risk_state_<symbol>.json`
- `logs/performance_history_<symbol>.json`
- `logs/paper_risk_state.json`
- `logs/performance_history.json`

Implementation details and applied changes:

- `docs/IMPROVEMENTS_APPLIED.md`
- `docs/COMO_FUNCIONA_TODO.md`
- `docs/RUNBOOK_OPERACIONES.md`
- `docs/WATCHTOWER.md`

## Linux native (systemd)

Recommended host layout:

- project: `/opt/bot-cripto`
- env file: `/etc/bot-cripto/bot-cripto.env`
- logs: `/var/log/bot-cripto`

Prepare virtualenv in host:

```bash
cd /opt/bot-cripto
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Install timers/services:

```bash
sudo PROJECT_DIR=/opt/bot-cripto bash /opt/bot-cripto/systemd/install_systemd.sh
```

One-shot deployment (recommended):

```bash
cd /opt/bot-cripto
bash scripts/deploy_linux_native.sh
# if not root, run the printed sudo command for systemd install
```

Check status:

```bash
systemctl status bot-cripto-inference.timer
systemctl status bot-cripto-retrain.timer
systemctl list-timers | grep bot-cripto
```

Manual runs:

```bash
sudo systemctl start bot-cripto-retrain.service
sudo systemctl start bot-cripto-inference.service
```

## Docker quickstart

Build:

```bash
docker build -f docker/Dockerfile.train -t bot-cripto-train:latest .
docker build -f docker/Dockerfile.infer -t bot-cripto-infer:latest .
```

Run one-shot inference:

```bash
docker run --rm --env-file .env -v ${PWD}/data:/mnt/data -v ${PWD}/models:/mnt/models -v ${PWD}/logs:/mnt/logs bot-cripto-infer:latest
```

## Kubernetes (optional)

`k8s/` manifests are kept in the repo but are optional if you run Linux-native with systemd.

## Telegram setup

Set in `.env` or `/etc/bot-cripto/bot-cripto.env`:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Example messages:

- `Starting job: train-trend`
- `Finished job: train-trend status=ok`
- `*Error* in \`inference\``

## Security notes

- Keep `LIVE_MODE=false` until paper trading is validated.
- Never commit `.env` or exchange credentials.
- Restrict permissions of `/etc/bot-cripto/bot-cripto.env` (`chmod 600`).
- Keep service user unprivileged (`botcripto`).

## Quality checks

```bash
ruff check src tests
black --check src tests
mypy src/bot_cripto
pytest tests -v
```
