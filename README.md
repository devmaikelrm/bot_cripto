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

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env       # Windows: copy .env.example .env
```

Run pipeline:

```bash
bot-cripto fetch --days 30
bot-cripto features
bot-cripto train-trend
bot-cripto train-return
bot-cripto train-risk
bot-cripto run-inference
```

Backtest and drift:

```bash
bot-cripto backtest --folds 4
bot-cripto detect-drift --history-file ./logs/performance_history.json
```

Watchtower dashboard:

```bash
pip install -e ".[ui]"
bot-cripto dashboard --host 0.0.0.0 --port 8501
```

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
- `RISK_PER_TRADE`
- `MAX_DAILY_DRAWDOWN`
- `MAX_WEEKLY_DRAWDOWN`
- `MAX_POSITION_SIZE`
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
