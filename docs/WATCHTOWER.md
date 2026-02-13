# Watchtower Dashboard

`Bot-Cripto Watchtower` is the operational dashboard built with Streamlit.

Run:

```bash
pip install -e ".[ui]"
bot-cripto dashboard --host 0.0.0.0 --port 8501
```

## Sections

1. Data Health
- Download progress percentage vs `DASHBOARD_TARGET_START`.
- Gap heatmap by year/month.
- Freshness status based on last candle age.

2. ML Ops
- Training metric curves from SQLite (`training_metrics`).
- Calibration snapshot (`brier_before` vs `brier_after` when available).

3. Execution
- Equity curve from SQLite (`equity`).
- Last 10 decisions (`decisions` table).
- Drawdown alert vs `MAX_DAILY_DRAWDOWN`.

4. API Health
- Provider latency and success status (`api_health` table).

## Data Sources

- SQLite: `WATCHTOWER_DB_PATH` (default `./logs/watchtower.db`)
- Raw parquet: `data/raw/{symbol}_{timeframe}.parquet`
- Existing `signal.json` and training jobs feed data into the SQLite store.

