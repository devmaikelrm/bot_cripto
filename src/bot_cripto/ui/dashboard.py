"""Streamlit Watchtower dashboard (reads SQLite + parquet)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from bot_cripto.core.config import get_settings


def _load_table(db_path: Path, query: str) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql_query(query, conn)
    except sqlite3.DatabaseError:
        return pd.DataFrame()


def _timeframe_to_seconds(timeframe: str) -> int:
    if timeframe.endswith("m"):
        return int(timeframe[:-1]) * 60
    if timeframe.endswith("h"):
        return int(timeframe[:-1]) * 3600
    if timeframe.endswith("d"):
        return int(timeframe[:-1]) * 86400
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def _raw_data_health(raw_path: Path, timeframe: str, target_start: str) -> dict[str, float | str]:
    if not raw_path.exists():
        return {"freshness": "missing", "progress_pct": 0.0, "gap_count": 0.0}
    df = pd.read_parquet(raw_path)
    if df.empty or "timestamp" not in df.columns:
        return {"freshness": "invalid", "progress_pct": 0.0, "gap_count": 0.0}

    ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    latest = ts.max()
    age_min = (datetime.now(tz=UTC) - latest).total_seconds() / 60
    freshness = "green" if age_min <= 5 else "red"

    target_dt = pd.Timestamp(target_start, tz="UTC")
    now = pd.Timestamp(datetime.now(tz=UTC))
    covered = (latest - target_dt).total_seconds()
    total = max((now - target_dt).total_seconds(), 1.0)
    progress_pct = max(0.0, min(100.0, (covered / total) * 100.0))

    step_ms = _timeframe_to_seconds(timeframe) * 1000
    gaps = df["timestamp"].diff().dropna()
    gap_count = float((gaps > step_ms).sum())
    return {"freshness": freshness, "progress_pct": progress_pct, "gap_count": gap_count}


def run() -> None:
    import streamlit as st

    settings = get_settings()
    st.set_page_config(page_title="Bot-Cripto Watchtower", layout="wide")
    st.title("Bot-Cripto Watchtower")
    st.caption("Data Health, ML Ops and Execution Monitoring")

    symbol = st.selectbox("Symbol", settings.symbols_list, index=0)
    safe_symbol = symbol.replace("/", "_")
    raw_path = settings.data_dir_raw / f"{safe_symbol}_{settings.timeframe}.parquet"
    db_path = settings.watchtower_db_path

    health = _raw_data_health(raw_path, settings.timeframe, settings.dashboard_target_start)
    c1, c2, c3 = st.columns(3)
    c1.metric("Data Progress %", f"{health['progress_pct']:.2f}%")
    c2.metric("Detected Gaps", f"{int(health['gap_count'])}")
    c3.metric("Freshness", str(health["freshness"]).upper())

    st.subheader("ML Ops")
    train_df = _load_table(
        db_path,
        (
            "SELECT ts, model_name, metric_name, metric_value "
            "FROM training_metrics ORDER BY id DESC LIMIT 500"
        ),
    )
    if train_df.empty:
        st.info("No training metrics logged yet.")
    else:
        metrics = sorted(train_df["metric_name"].dropna().unique().tolist())
        selected_metric = st.selectbox("Training metric", metrics)
        filtered = train_df[train_df["metric_name"] == selected_metric].copy()
        filtered["ts"] = pd.to_datetime(filtered["ts"], utc=True, errors="coerce")
        chart_df = filtered.sort_values("ts")[["ts", "metric_value"]].set_index("ts")
        st.line_chart(chart_df)

        calib_df = train_df[train_df["metric_name"].isin(["brier_before", "brier_after"])]
        if not calib_df.empty:
            st.markdown("**Calibration Snapshot**")
            pivot = calib_df.pivot_table(
                index="model_name",
                columns="metric_name",
                values="metric_value",
                aggfunc="last",
            )
            st.dataframe(pivot, use_container_width=True)

    st.subheader("Execution")
    equity_df = _load_table(db_path, "SELECT ts, source, equity FROM equity ORDER BY id ASC")
    if equity_df.empty:
        st.info("No equity data logged yet.")
    else:
        equity_df["ts"] = pd.to_datetime(equity_df["ts"], utc=True, errors="coerce")
        st.line_chart(equity_df.set_index("ts")["equity"])
        peak = float(equity_df["equity"].cummax().iloc[-1])
        current = float(equity_df["equity"].iloc[-1])
        drawdown = 0.0 if peak <= 0 else (peak - current) / peak
        if drawdown >= settings.max_daily_drawdown:
            st.error(f"Drawdown Alert: {drawdown:.2%} >= {settings.max_daily_drawdown:.2%}")
        else:
            st.success(f"Drawdown: {drawdown:.2%}")

    decisions_df = _load_table(
        db_path,
        (
            "SELECT ts, symbol, decision, confidence, reason, expected_return, risk_score, "
            "latency_ms "
            "FROM decisions ORDER BY id DESC LIMIT 10"
        ),
    )
    st.markdown("**Latest Decisions (10)**")
    if decisions_df.empty:
        st.info("No decision logs yet.")
    else:
        st.dataframe(decisions_df, use_container_width=True)

    st.subheader("API Health")
    api_df = _load_table(
        db_path,
        (
            "SELECT ts, provider, symbol, timeframe, latency_ms, ok "
            "FROM api_health ORDER BY id DESC LIMIT 20"
        ),
    )
    if api_df.empty:
        st.info("No API health logs yet.")
    else:
        st.dataframe(api_df, use_container_width=True)

    st.subheader("Adaptation Telemetry")
    adaptive_df = _load_table(
        db_path,
        (
            "SELECT ts, event_type, severity, payload_json "
            "FROM adaptive_events ORDER BY id DESC LIMIT 30"
        ),
    )
    if adaptive_df.empty:
        st.info("No adaptive telemetry yet.")
    else:
        st.dataframe(adaptive_df, use_container_width=True)


if __name__ == "__main__":
    run()
