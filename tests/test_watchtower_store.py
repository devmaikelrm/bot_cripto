import sqlite3

from bot_cripto.monitoring.watchtower_store import WatchtowerStore


def test_watchtower_store_logs_rows(tmp_path) -> None:
    store = WatchtowerStore(tmp_path / "watchtower.db")
    payload = {
        "ts": "2026-02-12T00:00:00Z",
        "symbol": "BTC/USDT",
        "decision": "LONG",
        "confidence": 0.7,
        "reason": "test",
        "expected_return": 0.01,
        "risk_score": 0.2,
        "regime": "TREND",
        "position_size": 0.5,
    }
    store.log_decision(payload, latency_ms=12.0)
    store.log_equity(ts=payload["ts"], equity=10010.0, source="paper")
    store.log_training_metrics(
        ts=payload["ts"],
        model_name="trend:BTC/USDT",
        metrics={"val_loss": 0.1, "brier_after": 0.2},
    )
    store.log_api_health(
        ts=payload["ts"],
        provider="binance",
        symbol="BTC/USDT",
        timeframe="5m",
        latency_ms=50.0,
        ok=True,
    )
    store.log_adaptive_event(
        ts=payload["ts"],
        event_type="auto_retrain",
        severity="medium",
        payload_json='{"should_retrain": true}',
    )
    with sqlite3.connect(tmp_path / "watchtower.db") as conn:
        decisions = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()
        equity = conn.execute("SELECT COUNT(*) FROM equity").fetchone()
        training = conn.execute("SELECT COUNT(*) FROM training_metrics").fetchone()
        api = conn.execute("SELECT COUNT(*) FROM api_health").fetchone()
        adaptive = conn.execute("SELECT COUNT(*) FROM adaptive_events").fetchone()
    assert decisions is not None and int(decisions[0]) == 1
    assert equity is not None and int(equity[0]) == 1
    assert training is not None and int(training[0]) == 2
    assert api is not None and int(api[0]) == 1
    assert adaptive is not None and int(adaptive[0]) == 1
