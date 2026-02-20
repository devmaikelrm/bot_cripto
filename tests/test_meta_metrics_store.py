from __future__ import annotations

from bot_cripto.monitoring.meta_metrics_store import MetaMetricsStore


def test_meta_metrics_store_append_and_filter(tmp_path) -> None:
    path = tmp_path / "meta_metrics_history.json"
    store = MetaMetricsStore(path)
    store.append(
        {
            "ts": "2026-02-20T00:00:00Z",
            "symbol": "BTC/USDT",
            "timeframe": "5m",
            "val_f1": 0.61,
        }
    )
    store.append(
        {
            "ts": "2026-02-20T01:00:00Z",
            "symbol": "ETH/USDT",
            "timeframe": "1h",
            "val_f1": 0.58,
        }
    )
    all_rows = store.records()
    btc_rows = store.records(symbol="BTC/USDT", timeframe="5m")
    assert len(all_rows) == 2
    assert len(btc_rows) == 1
    assert float(btc_rows[0]["val_f1"]) == 0.61
