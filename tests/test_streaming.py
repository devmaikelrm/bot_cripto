from __future__ import annotations

import pandas as pd

from bot_cripto.core.config import Settings
from bot_cripto.data.streaming import RealtimeStreamCollector, StreamSnapshot, _trade_imbalance


def _settings(tmp_path) -> Settings:
    settings = Settings(
        data_dir_raw=tmp_path / "raw",
        data_dir_processed=tmp_path / "processed",
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
    )
    settings.ensure_dirs()
    return settings


def test_trade_imbalance() -> None:
    assert _trade_imbalance(10.0, 0.0) == 1.0
    assert _trade_imbalance(0.0, 10.0) == -1.0
    assert _trade_imbalance(0.0, 0.0) == 0.0


def test_append_snapshots_persists_rows(tmp_path) -> None:
    settings = _settings(tmp_path)
    collector = RealtimeStreamCollector(settings)
    rows = [
        StreamSnapshot(
            ts="2026-02-19T10:00:00+00:00",
            symbol="BTC/USDT",
            last_price=100000.0,
            bid_volume=120.0,
            ask_volume=100.0,
            orderbook_imbalance=0.0909,
            buy_volume=12.0,
            sell_volume=10.0,
            trade_imbalance=0.0909,
            source="poll",
        ),
        StreamSnapshot(
            ts="2026-02-19T10:00:05+00:00",
            symbol="BTC/USDT",
            last_price=100010.0,
            bid_volume=130.0,
            ask_volume=90.0,
            orderbook_imbalance=0.1818,
            buy_volume=15.0,
            sell_volume=8.0,
            trade_imbalance=0.3043,
            source="poll",
        ),
    ]
    path = collector.append_snapshots("BTC/USDT", rows)
    assert path.exists()
    frame = pd.read_parquet(path)
    assert len(frame) == 2
    assert "orderbook_imbalance" in frame.columns


def test_append_snapshots_retention(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings = settings.model_copy(update={"stream_retention_days": 1})
    collector = RealtimeStreamCollector(settings)
    old = StreamSnapshot(
        ts="2020-01-01T00:00:00+00:00",
        symbol="ETH/USDT",
        last_price=100.0,
        bid_volume=1.0,
        ask_volume=1.0,
        orderbook_imbalance=0.0,
        buy_volume=1.0,
        sell_volume=1.0,
        trade_imbalance=0.0,
        source="poll",
    )
    new = StreamSnapshot(
        ts=pd.Timestamp.now(tz="UTC").isoformat(),
        symbol="ETH/USDT",
        last_price=101.0,
        bid_volume=2.0,
        ask_volume=1.0,
        orderbook_imbalance=0.33,
        buy_volume=2.0,
        sell_volume=1.0,
        trade_imbalance=0.33,
        source="poll",
    )
    path = collector.append_snapshots("ETH/USDT", [old, new])
    frame = pd.read_parquet(path)
    assert len(frame) == 1
