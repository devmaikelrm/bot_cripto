"""Realtime microstructure streaming ingestion."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from filelock import FileLock

from bot_cripto.core.config import Settings
from bot_cripto.core.logging import get_logger
from bot_cripto.data.adapters import build_adapter

logger = get_logger("data.streaming")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _trade_imbalance(buy_volume: float, sell_volume: float) -> float:
    total = buy_volume + sell_volume
    if total <= 0:
        return 0.0
    return float((buy_volume - sell_volume) / total)


@dataclass
class StreamSnapshot:
    ts: str
    symbol: str
    last_price: float
    bid_volume: float
    ask_volume: float
    orderbook_imbalance: float
    buy_volume: float
    sell_volume: float
    trade_imbalance: float
    source: str


class RealtimeStreamCollector:
    """Capture and persist realtime microstructure snapshots."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.stream_dir = settings.data_dir_raw / "stream"
        self.stream_dir.mkdir(parents=True, exist_ok=True)

    def stream_path(self, symbol: str) -> Path:
        safe_symbol = symbol.replace("/", "_")
        return self.stream_dir / f"{safe_symbol}_stream.parquet"

    def append_snapshots(self, symbol: str, rows: Iterable[StreamSnapshot]) -> Path:
        path = self.stream_path(symbol)
        frame = pd.DataFrame([row.__dict__ for row in rows])
        if frame.empty:
            return path
        frame["date"] = pd.to_datetime(frame["ts"], utc=True, format="mixed")
        frame = frame.set_index("date").drop(columns=["ts"])
        frame.index.name = "date"

        lock = FileLock(str(path) + ".lock")
        with lock:
            if path.exists():
                existing = pd.read_parquet(path)
                if not existing.empty:
                    frame = pd.concat([existing, frame]).sort_index()
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=self.settings.stream_retention_days)
            frame = frame[frame.index >= cutoff]
            frame.to_parquet(path)
        return path

    def capture(
        self,
        symbol: str,
        duration_seconds: int = 60,
        source: str = "cryptofeed",
        snapshot_every_seconds: int | None = None,
    ) -> Path:
        src = source.lower().strip()
        every = snapshot_every_seconds or self.settings.stream_snapshot_interval_seconds
        if src == "cryptofeed":
            try:
                return self.capture_cryptofeed(
                    symbol=symbol,
                    duration_seconds=duration_seconds,
                    snapshot_every_seconds=every,
                )
            except Exception as exc:
                logger.warning("cryptofeed_capture_failed_fallback_poll", symbol=symbol, error=str(exc))
                return self.capture_polling(
                    symbol=symbol,
                    duration_seconds=duration_seconds,
                    snapshot_every_seconds=every,
                )
        return self.capture_polling(
            symbol=symbol,
            duration_seconds=duration_seconds,
            snapshot_every_seconds=every,
        )

    def capture_polling(
        self,
        symbol: str,
        duration_seconds: int = 60,
        snapshot_every_seconds: int = 5,
    ) -> Path:
        """Fallback mode using exchange REST polling."""
        adapter = build_adapter(self.settings.data_provider)
        exchange = getattr(adapter, "client", adapter)
        start = time.time()
        rows: list[StreamSnapshot] = []
        while time.time() - start < duration_seconds:
            ob = exchange.fetch_order_book(symbol, limit=self.settings.stream_orderbook_depth)
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])
            bid_volume = float(sum(_safe_float(level[1]) for level in bids if len(level) >= 2))
            ask_volume = float(sum(_safe_float(level[1]) for level in asks if len(level) >= 2))
            ob_imb = 0.0
            total = bid_volume + ask_volume
            if total > 0:
                ob_imb = (bid_volume - ask_volume) / total

            trades = exchange.fetch_trades(symbol, limit=100)
            buy_volume = 0.0
            sell_volume = 0.0
            last_price = 0.0
            for trade in trades:
                amount = _safe_float(trade.get("amount"))
                side = str(trade.get("side", "")).lower()
                last_price = _safe_float(trade.get("price"), last_price)
                if side == "buy":
                    buy_volume += amount
                elif side == "sell":
                    sell_volume += amount

            rows.append(
                StreamSnapshot(
                    ts=pd.Timestamp.now(tz="UTC").isoformat(),
                    symbol=symbol,
                    last_price=last_price,
                    bid_volume=float(bid_volume),
                    ask_volume=float(ask_volume),
                    orderbook_imbalance=float(ob_imb),
                    buy_volume=float(buy_volume),
                    sell_volume=float(sell_volume),
                    trade_imbalance=_trade_imbalance(buy_volume, sell_volume),
                    source="poll",
                )
            )
            time.sleep(snapshot_every_seconds)
        path = self.append_snapshots(symbol=symbol, rows=rows)
        logger.info("stream_capture_done", symbol=symbol, source="poll", rows=len(rows), path=str(path))
        return path

    def capture_cryptofeed(
        self,
        symbol: str,
        duration_seconds: int = 60,
        snapshot_every_seconds: int = 5,
    ) -> Path:
        """Realtime capture via cryptofeed websocket channels."""
        try:
            from cryptofeed import FeedHandler
            from cryptofeed.defines import L2_BOOK, TRADES
            from cryptofeed.exchanges import Binance
        except Exception as exc:  # pragma: no cover - depends on optional deps
            raise RuntimeError(
                "cryptofeed is not installed. Install with: pip install -e \".[stream]\""
            ) from exc

        state_lock = threading.Lock()
        state: dict[str, float] = {
            "bid_volume": 0.0,
            "ask_volume": 0.0,
            "buy_volume": 0.0,
            "sell_volume": 0.0,
            "last_price": 0.0,
        }
        fh_container: dict[str, Any] = {}
        rows: list[StreamSnapshot] = []

        def _book_cb(book: Any, receipt_timestamp: float) -> None:
            with state_lock:
                bids = self._extract_levels(getattr(book, "book", None), side="bids")
                asks = self._extract_levels(getattr(book, "book", None), side="asks")
                state["bid_volume"] = float(sum(q for _, q in bids[: self.settings.stream_orderbook_depth]))
                state["ask_volume"] = float(sum(q for _, q in asks[: self.settings.stream_orderbook_depth]))

        def _trade_cb(trade: Any, receipt_timestamp: float) -> None:
            with state_lock:
                amount = _safe_float(getattr(trade, "amount", 0.0))
                side = str(getattr(trade, "side", "")).lower()
                price = _safe_float(getattr(trade, "price", 0.0))
                if price > 0:
                    state["last_price"] = price
                if side == "buy":
                    state["buy_volume"] += amount
                elif side == "sell":
                    state["sell_volume"] += amount

        def _run_feed() -> None:
            fh = FeedHandler()
            fh_container["fh"] = fh
            feed_symbol = symbol.replace("/", "-")
            fh.add_feed(
                Binance(
                    symbols=[feed_symbol],
                    channels=[TRADES, L2_BOOK],
                    callbacks={TRADES: _trade_cb, L2_BOOK: _book_cb},
                )
            )
            fh.run()

        thread = threading.Thread(target=_run_feed, daemon=True)
        thread.start()
        start = time.time()
        next_tick = start + snapshot_every_seconds
        try:
            while time.time() - start < duration_seconds:
                time.sleep(0.2)
                if time.time() < next_tick:
                    continue
                with state_lock:
                    bid_volume = float(state["bid_volume"])
                    ask_volume = float(state["ask_volume"])
                    buy_volume = float(state["buy_volume"])
                    sell_volume = float(state["sell_volume"])
                    last_price = float(state["last_price"])
                    state["buy_volume"] = 0.0
                    state["sell_volume"] = 0.0
                ob_total = bid_volume + ask_volume
                ob_imb = 0.0 if ob_total <= 0 else (bid_volume - ask_volume) / ob_total
                rows.append(
                    StreamSnapshot(
                        ts=pd.Timestamp.now(tz="UTC").isoformat(),
                        symbol=symbol,
                        last_price=last_price,
                        bid_volume=bid_volume,
                        ask_volume=ask_volume,
                        orderbook_imbalance=float(ob_imb),
                        buy_volume=buy_volume,
                        sell_volume=sell_volume,
                        trade_imbalance=_trade_imbalance(buy_volume, sell_volume),
                        source="cryptofeed",
                    )
                )
                next_tick += snapshot_every_seconds
        finally:
            fh = fh_container.get("fh")
            if fh is not None and hasattr(fh, "stop"):
                try:
                    fh.stop()
                except Exception:
                    pass
        path = self.append_snapshots(symbol=symbol, rows=rows)
        logger.info("stream_capture_done", symbol=symbol, source="cryptofeed", rows=len(rows), path=str(path))
        return path

    @staticmethod
    def _extract_levels(book: Any, side: str) -> list[tuple[float, float]]:
        if book is None:
            return []
        side_data = None
        if isinstance(book, dict):
            side_data = book.get(side) or book.get(side.rstrip("s"))
        else:
            side_data = getattr(book, side, None)
            if side_data is None:
                side_data = getattr(book, side.rstrip("s"), None)
        if side_data is None:
            return []
        if isinstance(side_data, dict):
            out = [(_safe_float(p), _safe_float(q)) for p, q in side_data.items()]
            out.sort(key=lambda x: x[0], reverse=(side == "bids"))
            return out
        if isinstance(side_data, list):
            out2: list[tuple[float, float]] = []
            for item in side_data:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    out2.append((_safe_float(item[0]), _safe_float(item[1])))
            return out2
        return []
