"""Market data adapters for multi-provider ingestion."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

import ccxt
import pandas as pd


class ExchangeAdapter(Protocol):
    name: str

    def parse_timeframe(self, timeframe: str) -> int:
        """Return timeframe duration in seconds."""

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: int,
        limit: int,
    ) -> list[list[float]]:
        """Return OHLCV rows in CCXT format."""


class BinanceAdapter:
    name = "binance"

    def __init__(self) -> None:
        self.client = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
        self.rate_limit_ms = int(self.client.rateLimit)

    def parse_timeframe(self, timeframe: str) -> int:
        return int(self.client.parse_timeframe(timeframe))

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: int,
        limit: int,
    ) -> list[list[float]]:
        rows: list[list[float]] = self.client.fetch_ohlcv(
            symbol,
            timeframe,
            since=since,
            limit=limit,
        )
        return rows


class CoinbaseAdapter:
    name = "coinbase"

    def __init__(self) -> None:
        self.client = ccxt.coinbase({"enableRateLimit": True})
        self.rate_limit_ms = int(self.client.rateLimit)

    def parse_timeframe(self, timeframe: str) -> int:
        return int(self.client.parse_timeframe(timeframe))

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: int,
        limit: int,
    ) -> list[list[float]]:
        return self.client.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)


class KrakenAdapter:
    name = "kraken"

    def __init__(self) -> None:
        self.client = ccxt.kraken({"enableRateLimit": True})
        self.rate_limit_ms = int(self.client.rateLimit)

    def parse_timeframe(self, timeframe: str) -> int:
        return int(self.client.parse_timeframe(timeframe))

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: int,
        limit: int,
    ) -> list[list[float]]:
        return self.client.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)


class OKXAdapter:
    name = "okx"

    def __init__(self) -> None:
        self.client = ccxt.okx({"enableRateLimit": True})
        self.rate_limit_ms = int(self.client.rateLimit)

    def parse_timeframe(self, timeframe: str) -> int:
        return int(self.client.parse_timeframe(timeframe))

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: int,
        limit: int,
    ) -> list[list[float]]:
        return self.client.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)


class YFinanceAdapter:
    name = "yfinance"
    rate_limit_ms = 1000

    @staticmethod
    def _to_yf_symbol(symbol: str) -> str:
        if "/" in symbol:
            base, quote = symbol.split("/", maxsplit=1)
            return f"{base}{quote}=X"
        return symbol

    @staticmethod
    def _parse_interval(timeframe: str) -> str:
        mapping = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "60m", "1d": "1d"}
        if timeframe not in mapping:
            raise ValueError(f"Unsupported yfinance timeframe: {timeframe}")
        return mapping[timeframe]

    @staticmethod
    def parse_timeframe(timeframe: str) -> int:
        if timeframe.endswith("m"):
            return int(timeframe[:-1]) * 60
        if timeframe.endswith("h"):
            return int(timeframe[:-1]) * 3600
        if timeframe.endswith("d"):
            return int(timeframe[:-1]) * 86400
        raise ValueError(f"Invalid timeframe: {timeframe}")

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: int,
        limit: int,
    ) -> list[list[float]]:
        import yfinance as yf  # local import to keep optional dependency

        interval = self._parse_interval(timeframe)
        step_seconds = self.parse_timeframe(timeframe)
        start = datetime.fromtimestamp(since / 1000, tz=UTC)
        end = datetime.fromtimestamp((since / 1000) + limit * step_seconds, tz=UTC)
        ticker = self._to_yf_symbol(symbol)
        frame = yf.download(
            tickers=ticker,
            start=start,
            end=end,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        if frame.empty:
            return []

        if isinstance(frame.columns, pd.MultiIndex):
            frame = frame.xs(ticker, axis=1, level=1, drop_level=True)
        frame = frame.rename(columns=str.lower)
        rows: list[list[float]] = []
        for ts, row in frame.iterrows():
            current_ts = pd.Timestamp(ts)
            if current_ts.tzinfo:
                ts_utc = current_ts.tz_convert("UTC")
            else:
                ts_utc = pd.Timestamp(current_ts, tz="UTC")
            rows.append(
                [
                    int(ts_utc.timestamp() * 1000),
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row.get("volume", 0.0)),
                ]
            )
        return rows


_ADAPTER_REGISTRY: dict[str, type] = {
    "binance": BinanceAdapter,
    "coinbase": CoinbaseAdapter,
    "kraken": KrakenAdapter,
    "okx": OKXAdapter,
    "yfinance": YFinanceAdapter,
    "forex": YFinanceAdapter,
}


def build_adapter(provider: str) -> ExchangeAdapter:
    normalized = provider.lower().strip()
    cls = _ADAPTER_REGISTRY.get(normalized)
    if cls is None:
        raise ValueError(
            f"Unsupported data provider: {provider}. "
            f"Available: {sorted(_ADAPTER_REGISTRY)}"
        )
    return cls()
