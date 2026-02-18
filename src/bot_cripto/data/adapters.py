"""Market data adapters for multi-provider ingestion."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol

import pandas as pd

try:
    import ccxt  # type: ignore[import-not-found]

    _CCXT_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on host interpreter/deps
    ccxt = None  # type: ignore[assignment]
    _CCXT_IMPORT_ERROR = exc


def _parse_timeframe_seconds(timeframe: str) -> int:
    if timeframe.endswith("m"):
        return int(timeframe[:-1]) * 60
    if timeframe.endswith("h"):
        return int(timeframe[:-1]) * 3600
    if timeframe.endswith("d"):
        return int(timeframe[:-1]) * 86400
    raise ValueError(f"Invalid timeframe: {timeframe}")


class _UnavailableCCXTClient:
    def __init__(self, exchange_id: str, error: Exception | None) -> None:
        self.exchange_id = exchange_id
        self._error = error
        self.rateLimit = 1000

    def parse_timeframe(self, timeframe: str) -> int:
        return _parse_timeframe_seconds(timeframe)

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, since: int, limit: int
    ) -> list[list[float]]:
        message = (
            f"ccxt unavailable for adapter '{self.exchange_id}'. "
            f"Cannot fetch OHLCV for {symbol} {timeframe}."
        )
        if self._error is not None:
            raise RuntimeError(f"{message} Import error: {self._error}") from self._error
        raise RuntimeError(message)


def _build_ccxt_client(exchange_id: str, options: dict[str, object]) -> object:
    if ccxt is None:
        return _UnavailableCCXTClient(exchange_id, _CCXT_IMPORT_ERROR)
    factory = getattr(ccxt, exchange_id, None)
    if factory is None:
        return _UnavailableCCXTClient(exchange_id, RuntimeError("exchange factory not found"))
    return factory(options)


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
        self.client = _build_ccxt_client(
            "binance", {"enableRateLimit": True, "options": {"defaultType": "spot"}}
        )
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
        self.client = _build_ccxt_client("coinbase", {"enableRateLimit": True})
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
        self.client = _build_ccxt_client("kraken", {"enableRateLimit": True})
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
        self.client = _build_ccxt_client("okx", {"enableRateLimit": True})
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
    _FX_CODES = {
        "USD",
        "EUR",
        "GBP",
        "JPY",
        "CHF",
        "AUD",
        "NZD",
        "CAD",
        "SEK",
        "NOK",
        "DKK",
    }

    @staticmethod
    def _to_yf_symbol(symbol: str) -> str:
        if "/" in symbol:
            base, quote = symbol.split("/", maxsplit=1)
            if base in YFinanceAdapter._FX_CODES and quote in YFinanceAdapter._FX_CODES:
                return f"{base}{quote}=X"
            if quote in {"USD", "USDT"}:
                return f"{base}-USD"
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
        return _parse_timeframe_seconds(timeframe)

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
        # Yahoo intraday data is limited to ~730 days and can reject boundary timestamps.
        if interval != "1d":
            max_lookback = datetime.now(tz=UTC) - timedelta(days=729)
            if start < max_lookback:
                start = max_lookback
        end = datetime.fromtimestamp((since / 1000) + limit * step_seconds, tz=UTC)
        if end <= start:
            return []
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
