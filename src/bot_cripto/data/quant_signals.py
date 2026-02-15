"""Quantitative signals for sentiment and market context."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests
from filelock import FileLock

from bot_cripto.core.config import Settings
from bot_cripto.core.logging import get_logger

logger = get_logger("data.quant_signals")

# ---------------------------------------------------------------------------
# Simple in-process cache with TTL (avoids hammering APIs every cycle)
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[float, float]] = {}  # key -> (value, expire_ts)
_DEFAULT_TTL = 300.0  # 5 minutes
_REQUEST_TIMEOUT = 5  # aggressive — fail fast, don't block inference


class QuantSignalFetcher:
    """Fetchers for Funding Rates, Fear & Greed, and other Quant signals."""

    def __init__(self, settings: Settings, cache_ttl: float = _DEFAULT_TTL) -> None:
        self.settings = settings
        self.data_dir = settings.data_dir_raw
        self.cache_ttl = cache_ttl

    @staticmethod
    def _cache_get(key: str) -> float | None:
        entry = _cache.get(key)
        if entry is None:
            return None
        value, expire_ts = entry
        if time.monotonic() > expire_ts:
            del _cache[key]
            return None
        return value

    def _cache_set(self, key: str, value: float) -> None:
        _cache[key] = (value, time.monotonic() + self.cache_ttl)

    def fetch_funding_rate(self, symbol: str = "BTC/USDT") -> float:
        """Fetch current funding rate from Binance Futures public API."""
        cache_key = f"funding:{symbol}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            safe_symbol = symbol.replace("/", "")
            url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={safe_symbol}"
            response = requests.get(url, timeout=_REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            rate = float(data.get("lastFundingRate", 0.0))
            logger.info("funding_rate_captured", symbol=symbol, rate=rate)
            self._cache_set(cache_key, rate)
            return rate
        except Exception as exc:
            logger.warning("funding_rate_fetch_failed", error=str(exc))
            return 0.0

    def fetch_fear_and_greed(self) -> float:
        """Fetch Fear & Greed Index (0-100, normalised to 0-1)."""
        cache_key = "fng"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=_REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            value = float(data["data"][0]["value"]) / 100.0
            logger.info("fear_greed_captured", value=value)
            self._cache_set(cache_key, value)
            return value
        except Exception as exc:
            logger.warning("fear_greed_fetch_failed", error=str(exc))
            return 0.5

    def save_signals(self, symbol: str, funding: float, fng: float) -> None:
        """Save signals to a parquet file for merging."""
        path = self.data_dir / f"signals_{symbol.replace('/', '_')}.parquet"
        df = pd.DataFrame({
            "funding_rate": [funding],
            "fear_greed": [fng]
        }, index=[pd.Timestamp.now(tz="UTC")])
        df.index.name = "date"

        lock = FileLock(str(path) + ".lock")
        with lock:
            if path.exists():
                existing = pd.read_parquet(path)
                df = pd.concat([existing, df]).sort_index()
                limit = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
                df = df[df.index >= limit]
            df.to_parquet(path)
