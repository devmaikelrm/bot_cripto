"""Quantitative signals for sentiment and market context."""

from __future__ import annotations

import requests
import pandas as pd
from filelock import FileLock
from pathlib import Path
from bot_cripto.core.config import Settings
from bot_cripto.core.logging import get_logger

logger = get_logger("data.quant_signals")

class QuantSignalFetcher:
    """Fetchers for Funding Rates, Fear & Greed, and other Quant signals."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.data_dir = settings.data_dir_raw

    def fetch_funding_rate(self, symbol: str = "BTC/USDT") -> float:
        """Fetch current funding rate from Binance Futures via CCXT (or public API)."""
        try:
            # Usamos la API pública de Binance para evitar autenticación de futuros si no es necesaria
            safe_symbol = symbol.replace("/", "")
            url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={safe_symbol}"
            response = requests.get(url, timeout=10)
            data = response.json()
            rate = float(data.get("lastFundingRate", 0.0))
            logger.info("funding_rate_captured", symbol=symbol, rate=rate)
            return rate
        except Exception as exc:
            logger.error("funding_rate_fetch_failed", error=str(exc))
            return 0.0

    def fetch_fear_and_greed(self) -> float:
        """Fetch Fear & Greed Index (0-100)."""
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=10)
            data = response.json()
            value = float(data["data"][0]["value"])
            logger.info("fear_greed_captured", value=value)
            return value / 100.0  # Normalizado 0-1
        except Exception as exc:
            logger.error("fear_greed_fetch_failed", error=str(exc))
            return 0.5  # Neutro

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
                # Mantener solo los últimos 7 días de señales de alta frecuencia
                limit = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
                df = df[df.index >= limit]
            df.to_parquet(path)
