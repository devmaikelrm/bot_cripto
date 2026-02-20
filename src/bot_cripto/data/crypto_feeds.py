"""Crypto feed layer (exchange-first market data access)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from bot_cripto.core.config import Settings
from bot_cripto.data.ingestion import BinanceFetcher


@dataclass(frozen=True)
class CryptoFeedResult:
    provider: str
    rows: int
    path: str


class CryptoFeedLayer:
    """Crypto data layer using configured exchange providers."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def fetch_history(
        self,
        symbol: str,
        timeframe: str,
        days: int,
        providers: list[str] | None = None,
    ) -> CryptoFeedResult:
        candidates = providers or [self.settings.data_provider, "binance", "bybit"]
        last_exc: Exception | None = None
        for provider in candidates:
            try:
                effective = self.settings.model_copy(update={"data_provider": provider, "timeframe": timeframe})
                fetcher = BinanceFetcher(effective)
                df = fetcher.fetch_history(symbol=symbol, timeframe=timeframe, days=days)
                if df.empty:
                    raise ValueError("empty dataset")
                out = fetcher.save_data(df, symbol=symbol, timeframe=timeframe)
                return CryptoFeedResult(provider=provider, rows=len(df), path=str(out))
            except Exception as exc:
                last_exc = exc
                continue
        raise RuntimeError(f"crypto feed failed for {symbol} {timeframe}: {last_exc}")

    def read_local(self, symbol: str, timeframe: str) -> pd.DataFrame:
        path = self.settings.data_dir_raw / f"{symbol.replace('/', '_')}_{timeframe}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing raw dataset: {path}")
        return pd.read_parquet(path)
