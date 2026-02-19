"""Backward-compatible sentiment fetcher wrapper."""

from __future__ import annotations

import pandas as pd
from filelock import FileLock

from bot_cripto.core.config import Settings
from bot_cripto.data.quant_signals import QuantSignalFetcher


class SentimentFetcher:
    """Compat API used by fetch command; delegates to quant signal stack."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.data_dir = settings.data_dir_raw
        self.quant = QuantSignalFetcher(settings)

    def fetch_sentiment(self, symbol: str = "BTC") -> float:
        pair = symbol if "/" in symbol else f"{symbol}/USDT"
        score01 = self.quant.fetch_social_sentiment(pair)
        return float((score01 * 2.0) - 1.0)

    def save_sentiment(self, symbol: str, score: float) -> None:
        """Persist score in [-1,1] with a small rolling history."""
        path = self.data_dir / f"sentiment_{symbol.replace('/', '_')}.parquet"
        df = pd.DataFrame({"sentiment": [float(score)]}, index=[pd.Timestamp.now(tz="UTC")])
        df.index.name = "date"

        lock = FileLock(str(path) + ".lock")
        with lock:
            if path.exists():
                existing = pd.read_parquet(path)
                df = pd.concat([existing, df]).sort_index()
                limit = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)
                df = df[df.index >= limit]
            df.to_parquet(path)
