"""Macro data ingestion (SPY, DXY, etc.) using yfinance."""

from __future__ import annotations

import pandas as pd
import yfinance as yf
from filelock import FileLock
from pathlib import Path
from bot_cripto.core.config import Settings
from bot_cripto.core.logging import get_logger

logger = get_logger("data.macro")

class MacroFetcher:
    """Fetch macro-economic indicators (S&P 500, DXY)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.data_dir = settings.data_dir_raw

    def fetch_macro_data(self, tickers: list[str] = ["SPY", "QQQ", "DX-Y.NYB", "GC=F"], days: int = 3500) -> None:
        """Download macro tickers (S&P, Nasdaq, DXY, Gold)."""
        self.settings.ensure_dirs()
        log = logger.bind(tickers=tickers, days=days)
        log.info("macro_ingestion_start")

        for ticker in tickers:
            try:
                # Use yfinance to download daily data. 3500 days covers back to 2016.
                period = "max" if days > 3000 else f"{days}d"
                data = yf.download(ticker, period=period, interval="1d", progress=False)
                
                if data.empty:
                    log.warning("macro_empty_data", ticker=ticker)
                    continue

                # Prepare DataFrame
                df = data.copy()
                df.index.name = "date"
                # Handle multi-index if yfinance returns it
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # Keep only Close
                df = df[["Close"]].rename(columns={"Close": f"macro_{ticker.replace('-', '_').replace('.', '_').lower()}_close"})
                df.index = pd.to_datetime(df.index, utc=True)

                # Save with locking
                safe_ticker = ticker.replace("-", "_").replace(".", "_")
                path = self.data_dir / f"macro_{safe_ticker}.parquet"
                
                lock = FileLock(str(path) + ".lock")
                with lock:
                    df.to_parquet(path, compression="snappy")
                
                log.info("macro_saved", ticker=ticker, path=str(path), rows=len(df))

            except Exception as exc:
                log.error("macro_fetch_failed", ticker=ticker, error=str(exc))

    def load_macro_all(self) -> pd.DataFrame:
        """Load all available macro parquet files and join them."""
        macro_dfs = []
        for path in self.data_dir.glob("macro_*.parquet"):
            if ".lock" in path.name:
                continue
            df = pd.read_parquet(path)
            macro_dfs.append(df)
        
        if not macro_dfs:
            return pd.DataFrame()
        
        # Join on date
        return pd.concat(macro_dfs, axis=1).sort_index()
