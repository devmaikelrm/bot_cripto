"""Data ingestion with OHLCV validation and gap filling."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter

import pandas as pd
from filelock import FileLock

from bot_cripto.core.config import Settings
from bot_cripto.core.logging import get_logger
from bot_cripto.data.adapters import ExchangeAdapter, build_adapter
from bot_cripto.monitoring.watchtower_store import WatchtowerStore

logger = get_logger("data.ingestion")


def _is_retryable_network_error(exc: Exception) -> bool:
    return exc.__class__.__name__ in {"NetworkError", "RequestTimeout"}


def _is_rate_limit_error(exc: Exception) -> bool:
    return exc.__class__.__name__ == "RateLimitExceeded"


class BinanceFetcher:
    """Download and manage OHLCV data from Binance spot."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.data_dir = settings.data_dir_raw
        self.adapter: ExchangeAdapter = build_adapter(settings.data_provider)
        self.exchange = getattr(self.adapter, "client", self.adapter)
        self.rate_limit_ms = int(getattr(self.adapter, "rate_limit_ms", 1000))
        self.watchtower = WatchtowerStore(settings.watchtower_db_path)

    def fetch_order_book_imbalance(self, symbol: str, depth: int = 20) -> float:
        """Fetch order book and calculate imbalance: (bids_vol - asks_vol) / total_vol."""
        try:
            # Some adapters might not have fetch_order_book, use the exchange client directly
            ob = self.exchange.fetch_order_book(symbol, limit=depth)
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])
            
            if not bids or not asks:
                return 0.0
                
            # Volume-weighted sum for the specified depth
            bids_vol = sum(b[1] for b in bids)
            asks_vol = sum(a[1] for a in asks)
            
            total_vol = bids_vol + asks_vol
            if total_vol == 0:
                return 0.0
                
            return (bids_vol - asks_vol) / total_vol
        except Exception as exc:
            logger.warning("order_book_fetch_failed", symbol=symbol, error=str(exc))
            return 0.0

    def fetch_whale_pressure(self, symbol: str, threshold_btc: float = 1.0) -> float:
        """Calculate net whale pressure: sum(large_buy_vol) - sum(large_sell_vol)."""
        try:
            # Fetch latest 100 trades
            trades = self.exchange.fetch_trades(symbol, limit=100)
            if not trades:
                return 0.0
                
            net_whale_vol = 0.0
            for t in trades:
                amount = float(t.get("amount", 0))
                if amount >= threshold_btc:
                    # if side is 'buy', positive; if 'sell', negative
                    side = t.get("side", "")
                    if side == "buy":
                        net_whale_vol += amount
                    elif side == "sell":
                        net_whale_vol -= amount
            return net_whale_vol
        except Exception as exc:
            logger.warning("whale_trades_fetch_failed", symbol=symbol, error=str(exc))
            return 0.0

    def fetch_history(
        self,
        symbol: str,
        timeframe: str,
        days: int = 30,
        request_limit: int = 1000,
        log_every_batches: int = 1,
        checkpoint_every_batches: int = 0,
    ) -> pd.DataFrame:
        started = perf_counter()
        now = datetime.now(tz=UTC)
        start_ts = int((now - timedelta(days=days)).timestamp() * 1000)
        end_ts = int(now.timestamp() * 1000)

        all_candles: list[list[float]] = []
        current_ts = start_ts
        batches = 0

        log = logger.bind(symbol=symbol, timeframe=timeframe, days=days)
        log.info("ingestion_start")

        while current_ts < end_ts:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=current_ts, limit=request_limit
                )
                if not ohlcv:
                    log.warning("ingestion_empty_batch", current_ts=current_ts)
                    break

                all_candles.extend(ohlcv)
                batches += 1
                last_ts = int(ohlcv[-1][0])
                duration_ms = int(self.exchange.parse_timeframe(timeframe) * 1000)
                current_ts = last_ts + duration_ms

                time.sleep(self.rate_limit_ms / 1000)
                if batches % max(1, log_every_batches) == 0:
                    elapsed_s = perf_counter() - started
                    log.info(
                        "ingestion_batch",
                        batch=batches,
                        batch_candles=len(ohlcv),
                        total_candles=len(all_candles),
                        last_ts=last_ts,
                        next_ts=current_ts,
                        elapsed_s=round(elapsed_s, 2),
                    )

                if checkpoint_every_batches and batches % checkpoint_every_batches == 0:
                    # Checkpoint raw data so operators can see progress on disk.
                    checkpoint_df = self._process_raw_data(all_candles)
                    checkpoint_df = self.fill_gaps(checkpoint_df, timeframe)
                    self.save_data(checkpoint_df, symbol, timeframe)
                    log.info(
                        "ingestion_checkpoint_saved",
                        batch=batches,
                        rows=len(checkpoint_df),
                        path=str(self.data_dir / f"{symbol.replace('/', '_')}_{timeframe}.parquet"),
                    )

            except Exception as exc:
                if _is_retryable_network_error(exc):
                    log.error("ingestion_network_retry", error=str(exc))
                    time.sleep(5)
                    continue
                if _is_rate_limit_error(exc):
                    log.error("ingestion_rate_limit", error=str(exc))
                    time.sleep(60)
                    continue
                self.watchtower.log_api_health(
                    ts=datetime.now(tz=UTC).isoformat(),
                    provider=self.adapter.name,
                    symbol=symbol,
                    timeframe=timeframe,
                    latency_ms=(perf_counter() - started) * 1000,
                    ok=False,
                )
                log.error("ingestion_fatal", error=str(exc))
                raise

        if not all_candles:
            return pd.DataFrame()

        df = self._process_raw_data(all_candles)
        self.validate_data(df, timeframe)
        filled = self.fill_gaps(df, timeframe)
        self.validate_data(filled, timeframe)
        log.info(
            "ingestion_done",
            batches=batches,
            rows=len(filled),
            elapsed_s=round(perf_counter() - started, 2),
        )
        self.watchtower.log_api_health(
            ts=datetime.now(tz=UTC).isoformat(),
            provider=self.adapter.name,
            symbol=symbol,
            timeframe=timeframe,
            latency_ms=(perf_counter() - started) * 1000,
            ok=True,
        )
        return filled

    def _process_raw_data(self, ohlcv: list[list[float]]) -> pd.DataFrame:
        frame = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        frame["date"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        frame = frame.set_index("date")
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        return frame

    def fill_gaps(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Fill missing candles to keep indicator windows stable."""
        if df.empty:
            return df

        step_seconds = int(self.exchange.parse_timeframe(timeframe))
        expected_index = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=f"{step_seconds}s",
            tz="UTC",
        )

        out = df.reindex(expected_index)
        missing = int(out["close"].isna().sum())
        if missing == 0:
            return out

        out["close"] = out["close"].ffill()
        out["open"] = out["open"].ffill()
        out["high"] = out["high"].ffill()
        out["low"] = out["low"].ffill()
        out["volume"] = out["volume"].fillna(0.0)

        out = out.dropna(subset=["open", "high", "low", "close"])
        out["timestamp"] = (out.index.astype("int64") // 10**6).astype("int64")

        logger.warning("gaps_filled", timeframe=timeframe, missing=missing)
        return out

    def validate_data(self, df: pd.DataFrame, timeframe: str) -> None:
        if df.empty:
            return

        expected_diff = int(self.exchange.parse_timeframe(timeframe) * 1000)
        actual_diff = df["timestamp"].diff().dropna()
        gaps = actual_diff[actual_diff > expected_diff]

        log = logger.bind(timeframe=timeframe)
        if not gaps.empty:
            log.warning(
                "gaps_detected",
                count=len(gaps),
                max_gap_ms=float(gaps.max()),
            )
        else:
            log.info("validation_ok_no_gaps")

    def save_microstructure_snapshot(
        self, symbol: str, obi: float, whale_pressure: float, sentiment: float = 0.0
    ) -> Path:
        """Append a point-in-time microstructure snapshot to an accumulating parquet."""
        safe_symbol = symbol.replace("/", "_")
        path = self.data_dir / f"{safe_symbol}_micro_snapshots.parquet"

        row = pd.DataFrame(
            {"obi": [obi], "whale_pressure": [whale_pressure], "sentiment": [sentiment]},
            index=[pd.Timestamp.now(tz="UTC")],
        )
        row.index.name = "date"

        lock = FileLock(str(path) + ".lock")
        with lock:
            if path.exists():
                try:
                    existing = pd.read_parquet(path)
                    row = pd.concat([existing, row]).sort_index()
                except Exception:
                    pass
                # Retain only last 7 days
                cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
                row = row[row.index >= cutoff]
            row.to_parquet(path)

        logger.info(
            "micro_snapshot_saved",
            path=str(path),
            rows=len(row),
            obi=obi,
            whale_pressure=whale_pressure,
        )
        return path

    def save_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Path:
        safe_symbol = symbol.replace("/", "_")
        path = self.data_dir / f"{safe_symbol}_{timeframe}.parquet"

        self.settings.ensure_dirs()

        # Concurrency hardening:
        # - File lock prevents cycle/retrain/manual writers from clobbering the same parquet.
        # - Atomic replace ensures readers never see a partially-written file.
        lock = FileLock(str(path) + ".lock")
        with lock:
            if path.exists():
                try:
                    existing = pd.read_parquet(path)
                except Exception as exc:
                    # If the parquet was corrupted in a previous run, quarantine it and proceed.
                    ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
                    quarantine = path.with_name(path.name + f".corrupt.{ts}")
                    try:
                        path.replace(quarantine)
                    except OSError:
                        pass
                    logger.error(
                        "existing_parquet_read_failed_quarantined",
                        path=str(path),
                        quarantine=str(quarantine),
                        error=str(exc),
                    )
                    existing = None

                if existing is not None and not existing.empty:
                    merged = pd.concat([existing, df])
                    df = merged[~merged.index.duplicated(keep="last")].sort_index()

            tmp_path = path.with_name(path.name + f".tmp.{os.getpid()}")
            df.to_parquet(tmp_path, compression="snappy")
            os.replace(tmp_path, path)
        logger.info(
            "data_saved",
            path=str(path),
            rows=len(df),
            start=df.index[0].isoformat(),
            end=df.index[-1].isoformat(),
        )
        return path
