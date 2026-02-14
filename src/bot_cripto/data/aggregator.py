"""Multi-source OHLCV aggregator with outlier detection.

Fetches the same symbol/timeframe from multiple exchanges in parallel,
aligns timestamps, detects per-bar outliers via median absolute deviation
(MAD), and produces a consensus OHLCV DataFrame with a quality report.

Usage::

    agg = RobustDataAggregator(providers=["binance", "coinbase", "kraken"])
    report = agg.fetch_and_validate("BTC/USDT", "1h", days=7)
    clean_df = report.consensus_df
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from bot_cripto.core.logging import get_logger
from bot_cripto.data.adapters import ExchangeAdapter, build_adapter

logger = get_logger("data.aggregator")

OHLCV_COLS = ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceFetchResult:
    """Result of fetching OHLCV from a single provider."""

    provider: str
    df: pd.DataFrame
    rows: int
    ok: bool
    error: str = ""


@dataclass(frozen=True)
class BarOutlier:
    """A single bar flagged as an outlier on a specific column."""

    timestamp: str
    column: str
    provider: str
    value: float
    median: float
    mad: float
    z_score: float


@dataclass(frozen=True)
class ValidationReport:
    """Cross-exchange validation report."""

    symbol: str
    timeframe: str
    providers_requested: int
    providers_ok: int
    providers_failed: list[str] = field(default_factory=list)
    total_bars: int = 0
    outlier_bars: int = 0
    outlier_ratio: float = 0.0
    outliers: list[BarOutlier] = field(default_factory=list)
    consensus_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    per_source: dict[str, dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core aggregator
# ---------------------------------------------------------------------------


class RobustDataAggregator:
    """Fetch OHLCV from multiple exchanges and produce validated consensus.

    Parameters
    ----------
    providers : list[str]
        Exchange names understood by ``build_adapter()``.
    mad_threshold : float
        MAD-based z-score threshold for flagging outliers (default 3.5).
    max_workers : int
        Thread pool size for parallel fetches.
    request_limit : int
        Max candles per API request (CCXT ``limit``).
    """

    def __init__(
        self,
        providers: list[str] | None = None,
        mad_threshold: float = 3.5,
        max_workers: int = 4,
        request_limit: int = 1000,
    ) -> None:
        self.providers = providers or ["binance", "coinbase", "kraken"]
        self.mad_threshold = mad_threshold
        self.max_workers = max_workers
        self.request_limit = request_limit

    # ------------------------------------------------------------------
    # Single-source fetch
    # ------------------------------------------------------------------

    def _fetch_single(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        start_ts: int,
        end_ts: int,
    ) -> SourceFetchResult:
        """Fetch full history from one provider, paginating as needed."""
        try:
            adapter: ExchangeAdapter = build_adapter(provider)
        except Exception as exc:
            return SourceFetchResult(provider, pd.DataFrame(), 0, False, str(exc))

        all_rows: list[list[float]] = []
        current_ts = start_ts
        step_ms = int(adapter.parse_timeframe(timeframe)) * 1000
        rate_ms = getattr(adapter, "rate_limit_ms", 1000)

        try:
            while current_ts < end_ts:
                ohlcv = adapter.fetch_ohlcv(
                    symbol, timeframe, since=current_ts, limit=self.request_limit
                )
                if not ohlcv:
                    break
                all_rows.extend(ohlcv)
                last_ts = int(ohlcv[-1][0])
                current_ts = last_ts + step_ms
                time.sleep(rate_ms / 1000)
        except Exception as exc:
            logger.warning("source_fetch_error", provider=provider, error=str(exc))
            if not all_rows:
                return SourceFetchResult(provider, pd.DataFrame(), 0, False, str(exc))

        if not all_rows:
            return SourceFetchResult(provider, pd.DataFrame(), 0, False, "no data")

        df = pd.DataFrame(all_rows, columns=["timestamp", *OHLCV_COLS])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("date")
        df = df[~df.index.duplicated(keep="last")].sort_index()

        return SourceFetchResult(provider, df, len(df), True)

    # ------------------------------------------------------------------
    # Parallel fetch all providers
    # ------------------------------------------------------------------

    def _fetch_all(
        self,
        symbol: str,
        timeframe: str,
        days: int,
    ) -> list[SourceFetchResult]:
        now = datetime.now(tz=UTC)
        start_ts = int((now - timedelta(days=days)).timestamp() * 1000)
        end_ts = int(now.timestamp() * 1000)

        results: list[SourceFetchResult] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(
                    self._fetch_single, prov, symbol, timeframe, start_ts, end_ts
                ): prov
                for prov in self.providers
            }
            for future in as_completed(futures):
                prov = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(
                        SourceFetchResult(prov, pd.DataFrame(), 0, False, str(exc))
                    )

        return results

    # ------------------------------------------------------------------
    # Outlier detection via MAD (Median Absolute Deviation)
    # ------------------------------------------------------------------

    @staticmethod
    def _mad_z_scores(values: np.ndarray) -> np.ndarray:
        """Compute MAD-based z-scores (robust to outliers).

        Uses the standard scaling factor 1.4826 to make MAD consistent
        with the standard deviation for normally distributed data.
        """
        median = np.nanmedian(values)
        mad = np.nanmedian(np.abs(values - median)) * 1.4826
        if mad == 0:
            return np.zeros_like(values)
        return np.abs(values - median) / mad

    def _detect_outliers(
        self,
        aligned: dict[str, pd.DataFrame],
        timestamps: pd.DatetimeIndex,
    ) -> list[BarOutlier]:
        """Compare per-bar values across sources, flag MAD outliers."""
        outliers: list[BarOutlier] = []
        providers = list(aligned.keys())
        if len(providers) < 2:
            return outliers

        for col in ["open", "high", "low", "close"]:
            # Build matrix: rows=timestamps, cols=providers
            matrix = pd.DataFrame(
                {prov: aligned[prov][col].reindex(timestamps) for prov in providers}
            )
            for idx in matrix.index:
                row_vals = matrix.loc[idx].dropna().values
                if len(row_vals) < 2:
                    continue
                z_scores = self._mad_z_scores(row_vals)
                prov_names = matrix.loc[idx].dropna().index.tolist()
                median_val = float(np.nanmedian(row_vals))
                mad_val = float(np.nanmedian(np.abs(row_vals - median_val)) * 1.4826)

                for z, prov, val in zip(z_scores, prov_names, row_vals):
                    if z > self.mad_threshold:
                        outliers.append(
                            BarOutlier(
                                timestamp=str(idx),
                                column=col,
                                provider=prov,
                                value=float(val),
                                median=median_val,
                                mad=mad_val,
                                z_score=float(z),
                            )
                        )
        return outliers

    # ------------------------------------------------------------------
    # Consensus: median across sources (robust to single-source outliers)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_consensus(
        aligned: dict[str, pd.DataFrame],
        timestamps: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Produce consensus OHLCV by taking the median across sources."""
        if not aligned:
            return pd.DataFrame()

        providers = list(aligned.keys())
        if len(providers) == 1:
            return aligned[providers[0]].reindex(timestamps).copy()

        consensus = pd.DataFrame(index=timestamps)
        for col in OHLCV_COLS:
            stacked = pd.DataFrame(
                {p: aligned[p][col].reindex(timestamps) for p in providers}
            )
            consensus[col] = stacked.median(axis=1)

        # Forward-fill small gaps, then drop remaining NaN
        consensus = consensus.ffill(limit=3).dropna()
        return consensus

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_and_validate(
        self,
        symbol: str,
        timeframe: str,
        days: int = 7,
    ) -> ValidationReport:
        """Fetch from all providers, detect outliers, build consensus.

        Parameters
        ----------
        symbol : str
            Trading pair (e.g. ``"BTC/USDT"``).
        timeframe : str
            Candle interval (e.g. ``"1h"``).
        days : int
            How many days of history to fetch.

        Returns
        -------
        ValidationReport
            Contains ``consensus_df``, outlier list, and per-source stats.
        """
        logger.info(
            "aggregation_start",
            symbol=symbol,
            timeframe=timeframe,
            days=days,
            providers=self.providers,
        )

        results = self._fetch_all(symbol, timeframe, days)

        ok_results = [r for r in results if r.ok and not r.df.empty]
        failed = [r.provider for r in results if not r.ok]

        if not ok_results:
            logger.error("aggregation_no_sources", failed=failed)
            return ValidationReport(
                symbol=symbol,
                timeframe=timeframe,
                providers_requested=len(self.providers),
                providers_ok=0,
                providers_failed=failed,
            )

        # Align all sources to the union of timestamps
        aligned: dict[str, pd.DataFrame] = {}
        all_timestamps: set = set()
        for r in ok_results:
            aligned[r.provider] = r.df
            all_timestamps.update(r.df.index)

        timestamps = pd.DatetimeIndex(sorted(all_timestamps))

        # Detect outliers
        outliers = self._detect_outliers(aligned, timestamps)
        outlier_ts = {o.timestamp for o in outliers}

        # Build consensus
        consensus = self._build_consensus(aligned, timestamps)

        # Per-source stats
        per_source: dict[str, dict[str, Any]] = {}
        for r in results:
            per_source[r.provider] = {
                "ok": r.ok,
                "rows": r.rows,
                "error": r.error,
            }

        total_bars = len(timestamps)
        outlier_bars = len(outlier_ts)

        report = ValidationReport(
            symbol=symbol,
            timeframe=timeframe,
            providers_requested=len(self.providers),
            providers_ok=len(ok_results),
            providers_failed=failed,
            total_bars=total_bars,
            outlier_bars=outlier_bars,
            outlier_ratio=outlier_bars / total_bars if total_bars > 0 else 0.0,
            outliers=outliers[:100],  # cap for readability
            consensus_df=consensus,
            per_source=per_source,
        )

        logger.info(
            "aggregation_done",
            providers_ok=report.providers_ok,
            total_bars=report.total_bars,
            outlier_bars=report.outlier_bars,
            outlier_ratio=round(report.outlier_ratio, 4),
            consensus_rows=len(consensus),
        )

        return report

    def validate_existing(
        self,
        primary_df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        check_providers: list[str] | None = None,
        days: int | None = None,
    ) -> ValidationReport:
        """Validate an existing DataFrame against other exchange sources.

        Useful for checking the quality of already-fetched data without
        re-downloading the primary source.

        Parameters
        ----------
        primary_df : pd.DataFrame
            The OHLCV DataFrame to validate (must have DatetimeIndex).
        symbol, timeframe : str
            Market parameters for fetching comparison data.
        check_providers : list[str] | None
            Providers to compare against. Defaults to all except the first.
        days : int | None
            Days of history. If None, inferred from primary_df span.
        """
        if days is None:
            span = (primary_df.index.max() - primary_df.index.min()).days
            days = max(span, 1)

        providers = check_providers or self.providers[1:]
        if not providers:
            providers = ["coinbase"]  # fallback

        # Fetch comparison sources
        now = datetime.now(tz=UTC)
        start_ts = int((now - timedelta(days=days)).timestamp() * 1000)
        end_ts = int(now.timestamp() * 1000)

        results: list[SourceFetchResult] = []
        for prov in providers:
            results.append(
                self._fetch_single(prov, symbol, timeframe, start_ts, end_ts)
            )

        # Include primary as "primary" source
        aligned: dict[str, pd.DataFrame] = {"primary": primary_df}
        all_timestamps: set = set(primary_df.index)

        ok_results = [r for r in results if r.ok and not r.df.empty]
        for r in ok_results:
            aligned[r.provider] = r.df
            all_timestamps.update(r.df.index)

        timestamps = pd.DatetimeIndex(sorted(all_timestamps))
        outliers = self._detect_outliers(aligned, timestamps)
        outlier_ts = {o.timestamp for o in outliers}

        per_source: dict[str, dict[str, Any]] = {
            "primary": {"ok": True, "rows": len(primary_df), "error": ""},
        }
        for r in results:
            per_source[r.provider] = {
                "ok": r.ok,
                "rows": r.rows,
                "error": r.error,
            }

        total_bars = len(timestamps)
        outlier_bars = len(outlier_ts)

        return ValidationReport(
            symbol=symbol,
            timeframe=timeframe,
            providers_requested=len(providers) + 1,
            providers_ok=len(ok_results) + 1,
            providers_failed=[r.provider for r in results if not r.ok],
            total_bars=total_bars,
            outlier_bars=outlier_bars,
            outlier_ratio=outlier_bars / total_bars if total_bars > 0 else 0.0,
            outliers=outliers[:100],
            consensus_df=primary_df,  # keep primary as consensus
            per_source=per_source,
        )
