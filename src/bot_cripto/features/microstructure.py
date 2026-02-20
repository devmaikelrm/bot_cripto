"""Microstructure features -- computed from OHLCV candles only.

Implements market microstructure estimators that approximate properties
typically derived from tick-level data using only Open/High/Low/Close/Volume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bot_cripto.core.logging import get_logger

logger = get_logger("features.microstructure")

_DEFAULT_WINDOW: int = 20


class MicrostructureFeatures:
    """Vectorized microstructure feature computations from OHLCV data."""

    @staticmethod
    def volume_imbalance(
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        window: int = _DEFAULT_WINDOW,
    ) -> pd.Series:
        """Bulk Volume Classification: approximate buy/sell pressure.

        buy_fraction = (close - low) / (high - low)
        imbalance = (buy_vol - sell_vol) / total_vol   in [-1, 1]
        """
        total_range = high - low
        buy_frac = (close - low) / total_range.replace(0, np.nan)
        buy_frac = buy_frac.fillna(0.5)  # doji -> neutral
        buy_vol = buy_frac * volume
        sell_vol = (1.0 - buy_frac) * volume
        total_vol = buy_vol + sell_vol
        imbalance = (buy_vol - sell_vol) / total_vol.replace(0, np.nan)
        imbalance = imbalance.fillna(0.0)
        return imbalance.rolling(window=window, min_periods=1).mean()

    @staticmethod
    def price_impact(
        log_ret: pd.Series,
        volume: pd.Series,
        window: int = _DEFAULT_WINDOW,
    ) -> pd.Series:
        """Price impact: |log_return| / sqrt(volume)."""
        safe_vol = volume.replace(0, np.nan)
        impact = log_ret.abs() / np.sqrt(safe_vol)
        impact = impact.fillna(0.0)
        return impact.rolling(window=window, min_periods=1).mean()

    @staticmethod
    def amihud_illiquidity(
        log_ret: pd.Series,
        volume: pd.Series,
        window: int = _DEFAULT_WINDOW,
    ) -> pd.Series:
        """Amihud illiquidity ratio: |log_return| / volume."""
        safe_vol = volume.replace(0, np.nan)
        illiq = log_ret.abs() / safe_vol
        illiq = illiq.fillna(0.0)
        return illiq.rolling(window=window, min_periods=1).mean()

    @staticmethod
    def kyle_lambda(
        close: pd.Series,
        volume: pd.Series,
        window: int = _DEFAULT_WINDOW,
    ) -> pd.Series:
        """Kyle's lambda: rolling OLS slope of delta_price ~ signed_volume."""
        delta_price = close.diff()
        sign = delta_price.apply(np.sign)
        signed_vol = sign * volume

        cov_xy = delta_price.rolling(window=window, min_periods=window).cov(signed_vol)
        var_x = signed_vol.rolling(window=window, min_periods=window).var()

        safe_var = var_x.replace(0, np.nan)
        lam = cov_xy / safe_var
        return lam.fillna(0.0)

    @staticmethod
    def parkinson_volatility(
        high: pd.Series,
        low: pd.Series,
        window: int = _DEFAULT_WINDOW,
    ) -> pd.Series:
        """Parkinson range-based volatility: (1/(4*ln2)) * ln(H/L)^2."""
        safe_low = low.replace(0, np.nan)
        hl_ratio = (high / safe_low).fillna(1.0)
        hl_log = np.log(hl_ratio)
        parkinson = (1.0 / (4.0 * np.log(2.0))) * (hl_log**2)
        return np.sqrt(parkinson.rolling(window=window, min_periods=1).mean())

    @staticmethod
    def garman_klass_volatility(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = _DEFAULT_WINDOW,
    ) -> pd.Series:
        """Garman-Klass OHLC volatility estimator."""
        safe_low = low.replace(0, np.nan).ffill().bfill()
        safe_open = open_.replace(0, np.nan).ffill().bfill()
        hl_log = np.log(high / safe_low)
        co_log = np.log(close / safe_open)
        gk = 0.5 * (hl_log**2) - (2.0 * np.log(2.0) - 1.0) * (co_log**2)
        rolling_mean = gk.rolling(window=window, min_periods=1).mean()
        return np.sqrt(rolling_mean.clip(lower=0.0))

    @staticmethod
    def rogers_satchell_volatility(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = _DEFAULT_WINDOW,
    ) -> pd.Series:
        """Rogers-Satchell drift-independent volatility."""
        safe_close = close.replace(0, np.nan).ffill().bfill()
        safe_open = open_.replace(0, np.nan).ffill().bfill()
        log_hc = np.log(high / safe_close)
        log_ho = np.log(high / safe_open)
        log_lc = np.log(low / safe_close)
        log_lo = np.log(low / safe_open)
        rs = (log_hc * log_ho) + (log_lc * log_lo)
        rolling_mean = rs.rolling(window=window, min_periods=1).mean()
        return np.sqrt(rolling_mean.clip(lower=0.0))

    @staticmethod
    def roll_spread(
        log_ret: pd.Series,
        window: int = _DEFAULT_WINDOW,
    ) -> pd.Series:
        """Roll (1984) implicit bid-ask spread estimator."""
        r_lag = log_ret.shift(1)
        gamma = log_ret.rolling(window=window, min_periods=window).cov(r_lag)
        spread = np.where(gamma < 0, 2.0 * np.sqrt(-gamma), 0.0)
        return pd.Series(spread, index=log_ret.index)

    @staticmethod
    def jump_score(
        log_ret: pd.Series,
        window: int = _DEFAULT_WINDOW,
    ) -> pd.Series:
        """Jump detection z-score via bipower variation."""
        r_sq = log_ret**2
        rv = r_sq.rolling(window=window, min_periods=window).sum()

        abs_r = log_ret.abs()
        abs_r_lag = abs_r.shift(1)
        bv = (np.pi / 2.0) * (abs_r * abs_r_lag).rolling(
            window=window, min_periods=window
        ).sum()

        jump_raw = (rv - bv).clip(lower=0.0)

        z_window = window * 5
        z_mean = jump_raw.rolling(window=z_window, min_periods=window).mean()
        z_std = jump_raw.rolling(window=z_window, min_periods=window).std()
        safe_std = z_std.replace(0, np.nan)
        z = ((jump_raw - z_mean) / safe_std).fillna(0.0)
        return z

    @staticmethod
    def wick_ratios(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.DataFrame:
        """Upper and lower wick ratios (per-candle, no rolling)."""
        body_top = pd.concat([open_, close], axis=1).max(axis=1)
        body_bot = pd.concat([open_, close], axis=1).min(axis=1)
        total_range = (high - low).replace(0, np.nan)

        upper_wick = ((high - body_top) / total_range).fillna(0.0)
        lower_wick = ((body_bot - low) / total_range).fillna(0.0)

        return pd.DataFrame(
            {
                "micro_upper_wick_ratio": upper_wick,
                "micro_lower_wick_ratio": lower_wick,
            },
            index=open_.index,
        )

    @staticmethod
    def vwap_deviation(
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        window: int = _DEFAULT_WINDOW,
    ) -> pd.Series:
        """Deviation of close from rolling VWAP proxy."""
        typical = (high + low + close) / 3.0
        tp_vol = typical * volume
        cum_tp_vol = tp_vol.rolling(window=window, min_periods=1).sum()
        cum_vol = volume.rolling(window=window, min_periods=1).sum()
        safe_cum_vol = cum_vol.replace(0, np.nan)
        vwap = cum_tp_vol / safe_cum_vol
        vwap = vwap.fillna(close)
        safe_close = close.replace(0, np.nan).ffill().bfill()
        deviation = (close - vwap) / safe_close
        return deviation.fillna(0.0)

    @classmethod
    def compute_all(
        cls, df: pd.DataFrame, window: int = _DEFAULT_WINDOW
    ) -> pd.DataFrame:
        """Compute all microstructure features and add to DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: open, high, low, close, volume.
            Should already have ``log_ret`` (from FeaturePipeline).
        window : int
            Default rolling window for most features.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with new ``micro_*`` columns appended.
        """
        out = df.copy()

        if "log_ret" not in out.columns:
            out["log_ret"] = np.log(out["close"] / out["close"].shift(1))

        log_ret = out["log_ret"]
        o, h, l, c, v = (
            out["open"],
            out["high"],
            out["low"],
            out["close"],
            out["volume"],
        )

        out["micro_volume_imbalance"] = cls.volume_imbalance(c, h, l, v, window)
        out["micro_price_impact"] = cls.price_impact(log_ret, v, window)
        out["micro_amihud"] = cls.amihud_illiquidity(log_ret, v, window)
        out["micro_kyle_lambda"] = cls.kyle_lambda(c, v, window)
        out["micro_parkinson_vol"] = cls.parkinson_volatility(h, l, window)
        out["micro_gk_vol"] = cls.garman_klass_volatility(o, h, l, c, window)
        out["micro_rs_vol"] = cls.rogers_satchell_volatility(o, h, l, c, window)
        out["micro_roll_spread"] = cls.roll_spread(log_ret, window)
        out["micro_jump_score"] = cls.jump_score(log_ret, window)

        wick_df = cls.wick_ratios(o, h, l, c)
        out["micro_upper_wick_ratio"] = wick_df["micro_upper_wick_ratio"]
        out["micro_lower_wick_ratio"] = wick_df["micro_lower_wick_ratio"]

        out["micro_vwap_deviation"] = cls.vwap_deviation(c, h, l, v, window)

        # OBI + Sentiment fusion: detect divergence between social mood and real order flow.
        if "sentiment_score" in out.columns and "obi_score" in out.columns:
            social_signed = (pd.to_numeric(out["sentiment_score"], errors="coerce").fillna(0.5) * 2.0) - 1.0
            obi_signed = pd.to_numeric(out["obi_score"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
            out["micro_sentiment_obi_divergence"] = (social_signed - obi_signed).clip(-2.0, 2.0)
            out["micro_orderflow_override"] = np.where(
                (social_signed > 0.6) & (obi_signed < -0.2),
                -1.0,
                np.where((social_signed < -0.6) & (obi_signed > 0.2), 1.0, 0.0),
            )
        else:
            out["micro_sentiment_obi_divergence"] = 0.0
            out["micro_orderflow_override"] = 0.0

        logger.info("microstructure_features_computed", count=14, window=window)
        return out
