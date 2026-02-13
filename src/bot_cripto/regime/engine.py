"""Market regime detection engine (rule-based)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from bot_cripto.core.logging import get_logger

logger = get_logger("regime.engine")


class MarketRegime(StrEnum):
    TREND = "TREND"
    RANGE = "RANGE"
    HIGH_VOL = "HIGH_VOL"


@dataclass(frozen=True)
class RegimeResult:
    regime: MarketRegime
    adx: float
    atr_pct: float
    reason: str


class RegimeEngine:
    """Detect market regime using ADX + ATR percentage filters."""

    def __init__(
        self,
        adx_period: int = 14,
        atr_period: int = 14,
        adx_trend_min: float = 20.0,
        atr_high_vol_pct: float = 0.02,
    ) -> None:
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.adx_trend_min = adx_trend_min
        self.atr_high_vol_pct = atr_high_vol_pct

    def _atr(self, df: pd.DataFrame) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(self.atr_period).mean()

    def _adx(self, df: pd.DataFrame) -> pd.Series:
        high = df["high"]
        low = df["low"]

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        atr = self._atr(df)
        plus_di = 100 * (plus_dm.rolling(self.adx_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(self.adx_period).mean() / atr)

        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)).fillna(0.0) * 100
        return dx.rolling(self.adx_period).mean()

    def detect(self, df: pd.DataFrame) -> RegimeResult:
        required = {"high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns for regime detection: {sorted(missing)}")

        adx_series = self._adx(df)
        atr_series = self._atr(df)
        close = df["close"]

        adx = float(adx_series.iloc[-1]) if not adx_series.empty else 0.0
        atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
        close_last = float(close.iloc[-1])
        atr_pct = atr / close_last if close_last else 0.0

        if atr_pct >= self.atr_high_vol_pct:
            regime = MarketRegime.HIGH_VOL
            reason = f"ATR% {atr_pct:.3f} >= {self.atr_high_vol_pct:.3f}"
        elif adx >= self.adx_trend_min:
            regime = MarketRegime.TREND
            reason = f"ADX {adx:.2f} >= {self.adx_trend_min:.2f}"
        else:
            regime = MarketRegime.RANGE
            reason = f"ADX {adx:.2f} < {self.adx_trend_min:.2f}"

        logger.info("regime_detected", regime=regime.value, adx=adx, atr_pct=atr_pct, reason=reason)
        return RegimeResult(regime=regime, adx=adx, atr_pct=atr_pct, reason=reason)
