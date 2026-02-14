"""Risk engine with dynamic position sizing and drawdown limits."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from bot_cripto.core.logging import get_logger
from bot_cripto.models.base import PredictionOutput
from bot_cripto.regime.engine import MarketRegime

logger = get_logger("risk.engine")


@dataclass
class RiskLimits:
    risk_per_trade: float = 0.01
    max_daily_drawdown: float = 0.03
    max_weekly_drawdown: float = 0.07
    max_position_size: float = 1.0
    risk_score_block_threshold: float = 0.9
    position_size_multiplier: float = 10.0


@dataclass
class RiskState:
    equity: float = 10_000.0
    day_start_equity: float = 10_000.0
    week_start_equity: float = 10_000.0
    day_id: str = ""
    week_id: str = ""


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    position_size: float
    reason: str


class RiskEngine:
    """Applies risk controls and computes dynamic position size."""

    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()

    def _refresh_periods(self, state: RiskState) -> None:
        now = datetime.now(tz=UTC)
        day_id = now.strftime("%Y-%m-%d")
        week_id = now.strftime("%G-%V")

        if state.day_id != day_id:
            state.day_id = day_id
            state.day_start_equity = state.equity
        if state.week_id != week_id:
            state.week_id = week_id
            state.week_start_equity = state.equity

    def _dd(self, current: float, start: float) -> float:
        if start <= 0:
            return 0.0
        return max(0.0, (start - current) / start)

    def evaluate(
        self,
        prediction: PredictionOutput,
        regime_str: str,
        state: RiskState,
    ) -> RiskDecision:
        self._refresh_periods(state)

        daily_dd = self._dd(state.equity, state.day_start_equity)
        weekly_dd = self._dd(state.equity, state.week_start_equity)

        if daily_dd >= self.limits.max_daily_drawdown:
            return RiskDecision(False, 0.0, f"Daily DD limit reached: {daily_dd:.2%}")
        if weekly_dd >= self.limits.max_weekly_drawdown:
            return RiskDecision(False, 0.0, f"Weekly DD limit reached: {weekly_dd:.2%}")
        
        # Dynamic Multipliers based on ML Regime
        regime_multipliers = {
            "BULL_TREND": 1.2,
            "BEAR_TREND": 1.0,
            "RANGE_SIDEWAYS": 0.5,
            "CRISIS_HIGH_VOL": 0.0,
            "UNKNOWN": 0.5
        }
        multiplier = regime_multipliers.get(regime_str, 0.5)
        
        if multiplier <= 0:
            return RiskDecision(False, 0.0, f"Regime {regime_str} blocked risk")

        if prediction.risk_score >= self.limits.risk_score_block_threshold:
            return RiskDecision(False, 0.0, "Prediction risk_score too high")

        # Dynamic sizing: use TFT confidence if it was enhanced with trajectory consistency
        # Otherwise fallback to standard prob-based component
        conf = getattr(prediction, "confidence", max(0.0, prediction.prob_up - 0.5) * 2.0)
        
        raw = (
            self.limits.risk_per_trade
            * conf
            * multiplier
            * self.limits.position_size_multiplier
        )
        size = min(max(raw, 0.0), self.limits.max_position_size)

        reason = (
            f"Risk OK: {regime_str} (mult={multiplier}), "
            f"daily_dd={daily_dd:.2%}, size={size:.4f}"
        )
        logger.info("risk_decision", allowed=True, position_size=size, reason=reason)
        return RiskDecision(True, size, reason)
