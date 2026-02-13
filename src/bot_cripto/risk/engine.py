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
        regime: MarketRegime,
        state: RiskState,
    ) -> RiskDecision:
        self._refresh_periods(state)

        daily_dd = self._dd(state.equity, state.day_start_equity)
        weekly_dd = self._dd(state.equity, state.week_start_equity)

        if daily_dd >= self.limits.max_daily_drawdown:
            return RiskDecision(False, 0.0, f"Daily DD limit reached: {daily_dd:.2%}")
        if weekly_dd >= self.limits.max_weekly_drawdown:
            return RiskDecision(False, 0.0, f"Weekly DD limit reached: {weekly_dd:.2%}")
        if regime == MarketRegime.HIGH_VOL:
            return RiskDecision(False, 0.0, "Regime HIGH_VOL blocked")
        if prediction.risk_score >= self.limits.risk_score_block_threshold:
            return RiskDecision(False, 0.0, "Prediction risk_score too high")

        confidence_component = max(0.0, prediction.prob_up - 0.5) * 2.0
        risk_penalty = max(0.0, 1.0 - prediction.risk_score)
        raw = (
            self.limits.risk_per_trade
            * confidence_component
            * risk_penalty
            * self.limits.position_size_multiplier
        )
        size = min(max(raw, 0.0), self.limits.max_position_size)

        reason = f"Risk OK: daily_dd={daily_dd:.2%}, weekly_dd={weekly_dd:.2%}, " f"size={size:.4f}"
        logger.info("risk_decision", allowed=True, position_size=size, reason=reason)
        return RiskDecision(True, size, reason)
