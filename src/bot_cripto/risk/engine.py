"""Risk engine with dynamic position sizing and drawdown limits."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np

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
    cooldown_minutes: int = 15
    enable_kelly: bool = True
    kelly_fraction: float = 0.2  # Fractional Kelly (safer)
    cvar_enabled: bool = True
    cvar_alpha: float = 0.05
    cvar_min_samples: int = 60
    cvar_limit: float = -0.03
    circuit_breaker_minutes: int = 60


@dataclass
class RiskState:
    equity: float = 10_000.0
    day_start_equity: float = 10_000.0
    week_start_equity: float = 10_000.0
    day_id: str = ""
    week_id: str = ""
    last_trade_ts: str = ""
    circuit_breaker_until: str = ""


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

    def _calculate_kelly_size(self, win_prob: float, win_loss_ratio: float = 1.5) -> float:
        """
        Kelly Criterion formula: f* = (p*b - q) / b
        p = probability of win
        q = probability of loss (1-p)
        b = win/loss ratio (payout)
        """
        if win_loss_ratio <= 0:
            return 0.0
        p = win_prob
        q = 1.0 - p
        b = win_loss_ratio
        
        kelly_f = (p * b - q) / b
        return max(0.0, kelly_f)

    def _compute_cvar(self, returns: list[float], alpha: float) -> float:
        """Historical CVaR (Expected Shortfall) on tail losses."""
        if not returns:
            return 0.0
        arr = np.asarray(returns, dtype=float)
        if arr.size == 0:
            return 0.0
        var_q = float(np.quantile(arr, alpha))
        tail = arr[arr <= var_q]
        if tail.size == 0:
            return var_q
        return float(np.mean(tail))

    def evaluate(
        self,
        prediction: PredictionOutput,
        regime_str: str,
        state: RiskState,
        recent_returns: list[float] | None = None,
    ) -> RiskDecision:
        self._refresh_periods(state)

        now = datetime.now(tz=UTC)
        if state.circuit_breaker_until:
            try:
                until = datetime.fromisoformat(state.circuit_breaker_until)
                if now < until:
                    remaining = int((until - now).total_seconds() / 60.0)
                    return RiskDecision(False, 0.0, f"Circuit breaker active: {remaining}min")
            except (ValueError, TypeError):
                state.circuit_breaker_until = ""

        daily_dd = self._dd(state.equity, state.day_start_equity)
        weekly_dd = self._dd(state.equity, state.week_start_equity)

        if daily_dd >= self.limits.max_daily_drawdown:
            return RiskDecision(False, 0.0, f"Daily DD limit reached: {daily_dd:.2%}")
        if weekly_dd >= self.limits.max_weekly_drawdown:
            return RiskDecision(False, 0.0, f"Weekly DD limit reached: {weekly_dd:.2%}")

        # CVaR guard on realized recent returns.
        if (
            self.limits.cvar_enabled
            and recent_returns is not None
            and len(recent_returns) >= self.limits.cvar_min_samples
        ):
            cvar = self._compute_cvar(recent_returns, alpha=self.limits.cvar_alpha)
            if cvar <= self.limits.cvar_limit:
                if self.limits.circuit_breaker_minutes > 0:
                    state.circuit_breaker_until = (
                        now + timedelta(minutes=self.limits.circuit_breaker_minutes)
                    ).isoformat()
                return RiskDecision(
                    False,
                    0.0,
                    f"CVaR breach: {cvar:.2%} <= {self.limits.cvar_limit:.2%}",
                )

        # Cooldown: block trades too soon after the last one
        if state.last_trade_ts and self.limits.cooldown_minutes > 0:
            try:
                last = datetime.fromisoformat(state.last_trade_ts)
                elapsed = (datetime.now(tz=UTC) - last).total_seconds() / 60.0
                if elapsed < self.limits.cooldown_minutes:
                    remaining = self.limits.cooldown_minutes - elapsed
                    return RiskDecision(
                        False, 0.0,
                        f"Cooldown active: {remaining:.0f}min remaining",
                    )
            except (ValueError, TypeError):
                pass  # corrupt timestamp — ignore, allow trade
        
        # Dynamic Multipliers based on ML Regime
        regime_multipliers = {
            "BULL_TREND": 1.2,
            "BEAR_TREND": 1.0,
            "RANGE_SIDEWAYS": 0.5,
            "CRISIS_HIGH_VOL": 0.0,
            "UNKNOWN": 0.0
        }
        multiplier = regime_multipliers.get(regime_str, 0.0)
        
        if multiplier <= 0:
            return RiskDecision(False, 0.0, f"Regime {regime_str} blocked risk")

        if prediction.risk_score >= self.limits.risk_score_block_threshold:
            return RiskDecision(False, 0.0, "Prediction risk_score too high")

        # Dynamic sizing: use Kelly if enabled
        if self.limits.enable_kelly:
            # win_prob based on prob_up. If it's a Long, prob_up is the win prob.
            # If we were doing Shorts, it would be 1 - prob_up.
            # Assuming Long-only strategy for now.
            win_prob = prediction.prob_up
            k_size = self._calculate_kelly_size(win_prob)
            raw = k_size * self.limits.kelly_fraction * multiplier
        else:
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
            f"daily_dd={daily_dd:.2%}, size={size:.4f} (Kelly={self.limits.enable_kelly})"
        )
        logger.info("risk_decision", allowed=True, position_size=size, reason=reason)
        return RiskDecision(True, size, reason)
