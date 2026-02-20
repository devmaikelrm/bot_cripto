"""Live trading executor with hard guardrails."""

from __future__ import annotations

from datetime import UTC, datetime
from time import monotonic
from typing import Any

from bot_cripto.core.config import Settings, get_settings
from bot_cripto.core.logging import get_logger
from bot_cripto.decision.engine import Action, TradeSignal
from bot_cripto.models.base import PredictionOutput
from bot_cripto.ops.operator_flags import default_flags_store
from bot_cripto.risk.engine import RiskState
from bot_cripto.risk.state_store import RiskStateStore

logger = get_logger("execution.live")


class LiveExecutor:
    """Guarded live executor. Order placement is intentionally stubbed."""

    REQUIRED_CONFIRMATION = "I_UNDERSTAND_LIVE_TRADING"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.state_store = RiskStateStore(self.settings.logs_dir / "live_risk_state.json")
        self._state_cache: RiskState | None = None
        self._state_cache_ts: float = 0.0

    def _state(self) -> RiskState:
        now = monotonic()
        if self._state_cache is None or (now - self._state_cache_ts) >= self.settings.live_state_refresh_seconds:
            self._state_cache = self.state_store.load(initial_equity=self.settings.initial_equity)
            self._state_cache_ts = now
        return self._state_cache

    def _daily_drawdown(self) -> float:
        state = self._state()
        if state.day_start_equity <= 0:
            return 0.0
        return max(0.0, (state.day_start_equity - state.equity) / state.day_start_equity)

    def _base_response(
        self, status: str, symbol: str, action: str, reason: str, **extra: Any
    ) -> dict[str, Any]:
        return {
            "ts": datetime.now(tz=UTC).isoformat(),
            "status": status,
            "symbol": symbol,
            "action": action,
            "reason": reason,
            **extra,
        }

    def execute_signal(
        self,
        symbol: str,
        signal: TradeSignal,
        price: float | None = None,
        prediction: PredictionOutput | None = None,
    ) -> dict[str, Any]:
        flags = default_flags_store(self.settings).load()
        if flags.kill_switch:
            return self._base_response(
                status="blocked",
                symbol=symbol,
                action=signal.action.value,
                reason="operator kill switch enabled",
            )

        # Pause blocks new entries, but allow exits.
        if flags.is_paused() and signal.action == Action.BUY:
            return self._base_response(
                status="blocked",
                symbol=symbol,
                action=signal.action.value,
                reason="operator pause enabled",
            )

        # Live execution additionally requires temporary operator arming.
        if not flags.is_live_armed():
            return self._base_response(
                status="blocked",
                symbol=symbol,
                action=signal.action.value,
                reason="live not armed by operator",
            )

        if not self.settings.live_mode:
            return self._base_response(
                status="blocked",
                symbol=symbol,
                action=signal.action.value,
                reason="LIVE_MODE disabled",
            )

        if self.settings.live_confirm_token != self.REQUIRED_CONFIRMATION:
            return self._base_response(
                status="blocked",
                symbol=symbol,
                action=signal.action.value,
                reason="live confirmation token missing",
            )

        daily_dd = self._daily_drawdown()
        if daily_dd >= self.settings.live_max_daily_loss:
            return self._base_response(
                status="blocked",
                symbol=symbol,
                action=signal.action.value,
                reason=f"daily loss limit reached ({daily_dd:.2%})",
                daily_drawdown=daily_dd,
            )

        if signal.action in {Action.HOLD, Action.NEUTRAL}:
            return self._base_response(
                status="skipped",
                symbol=symbol,
                action=signal.action.value,
                reason="no actionable signal",
            )

        hard_stop_price: float | None = None
        if signal.action == Action.BUY:
            if prediction is None or price is None:
                return self._base_response(
                    status="blocked",
                    symbol=symbol,
                    action=signal.action.value,
                    reason="missing prediction/price for hard stop validation",
                )
            expected_loss = max(0.0, -prediction.p10)
            if expected_loss >= self.settings.hard_stop_max_loss:
                return self._base_response(
                    status="blocked",
                    symbol=symbol,
                    action=signal.action.value,
                    reason=f"hard stop expected loss too high ({expected_loss:.2%})",
                )
            hard_stop_price = price * (1 + prediction.p10 - self.settings.stop_loss_buffer)

        logger.warning(
            "live_execution_dry_stub",
            symbol=symbol,
            action=signal.action.value,
            price=price,
            confidence=signal.confidence,
            hard_stop_price=hard_stop_price,
        )
        return self._base_response(
            status="ready",
            symbol=symbol,
            action=signal.action.value,
            reason="all guardrails passed; exchange integration pending",
            price=price,
            confidence=signal.confidence,
            hard_stop_price=hard_stop_price,
        )
