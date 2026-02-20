"""Execution layer router by market domain and mode."""

from __future__ import annotations

from typing import Any

from bot_cripto.core.config import Settings
from bot_cripto.core.market import market_domain
from bot_cripto.decision.engine import TradeSignal
from bot_cripto.execution.live import LiveExecutor
from bot_cripto.execution.paper import PaperExecutor
from bot_cripto.models.base import PredictionOutput


class ExecutionRouter:
    """Routes trade execution through live/paper executors preserving market domain metadata."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.paper = PaperExecutor(settings=settings)
        self.live = LiveExecutor(settings=settings)

    def execute(
        self,
        *,
        symbol: str,
        signal: TradeSignal,
        price: float,
        qty: float = 1.0,
        prediction: PredictionOutput | None = None,
    ) -> dict[str, Any]:
        domain = market_domain(symbol)
        if self.settings.live_mode:
            payload = self.live.execute_signal(
                symbol=symbol,
                signal=signal,
                price=price,
                prediction=prediction,
            )
            payload["market_domain"] = domain
            payload["execution_mode"] = "live"
            return payload

        rec = self.paper.on_signal(
            symbol=symbol,
            signal=signal,
            price=price,
            qty=qty,
            prediction=prediction,
        )
        return {
            "market_domain": domain,
            "execution_mode": "paper",
            "trade_record": rec.__dict__ if rec is not None else None,
        }
