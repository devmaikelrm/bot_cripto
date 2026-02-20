"""Persistence for risk state across runs."""

from __future__ import annotations

import json
import os
from pathlib import Path

from bot_cripto.core.logging import get_logger
from bot_cripto.risk.engine import RiskState

logger = get_logger("risk.state_store")


class RiskStateStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self, initial_equity: float = 10_000.0) -> RiskState:
        if not self.path.exists():
            return RiskState(
                equity=initial_equity,
                day_start_equity=initial_equity,
                week_start_equity=initial_equity,
            )

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("risk_state_load_failed_defaults", path=str(self.path), error=str(exc))
            return RiskState(
                equity=initial_equity,
                day_start_equity=initial_equity,
                week_start_equity=initial_equity,
            )
        if not isinstance(raw, dict):
            logger.warning("risk_state_invalid_payload_defaults", path=str(self.path))
            return RiskState(
                equity=initial_equity,
                day_start_equity=initial_equity,
                week_start_equity=initial_equity,
            )

        return RiskState(
            equity=float(raw.get("equity", initial_equity)),
            day_start_equity=float(raw.get("day_start_equity", initial_equity)),
            week_start_equity=float(raw.get("week_start_equity", initial_equity)),
            day_id=str(raw.get("day_id", "")),
            week_id=str(raw.get("week_id", "")),
            last_trade_ts=str(raw.get("last_trade_ts", "")),
            circuit_breaker_until=str(raw.get("circuit_breaker_until", "")),
        )

    def save(self, state: RiskState) -> None:
        payload = {
            "equity": state.equity,
            "day_start_equity": state.day_start_equity,
            "week_start_equity": state.week_start_equity,
            "day_id": state.day_id,
            "week_id": state.week_id,
            "last_trade_ts": state.last_trade_ts,
            "circuit_breaker_until": state.circuit_breaker_until,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)
