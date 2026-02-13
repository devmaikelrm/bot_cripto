"""Persistence for risk state across runs."""

from __future__ import annotations

import json
from pathlib import Path

from bot_cripto.risk.engine import RiskState


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

        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Invalid risk state payload")

        return RiskState(
            equity=float(raw.get("equity", initial_equity)),
            day_start_equity=float(raw.get("day_start_equity", initial_equity)),
            week_start_equity=float(raw.get("week_start_equity", initial_equity)),
            day_id=str(raw.get("day_id", "")),
            week_id=str(raw.get("week_id", "")),
        )

    def save(self, state: RiskState) -> None:
        payload = {
            "equity": state.equity,
            "day_start_equity": state.day_start_equity,
            "week_start_equity": state.week_start_equity,
            "day_id": state.day_id,
            "week_id": state.week_id,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
