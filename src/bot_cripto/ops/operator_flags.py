"""Operator-controlled flags (pause/kill-switch) persisted to disk.

This is intentionally simple: a JSON file in LOGS_DIR that can be modified by
automation (Telegram control bot, ops scripts) and read by inference/execution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bot_cripto.core.config import Settings, get_settings
from bot_cripto.core.logging import get_logger

logger = get_logger("ops.flags")


@dataclass
class OperatorFlags:
    pause_until: str | None = None  # ISO timestamp UTC
    kill_switch: bool = False
    live_armed_until: str | None = None  # ISO timestamp UTC
    note: str = ""

    def is_paused(self, now: datetime | None = None) -> bool:
        if self.kill_switch:
            return True
        if not self.pause_until:
            return False
        now_dt = now or datetime.now(tz=UTC)
        try:
            until = datetime.fromisoformat(self.pause_until.replace("Z", "+00:00"))
        except ValueError:
            return False
        return now_dt < until

    def is_live_armed(self, now: datetime | None = None) -> bool:
        if not self.live_armed_until:
            return False
        now_dt = now or datetime.now(tz=UTC)
        try:
            until = datetime.fromisoformat(self.live_armed_until.replace("Z", "+00:00"))
        except ValueError:
            return False
        return now_dt < until


class OperatorFlagsStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> OperatorFlags:
        if not self.path.exists():
            return OperatorFlags()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("flags_load_failed", path=str(self.path), error=str(exc))
            return OperatorFlags()

        return OperatorFlags(
            pause_until=raw.get("pause_until"),
            kill_switch=bool(raw.get("kill_switch", False)),
            live_armed_until=raw.get("live_armed_until"),
            note=str(raw.get("note", "")),
        )

    def save(self, flags: OperatorFlags) -> None:
        payload: dict[str, Any] = {
            "pause_until": flags.pause_until,
            "kill_switch": bool(flags.kill_switch),
            "live_armed_until": flags.live_armed_until,
            "note": flags.note,
            "updated_at": datetime.now(tz=UTC).isoformat(),
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def default_flags_store(settings: Settings | None = None) -> OperatorFlagsStore:
    s = settings or get_settings()
    return OperatorFlagsStore(s.logs_dir / "operator_flags.json")

