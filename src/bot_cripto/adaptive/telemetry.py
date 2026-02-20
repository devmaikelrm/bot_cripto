"""Adaptive-system telemetry helpers for Watchtower."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from bot_cripto.core.config import Settings
from bot_cripto.monitoring.watchtower_store import WatchtowerStore


def log_adaptive_telemetry(
    settings: Settings,
    *,
    event_type: str,
    severity: str,
    payload: dict[str, Any],
) -> None:
    store = WatchtowerStore(settings.watchtower_db_path)
    store.log_adaptive_event(
        ts=datetime.now(tz=UTC).isoformat(),
        event_type=event_type,
        severity=severity,
        payload_json=json.dumps(payload, default=str),
    )
