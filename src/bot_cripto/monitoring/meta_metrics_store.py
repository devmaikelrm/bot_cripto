"""Append-only store for meta-model historical metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MetaMetricsStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(raw, list):
            return []
        out: list[dict[str, Any]] = []
        for row in raw:
            if isinstance(row, dict):
                out.append(row)
        return out

    def append(self, payload: dict[str, Any]) -> None:
        rows = self._read()
        rows.append(payload)
        self.path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    def records(self, symbol: str | None = None, timeframe: str | None = None) -> list[dict[str, Any]]:
        rows = self._read()
        if symbol is not None:
            rows = [r for r in rows if str(r.get("symbol", "")) == symbol]
        if timeframe is not None:
            rows = [r for r in rows if str(r.get("timeframe", "")) == timeframe]
        return rows
