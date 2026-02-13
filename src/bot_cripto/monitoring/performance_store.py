"""Persistent performance history storage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PerformancePoint:
    ts: str
    metric: float


class PerformanceStore:
    """Append/read model or trading performance metrics in JSON format."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> list[PerformancePoint]:
        if not self.path.exists():
            return []
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            return []
        out: list[PerformancePoint] = []
        for item in raw:
            if (
                isinstance(item, dict)
                and isinstance(item.get("ts"), str)
                and isinstance(item.get("metric"), (float, int))
            ):
                out.append(PerformancePoint(ts=item["ts"], metric=float(item["metric"])))
        return out

    def append(self, point: PerformancePoint) -> None:
        rows = self.read()
        rows.append(point)
        payload = [{"ts": p.ts, "metric": p.metric} for p in rows]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def metrics(self) -> list[float]:
        return [p.metric for p in self.read()]
