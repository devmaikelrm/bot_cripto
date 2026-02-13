"""SQLite store for operational monitoring."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DecisionRow:
    ts: str
    symbol: str
    decision: str
    confidence: float
    reason: str
    expected_return: float
    risk_score: float


class WatchtowerStore:
    """Persistent store for dashboard metrics and operational traces."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reason TEXT NOT NULL,
                    expected_return REAL NOT NULL,
                    risk_score REAL NOT NULL,
                    regime TEXT,
                    position_size REAL,
                    latency_ms REAL
                )
                """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS equity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    source TEXT NOT NULL,
                    equity REAL NOT NULL
                )
                """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL
                )
                """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    ok INTEGER NOT NULL
                )
                """)
            conn.commit()

    def log_decision(self, payload: dict[str, Any], latency_ms: float | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO decisions (
                    ts, symbol, decision, confidence, reason, expected_return, risk_score,
                    regime, position_size, latency_ms
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(payload.get("ts", "")),
                    str(payload.get("symbol", "")),
                    str(payload.get("decision", "NO_TRADE")),
                    float(payload.get("confidence", 0.0)),
                    str(payload.get("reason", "")),
                    float(payload.get("expected_return", 0.0)),
                    float(payload.get("risk_score", 0.0)),
                    str(payload.get("regime", "")),
                    float(payload.get("position_size", 0.0)),
                    float(latency_ms if latency_ms is not None else 0.0),
                ),
            )
            conn.commit()

    def log_equity(self, ts: str, equity: float, source: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO equity (ts, source, equity) VALUES (?, ?, ?)",
                (ts, source, equity),
            )
            conn.commit()

    def log_training_metrics(self, ts: str, model_name: str, metrics: dict[str, float]) -> None:
        rows = [(ts, model_name, key, float(value)) for key, value in metrics.items()]
        if not rows:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO training_metrics (ts, model_name, metric_name, metric_value)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

    def log_api_health(
        self,
        ts: str,
        provider: str,
        symbol: str,
        timeframe: str,
        latency_ms: float,
        ok: bool,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO api_health (ts, provider, symbol, timeframe, latency_ms, ok)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ts, provider, symbol, timeframe, float(latency_ms), int(ok)),
            )
            conn.commit()
