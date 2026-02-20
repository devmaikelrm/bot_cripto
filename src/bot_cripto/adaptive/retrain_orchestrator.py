"""Retrain orchestration helpers."""

from __future__ import annotations

from dataclasses import dataclass

from bot_cripto.core.config import Settings


@dataclass(frozen=True)
class RetrainJobResult:
    job: str
    status: str
    output: str = ""
    error: str = ""


def execute_retrain_plan(
    settings: Settings,
    symbol: str,
    timeframe: str,
    dry_run: bool = True,
    include_meta: bool = True,
) -> list[RetrainJobResult]:
    """Run standard retrain sequence for a symbol/timeframe."""
    plan = ["trend", "return", "risk"]
    if include_meta:
        plan.append("meta")

    if dry_run:
        return [RetrainJobResult(job=p, status="planned") for p in plan]

    out: list[RetrainJobResult] = []
    for p in plan:
        try:
            if p == "trend":
                from bot_cripto.jobs.train_trend import run as run_job
            elif p == "return":
                from bot_cripto.jobs.train_return import run as run_job
            elif p == "risk":
                from bot_cripto.jobs.train_risk import run as run_job
            else:
                from bot_cripto.jobs.train_meta import run as run_job
            output = run_job(symbol=symbol, timeframe=timeframe)
            out.append(RetrainJobResult(job=p, status="ok", output=str(output)))
        except Exception as exc:
            out.append(RetrainJobResult(job=p, status="error", error=str(exc)))
            break
    return out
