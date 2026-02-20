"""Walk-forward benchmark runner across multiple model families."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

import pandas as pd

from bot_cripto.backtesting.walk_forward import BacktestReport, WalkForwardBacktester
from bot_cripto.models.base import BasePredictor


@dataclass(frozen=True)
class BenchmarkResult:
    model: str
    status: str
    accuracy: float
    total_net_return: float
    sharpe: float
    stability_score: float
    error: str = ""


def resolve_model_factory(model_name: str) -> tuple[Callable[[], BasePredictor] | None, str]:
    key = model_name.strip().lower()
    if key == "baseline":
        from bot_cripto.models.baseline import BaselineModel

        return (lambda: BaselineModel()), ""
    if key == "tft":
        from bot_cripto.models.tft import TFTPredictor

        return (lambda: TFTPredictor()), ""
    if key == "nbeats":
        from bot_cripto.models.nbeats import NBeatsPredictor

        return (lambda: NBeatsPredictor()), ""
    if key in {"itransformer", "patchtst"}:
        if find_spec("neuralforecast") is None:
            return None, "neuralforecast not installed"
        from bot_cripto.models.neuralforecast_adapter import NeuralForecastAdapter

        return (lambda: NeuralForecastAdapter(model_name=key)), ""
    return None, f"unsupported model '{model_name}'"


def benchmark_models(
    df: pd.DataFrame,
    model_names: list[str],
    backtester: WalkForwardBacktester,
) -> list[BenchmarkResult]:
    out: list[BenchmarkResult] = []
    for model_name in model_names:
        factory, reason = resolve_model_factory(model_name)
        if factory is None:
            out.append(
                BenchmarkResult(
                    model=model_name,
                    status="skipped",
                    accuracy=0.0,
                    total_net_return=0.0,
                    sharpe=0.0,
                    stability_score=0.0,
                    error=reason,
                )
            )
            continue

        try:
            report: BacktestReport = backtester.run(df, model_factory=factory)
            out.append(
                BenchmarkResult(
                    model=model_name,
                    status="ok",
                    accuracy=float(report.accuracy),
                    total_net_return=float(report.total_net_return),
                    sharpe=float(report.sharpe),
                    stability_score=float(report.stability_score),
                )
            )
        except Exception as exc:
            out.append(
                BenchmarkResult(
                    model=model_name,
                    status="error",
                    accuracy=0.0,
                    total_net_return=0.0,
                    sharpe=0.0,
                    stability_score=0.0,
                    error=str(exc),
                )
            )
    return sort_benchmark_results(out)


def sort_benchmark_results(results: list[BenchmarkResult]) -> list[BenchmarkResult]:
    """Sort by net return, then Sharpe, then stability for successful models."""
    ok = [r for r in results if r.status == "ok"]
    non_ok = [r for r in results if r.status != "ok"]
    ok_sorted = sorted(
        ok,
        key=lambda r: (r.total_net_return, r.sharpe, r.stability_score),
        reverse=True,
    )
    return ok_sorted + non_ok


def build_benchmark_summary(results: list[BenchmarkResult]) -> dict[str, Any]:
    """Build winner and delta metrics against TFT baseline when available."""
    ok = [r for r in results if r.status == "ok"]
    winner = ok[0] if ok else None
    tft = next((r for r in ok if r.model.lower() == "tft"), None)

    summary: dict[str, Any] = {
        "winner": (
            {
                "model": winner.model,
                "total_net_return": winner.total_net_return,
                "sharpe": winner.sharpe,
                "stability_score": winner.stability_score,
                "accuracy": winner.accuracy,
            }
            if winner is not None
            else None
        ),
        "deltas_vs_tft": None,
    }
    if winner is not None and tft is not None:
        summary["deltas_vs_tft"] = {
            "net_return": float(winner.total_net_return - tft.total_net_return),
            "sharpe": float(winner.sharpe - tft.sharpe),
            "stability_score": float(winner.stability_score - tft.stability_score),
            "accuracy": float(winner.accuracy - tft.accuracy),
            "tft_model_present": True,
            "tft_model_status": tft.status,
        }
    return summary
