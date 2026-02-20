"""Threshold tuning utilities for decision calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ThresholdEval:
    prob_min: float
    min_expected_return: float
    trades: int
    hit_rate: float
    total_net_return: float
    sharpe: float
    objective: float


def _sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.asarray(returns, dtype=float)
    std = float(arr.std(ddof=1))
    if std <= 1e-12:
        return 0.0
    return float(arr.mean() / std * math.sqrt(252.0))


def evaluate_thresholds(
    pred_frame: pd.DataFrame,
    prob_min: float,
    min_expected_return: float,
    roundtrip_cost: float,
) -> ThresholdEval:
    """Evaluate trade quality for a given threshold pair."""
    needed = {"prob_up", "expected_return", "realized_return"}
    missing = needed - set(pred_frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    returns: list[float] = []
    hits = 0
    trades = 0
    for _, row in pred_frame.iterrows():
        prob_up = float(row["prob_up"])
        exp_ret = float(row["expected_return"])
        realized = float(row["realized_return"])

        signal = 0
        if exp_ret >= min_expected_return and prob_up >= prob_min:
            signal = 1
        elif exp_ret <= -min_expected_return and prob_up <= (1.0 - prob_min):
            signal = -1
        if signal == 0:
            continue

        gross = signal * realized
        net = gross - roundtrip_cost
        returns.append(net)
        trades += 1
        hits += int(gross > 0.0)

    total_net = float(sum(returns))
    hit_rate = float(hits / trades) if trades > 0 else 0.0
    sharpe = _sharpe(returns)
    # Balance profitability and stability; penalize no-trade configs.
    objective = total_net + (0.05 * sharpe) - (0.25 if trades == 0 else 0.0)
    return ThresholdEval(
        prob_min=float(prob_min),
        min_expected_return=float(min_expected_return),
        trades=int(trades),
        hit_rate=float(hit_rate),
        total_net_return=total_net,
        sharpe=float(sharpe),
        objective=float(objective),
    )


def tune_thresholds(
    pred_frame: pd.DataFrame,
    prob_grid: list[float],
    return_grid: list[float],
    roundtrip_cost: float,
) -> tuple[ThresholdEval, list[ThresholdEval]]:
    """Run grid search and return best + full evaluations sorted by objective."""
    rows: list[ThresholdEval] = []
    for p in prob_grid:
        for r in return_grid:
            rows.append(
                evaluate_thresholds(
                    pred_frame=pred_frame,
                    prob_min=float(p),
                    min_expected_return=float(r),
                    roundtrip_cost=roundtrip_cost,
                )
            )
    if not rows:
        raise ValueError("No threshold candidates to evaluate")
    rows_sorted = sorted(rows, key=lambda x: x.objective, reverse=True)
    return rows_sorted[0], rows_sorted
