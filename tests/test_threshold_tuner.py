from __future__ import annotations

import pandas as pd

from bot_cripto.backtesting.threshold_tuner import evaluate_thresholds, tune_thresholds


def test_evaluate_thresholds_counts_trades_and_metrics() -> None:
    frame = pd.DataFrame(
        {
            "prob_up": [0.70, 0.30, 0.52, 0.80],
            "expected_return": [0.004, -0.003, 0.0001, 0.005],
            "realized_return": [0.002, -0.001, 0.0002, -0.002],
        }
    )
    result = evaluate_thresholds(
        pred_frame=frame,
        prob_min=0.60,
        min_expected_return=0.002,
        roundtrip_cost=0.0001,
    )
    assert result.trades == 3
    assert 0.0 <= result.hit_rate <= 1.0


def test_tune_thresholds_returns_best_candidate() -> None:
    frame = pd.DataFrame(
        {
            "prob_up": [0.65, 0.67, 0.62, 0.45, 0.40, 0.35],
            "expected_return": [0.004, 0.003, 0.002, -0.002, -0.003, -0.004],
            "realized_return": [0.003, 0.002, -0.001, -0.001, -0.002, -0.003],
        }
    )
    best, rows = tune_thresholds(
        pred_frame=frame,
        prob_grid=[0.55, 0.60, 0.65],
        return_grid=[0.001, 0.002],
        roundtrip_cost=0.0001,
    )
    assert len(rows) == 6
    assert best.objective >= rows[-1].objective
