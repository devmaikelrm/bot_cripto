from __future__ import annotations

import pandas as pd

from bot_cripto.adaptive.champion_challenger import (
    evaluate_predictor,
)
from bot_cripto.models.base import PredictionOutput


class _DummyPredictor:
    def __init__(self, exp_ret: float, prob_up: float) -> None:
        self.exp_ret = exp_ret
        self.prob_up = prob_up

    def predict(self, df: pd.DataFrame) -> PredictionOutput:  # noqa: ARG002
        return PredictionOutput(
            prob_up=float(self.prob_up),
            expected_return=float(self.exp_ret),
            p10=float(self.exp_ret - 0.01),
            p50=float(self.exp_ret),
            p90=float(self.exp_ret + 0.01),
            risk_score=0.2,
        )


def test_evaluate_predictor_counts_trades() -> None:
    n = 120
    close = pd.Series([100 + i * 0.1 for i in range(n)], dtype=float)
    df = pd.DataFrame({"close": close})
    pred = _DummyPredictor(exp_ret=0.003, prob_up=0.8)
    out = evaluate_predictor(
        predictor=pred,  # type: ignore[arg-type]
        df=df,
        start_idx=20,
        prob_min=0.6,
        min_expected_return=0.002,
        roundtrip_cost=0.0001,
    )
    assert out.trades > 0
    assert 0.0 <= out.win_rate <= 1.0


def test_evaluate_predictor_no_trades_when_below_threshold() -> None:
    n = 100
    close = pd.Series([100 + ((-1) ** i) * 0.01 for i in range(n)], dtype=float)
    df = pd.DataFrame({"close": close})
    pred = _DummyPredictor(exp_ret=0.0001, prob_up=0.51)
    out = evaluate_predictor(
        predictor=pred,  # type: ignore[arg-type]
        df=df,
        start_idx=20,
        prob_min=0.7,
        min_expected_return=0.002,
        roundtrip_cost=0.0001,
    )
    assert out.trades == 0
