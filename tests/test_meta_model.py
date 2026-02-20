import pandas as pd
import numpy as np

from bot_cripto.models.meta import MetaModel


def test_meta_model_fit_and_filter() -> None:
    n = 140
    x = pd.DataFrame(
        {
            "prob_up": [0.8] * (n // 2) + [0.2] * (n // 2),
            "expected_return": [0.01] * (n // 2) + [-0.01] * (n // 2),
            "risk_score": [0.2] * n,
            "confidence": [0.8] * n,
            "regime_bull": [1.0] * n,
            "funding_rate": [0.0] * n,
            "fear_greed": [0.6] * n,
        }
    )
    y = pd.Series([1] * (n // 2) + [0] * (n // 2))

    meta = MetaModel(min_prob_success=0.55)
    meta.fit(x, y)
    assert meta.is_fitted is True

    bad_signal = {
        "prob_up": 0.2,
        "expected_return": -0.01,
        "risk_score": 0.2,
        "confidence": 0.8,
    }
    should_block = meta.should_filter(
        bad_signal,
        regime_str="BULL_TREND",
        quant_signals={"funding_rate": 0.0, "fear_greed": 0.6},
    )
    assert isinstance(should_block, bool)


def test_meta_model_predict_success_prob_unfitted_returns_one() -> None:
    meta = MetaModel()
    prob = meta.predict_success_prob(
        {"prob_up": 0.5, "expected_return": 0.0, "risk_score": 0.5, "confidence": 0.0},
        regime_str="RANGE_SIDEWAYS",
        quant_signals={"funding_rate": 0.0, "fear_greed": 0.5},
    )
    assert prob == 1.0


def test_meta_model_optimize_threshold_returns_valid_range() -> None:
    probs = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2], dtype=float)
    labels = np.array([1, 1, 1, 0, 0, 0], dtype=int)
    best = MetaModel.optimize_threshold(
        probs=probs,
        labels=labels,
        threshold_min=0.5,
        threshold_max=0.8,
        threshold_step=0.1,
        min_positive_predictions=1,
    )
    assert 0.5 <= best["threshold"] <= 0.8
    assert 0.0 <= best["precision"] <= 1.0
    assert 0.0 <= best["recall"] <= 1.0
    assert 0.0 <= best["f1"] <= 1.0
