from __future__ import annotations

import numpy as np
import pandas as pd

from bot_cripto.backtesting.meta_cpcv import run_meta_cpcv_validation
from bot_cripto.models.meta import MetaModel


def test_run_meta_cpcv_validation_returns_metrics() -> None:
    n = 240
    rng = np.random.default_rng(123)
    x = pd.DataFrame(
        {
            "prob_up": rng.uniform(0.2, 0.8, n),
            "expected_return": rng.normal(0.0, 0.01, n),
            "risk_score": rng.uniform(0.1, 0.9, n),
            "confidence": rng.uniform(0.0, 1.0, n),
            "regime_bull": rng.integers(0, 2, n),
            "funding_rate": rng.normal(0.0, 0.001, n),
            "fear_greed": rng.uniform(0.0, 1.0, n),
        }
    )
    x = MetaModel.ensure_feature_columns(x)
    y = pd.Series((x["prob_up"].to_numpy() > 0.5).astype(int))

    report = run_meta_cpcv_validation(
        x_meta=x,
        y_meta=y,
        n_groups=6,
        n_test_groups=2,
        purge_size=3,
        embargo_size=3,
        threshold_min=0.5,
        threshold_max=0.8,
        threshold_step=0.05,
        min_positive_predictions=2,
    )
    assert report.combinations_total > 0
    assert 0.0 <= report.f1_mean <= 1.0
    assert 0.0 <= report.f1_p5 <= 1.0
