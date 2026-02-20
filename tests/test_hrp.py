from __future__ import annotations

import numpy as np
import pandas as pd

from bot_cripto.risk.hrp import hrp_allocate


def test_hrp_allocate_returns_valid_weights() -> None:
    n = 300
    rng = np.random.default_rng(42)
    returns = pd.DataFrame(
        {
            "BTC/USDT": rng.normal(0.0005, 0.01, n),
            "ETH/USDT": rng.normal(0.0004, 0.012, n),
            "SOL/USDT": rng.normal(0.0007, 0.018, n),
        }
    )
    report = hrp_allocate(returns)
    assert report.method == "hrp_greedy"
    assert len(report.weights) == 3
    total = sum(report.weights.values())
    assert abs(total - 1.0) < 1e-9
    assert all(0.0 <= w <= 1.0 for w in report.weights.values())
    assert sorted(report.ordered_assets) == sorted(list(returns.columns))
