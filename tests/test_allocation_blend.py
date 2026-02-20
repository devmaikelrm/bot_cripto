from __future__ import annotations

import numpy as np
import pandas as pd

from bot_cripto.risk.allocation_blend import blend_allocations


def test_blend_allocations_outputs_normalized_components() -> None:
    n = 400
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(
        {
            "BTC/USDT": rng.normal(0.0005, 0.010, n),
            "ETH/USDT": rng.normal(0.0004, 0.012, n),
            "SOL/USDT": rng.normal(0.0007, 0.018, n),
        }
    )
    views = {"BTC/USDT": 0.8, "ETH/USDT": 0.2, "SOL/USDT": -0.2}
    out = blend_allocations(
        returns=returns,
        views=views,
        w_hrp=0.5,
        w_kelly=0.3,
        w_views=0.2,
    )
    assert out.method == "blend_hrp_kelly_views_corr_proxy"
    assert abs(sum(out.weights.values()) - 1.0) < 1e-9
    assert abs(sum(out.hrp_weights.values()) - 1.0) < 1e-9
    assert abs(sum(out.kelly_weights.values()) - 1.0) < 1e-9
    assert abs(sum(out.view_weights.values()) - 1.0) < 1e-9
    assert all(0.0 <= w <= 1.0 for w in out.weights.values())
    assert out.mean_abs_corr >= 0.0
    assert out.corr_shrink_applied >= 0.0


def test_blend_allocations_applies_corr_shrink_when_corr_is_high() -> None:
    n = 300
    rng = np.random.default_rng(21)
    base = rng.normal(0.0002, 0.01, n)
    returns = pd.DataFrame(
        {
            "BTC/USDT": base,
            "ETH/USDT": base + rng.normal(0.0, 0.0005, n),
            "SOL/USDT": base + rng.normal(0.0, 0.0005, n),
        }
    )
    out = blend_allocations(
        returns=returns,
        views={"BTC/USDT": 0.9, "ETH/USDT": 0.1, "SOL/USDT": -0.1},
        corr_threshold=0.10,
        corr_max_shrink=0.70,
    )
    assert out.mean_abs_corr > 0.10
    assert out.corr_shrink_applied > 0.0
