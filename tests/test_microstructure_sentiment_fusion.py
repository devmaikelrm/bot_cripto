from __future__ import annotations

import pandas as pd

from bot_cripto.features.microstructure import MicrostructureFeatures


def test_microstructure_adds_sentiment_obi_fusion_features() -> None:
    idx = pd.date_range("2026-02-01", periods=40, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [100.0] * 40,
            "high": [101.0] * 40,
            "low": [99.0] * 40,
            "close": [100.5] * 40,
            "volume": [10.0] * 40,
            "log_ret": [0.001] * 40,
            "sentiment_score": [0.95] * 40,
            "obi_score": [-0.6] * 40,
        },
        index=idx,
    )
    out = MicrostructureFeatures.compute_all(df, window=5)
    assert "micro_sentiment_obi_divergence" in out.columns
    assert "micro_orderflow_override" in out.columns
    assert float(out["micro_orderflow_override"].iloc[-1]) == -1.0

