import numpy as np
import pandas as pd
import pytest

from bot_cripto.regime.engine import MarketRegime, RegimeEngine


def _sample_df(vol_scale: float = 1.0, trend: float = 0.0) -> pd.DataFrame:
    n = 300
    x = np.linspace(0, 10, n)
    close = 100 + trend * x + np.sin(x) * vol_scale
    high = close + 1.5 * vol_scale
    low = close - 1.5 * vol_scale
    return pd.DataFrame({"high": high, "low": low, "close": close})


def test_regime_high_vol_detected() -> None:
    engine = RegimeEngine(atr_high_vol_pct=0.005)
    result = engine.detect(_sample_df(vol_scale=4.0))
    assert result.regime in {MarketRegime.HIGH_VOL, MarketRegime.TREND, MarketRegime.RANGE}


def test_regime_columns_required() -> None:
    engine = RegimeEngine()
    with pytest.raises(ValueError):
        engine.detect(pd.DataFrame({"close": [1, 2, 3]}))
