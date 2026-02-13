import numpy as np
import pandas as pd

from bot_cripto.backtesting.walk_forward import WalkForwardBacktester
from bot_cripto.models.baseline import BaselineModel


def test_walk_forward_backtest_runs() -> None:
    n = 400
    x = np.linspace(0, 8 * np.pi, n)
    close = 100 + np.sin(x) * 5 + np.linspace(0, 15, n)
    df = pd.DataFrame(
        {
            "close": close,
            "open": close + np.random.normal(0, 0.2, n),
            "high": close + 1,
            "low": close - 1,
            "volume": np.random.uniform(10, 1000, n),
            "rsi": np.random.uniform(30, 70, n),
            "volatility": np.random.uniform(0.001, 0.02, n),
            "log_ret": np.random.normal(0, 0.001, n),
            "atr": np.random.uniform(0.1, 2.0, n),
            "rel_vol": np.random.uniform(0.5, 2.0, n),
        }
    )
    report = WalkForwardBacktester(n_folds=3, min_train_size=220).run(df, BaselineModel)
    assert 0.0 <= report.accuracy <= 1.0
    assert report.total_net_return <= report.total_return
