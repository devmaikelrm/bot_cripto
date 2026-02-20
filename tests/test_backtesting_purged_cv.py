import numpy as np
import pandas as pd

from bot_cripto.backtesting.purged_cv import (
    build_cpcv_splits,
    build_purged_kfold_splits,
    run_cpcv_backtest,
    run_purged_cv_backtest,
)
from bot_cripto.models.base import PredictionOutput


class DummyModel:
    def train(self, df: pd.DataFrame, target_col: str = "close") -> object:  # noqa: ARG002
        return object()

    def predict(self, df: pd.DataFrame) -> PredictionOutput:
        last = float(df["close"].iloc[-1])
        if len(df) >= 2:
            prev = float(df["close"].iloc[-2])
            ret = (last - prev) / prev if prev != 0 else 0.0
        else:
            ret = 0.0
        prob_up = 0.6 if ret >= 0 else 0.4
        return PredictionOutput(
            prob_up=prob_up,
            expected_return=ret,
            p10=ret - 0.01,
            p50=ret,
            p90=ret + 0.01,
            risk_score=0.2,
        )


def test_build_purged_kfold_splits_respects_purge_and_embargo() -> None:
    splits = build_purged_kfold_splits(
        n_samples=100,
        n_splits=5,
        purge_size=3,
        embargo_size=4,
    )
    assert len(splits) == 5

    for train_idx, test_idx in splits:
        train_set = set(train_idx)
        test_set = set(test_idx)
        assert train_set.isdisjoint(test_set)
        assert len(test_set) > 0

        test_start = min(test_idx)
        test_end = max(test_idx) + 1
        for i in range(max(0, test_start - 3), test_start):
            assert i not in train_set
        for i in range(test_end, min(100, test_end + 4)):
            assert i not in train_set


def test_run_purged_cv_backtest_runs() -> None:
    n = 420
    x = np.linspace(0, 10 * np.pi, n)
    close = 100 + np.sin(x) * 4 + np.linspace(0, 12, n)
    df = pd.DataFrame(
        {
            "close": close,
            "open": close + np.random.normal(0, 0.2, n),
            "high": close + 1.0,
            "low": close - 1.0,
            "volume": np.random.uniform(10, 1000, n),
            "rsi": np.random.uniform(30, 70, n),
            "volatility": np.random.uniform(0.001, 0.02, n),
            "log_ret": np.random.normal(0, 0.001, n),
            "atr": np.random.uniform(0.1, 2.0, n),
            "rel_vol": np.random.uniform(0.5, 2.0, n),
        }
    )

    report = run_purged_cv_backtest(
        df=df,
        model_factory=DummyModel,
        n_splits=5,
        purge_size=5,
        embargo_size=5,
    )

    assert report.folds >= 2
    assert 0.0 <= report.accuracy <= 1.0
    assert report.total_net_return <= report.total_return


def test_build_cpcv_splits_count_and_isolation() -> None:
    splits = build_cpcv_splits(
        n_samples=120,
        n_groups=6,
        n_test_groups=2,
        purge_size=3,
        embargo_size=4,
    )
    # C(6,2) = 15 combinations
    assert len(splits) == 15
    for train_idx, test_idx, _combo in splits:
        assert set(train_idx).isdisjoint(set(test_idx))
        assert len(test_idx) > 0
        assert len(train_idx) > 0


def test_run_cpcv_backtest_runs() -> None:
    n = 420
    x = np.linspace(0, 10 * np.pi, n)
    close = 100 + np.sin(x) * 4 + np.linspace(0, 12, n)
    df = pd.DataFrame(
        {
            "close": close,
            "open": close + np.random.normal(0, 0.2, n),
            "high": close + 1.0,
            "low": close - 1.0,
            "volume": np.random.uniform(10, 1000, n),
            "rsi": np.random.uniform(30, 70, n),
            "volatility": np.random.uniform(0.001, 0.02, n),
            "log_ret": np.random.normal(0, 0.001, n),
            "atr": np.random.uniform(0.1, 2.0, n),
            "rel_vol": np.random.uniform(0.5, 2.0, n),
        }
    )
    report = run_cpcv_backtest(
        df=df,
        model_factory=DummyModel,
        n_groups=6,
        n_test_groups=2,
        purge_size=5,
        embargo_size=5,
    )
    assert report.combinations_total >= 5
    assert 0.0 <= report.accuracy_mean <= 1.0
    assert isinstance(report.sharpe_mean, float)
    assert isinstance(report.sharpe_p5, float)
    assert all(isinstance(f.sharpe, float) for f in report.fold_results)
