"""Walk-forward backtesting utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from bot_cripto.core.logging import get_logger
from bot_cripto.models.base import PredictionOutput

logger = get_logger("backtesting.walk_forward")


class SupportsTrainPredict(Protocol):
    def train(self, df: pd.DataFrame, target_col: str = "close") -> object: ...

    def predict(self, df: pd.DataFrame) -> PredictionOutput: ...


@dataclass(frozen=True)
class BacktestReport:
    folds: int
    accuracy: float
    avg_return: float
    total_return: float
    avg_net_return: float
    total_net_return: float


class WalkForwardBacktester:
    """Simple temporal walk-forward evaluation."""

    def __init__(
        self,
        n_folds: int = 4,
        min_train_size: int = 200,
        fees_bps: int = 10,
        spread_bps: int = 2,
        slippage_bps: int = 3,
    ) -> None:
        self.n_folds = n_folds
        self.min_train_size = min_train_size
        self.fees_bps = fees_bps
        self.spread_bps = spread_bps
        self.slippage_bps = slippage_bps

    def run(
        self,
        df: pd.DataFrame,
        model_factory: Callable[[], SupportsTrainPredict],
    ) -> BacktestReport:
        if len(df) < self.min_train_size + self.n_folds + 1:
            raise ValueError("Dataset too small for walk-forward backtest")

        fold_size = max(1, (len(df) - self.min_train_size) // self.n_folds)
        hits = 0
        total = 0
        total_ret = 0.0
        total_net_ret = 0.0
        roundtrip_cost = (2 * self.fees_bps + self.spread_bps + self.slippage_bps) / 10_000

        for fold in range(self.n_folds):
            train_end = self.min_train_size + fold * fold_size
            test_end = min(len(df), train_end + fold_size)
            train_df = df.iloc[:train_end]
            test_df = df.iloc[train_end:test_end]
            if test_df.empty:
                break

            model: SupportsTrainPredict = model_factory()
            model.train(train_df, target_col="close")

            for idx in range(1, len(test_df)):
                window = pd.concat([train_df, test_df.iloc[:idx]])
                pred = model.predict(window)
                realized = (test_df["close"].iloc[idx] - test_df["close"].iloc[idx - 1]) / test_df[
                    "close"
                ].iloc[idx - 1]
                expected_sign = 1 if pred.expected_return >= 0 else -1
                realized_sign = 1 if realized >= 0 else -1
                gross = expected_sign * realized
                net = gross - roundtrip_cost
                hits += int(expected_sign == realized_sign)
                total += 1
                total_ret += gross
                total_net_ret += net

        if total == 0:
            raise ValueError("No evaluation points generated")

        report = BacktestReport(
            folds=self.n_folds,
            accuracy=hits / total,
            avg_return=total_ret / total,
            total_return=total_ret,
            avg_net_return=total_net_ret / total,
            total_net_return=total_net_ret,
        )
        logger.info("walk_forward_completed", **report.__dict__)
        return report
