"""Walk-forward backtesting utilities with stability scoring."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from bot_cripto.core.logging import get_logger
from bot_cripto.models.base import PredictionOutput

logger = get_logger("backtesting.walk_forward")


class SupportsTrainPredict(Protocol):
    def train(self, df: pd.DataFrame, target_col: str = "close") -> object: ...

    def predict(self, df: pd.DataFrame) -> PredictionOutput: ...


@dataclass(frozen=True)
class FoldResult:
    """Metrics for a single walk-forward fold."""

    fold: int
    train_size: int
    test_size: int
    accuracy: float
    total_return: float
    total_net_return: float
    sharpe: float
    max_drawdown: float
    profit_factor: float


@dataclass(frozen=True)
class BacktestReport:
    folds: int
    accuracy: float
    avg_return: float
    total_return: float
    avg_net_return: float
    total_net_return: float
    stability_score: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    fold_results: list[FoldResult] = field(default_factory=list)


def _sharpe_ratio(returns: list[float], annualize: float = 1.0) -> float:
    """Sharpe ratio from a list of per-step returns."""
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns)
    std = float(np.std(arr, ddof=1))
    if std == 0:
        return 0.0
    return float(np.mean(arr)) / std * math.sqrt(annualize)


def _max_drawdown(returns: list[float]) -> float:
    """Maximum drawdown from cumulative return series."""
    if not returns:
        return 0.0
    equity = np.cumprod(1.0 + np.array(returns))
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1.0)
    return float(np.min(dd))  # most negative value


def _profit_factor(returns: list[float]) -> float:
    """Sum of positive returns / abs(sum of negative returns)."""
    arr = np.array(returns)
    gross_profit = float(arr[arr > 0].sum())
    gross_loss = float(np.abs(arr[arr < 0].sum()))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def _stability_score(fold_results: list[FoldResult]) -> float:
    """Compute a stability score 0-100 across folds.

    Components (equally weighted, 25 pts each):
    1. Accuracy consistency — low std across folds
    2. Return consistency — fraction of profitable folds
    3. Sharpe consistency — low std of Sharpe across folds
    4. Drawdown consistency — no extreme outlier drawdowns
    """
    if len(fold_results) < 2:
        return 50.0  # not enough folds to assess stability

    accuracies = [f.accuracy for f in fold_results]
    net_returns = [f.total_net_return for f in fold_results]
    sharpes = [f.sharpe for f in fold_results]
    drawdowns = [f.max_drawdown for f in fold_results]

    # 1. Accuracy consistency (25 pts): lower std = better
    # Perfect: std=0 -> 25 pts. Worst: std >= 0.15 -> 0 pts
    acc_std = float(np.std(accuracies))
    acc_score = max(0.0, 25.0 * (1.0 - acc_std / 0.15))

    # 2. Profitable folds (25 pts): fraction of folds with net return > 0
    profitable_ratio = sum(1 for r in net_returns if r > 0) / len(net_returns)
    ret_score = 25.0 * profitable_ratio

    # 3. Sharpe consistency (25 pts): lower std = better
    sharpe_std = float(np.std(sharpes))
    sharpe_score = max(0.0, 25.0 * (1.0 - sharpe_std / 1.5))

    # 4. Drawdown consistency (25 pts): no extreme outliers
    # All drawdowns within 2x of median -> full score
    dd_abs = [abs(d) for d in drawdowns]
    dd_median = float(np.median(dd_abs)) if dd_abs else 0.0
    if dd_median == 0:
        dd_score = 25.0
    else:
        outlier_ratio = sum(1 for d in dd_abs if d > 2.0 * dd_median) / len(dd_abs)
        dd_score = 25.0 * (1.0 - outlier_ratio)

    return round(acc_score + ret_score + sharpe_score + dd_score, 1)


class WalkForwardBacktester:
    """Walk-forward evaluation with per-fold metrics and stability scoring.

    Supports two windowing modes:
    - **anchored** (default): training window grows from a fixed start,
      always including all data up to the current split point.
    - **rolling**: training window has a fixed size and slides forward.
    """

    def __init__(
        self,
        n_folds: int = 4,
        min_train_size: int = 200,
        fees_bps: int = 10,
        spread_bps: int = 2,
        slippage_bps: int = 3,
        train_size: int | None = None,
        test_size: int | None = None,
        step_size: int | None = None,
        anchored: bool = True,
    ) -> None:
        self.n_folds = n_folds
        self.min_train_size = min_train_size
        self.fees_bps = fees_bps
        self.spread_bps = spread_bps
        self.slippage_bps = slippage_bps
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.anchored = anchored

    def _build_splits(
        self, n_rows: int
    ) -> list[tuple[int, int, int]]:
        """Return list of (train_start, train_end, test_end) index tuples."""
        splits: list[tuple[int, int, int]] = []

        if self.train_size and self.test_size:
            # Explicit train/test/step sizes (e.g. 90d/30d/15d in candles)
            step = self.step_size or self.test_size
            train_start = 0
            train_end = self.train_size

            while train_end + 1 < n_rows:
                test_end = min(n_rows, train_end + self.test_size)
                splits.append((train_start, train_end, test_end))
                if not self.anchored:
                    train_start += step
                train_end += step

        else:
            # Legacy fold-based mode
            fold_size = max(1, (n_rows - self.min_train_size) // self.n_folds)
            for fold in range(self.n_folds):
                train_end = self.min_train_size + fold * fold_size
                test_end = min(n_rows, train_end + fold_size)
                if train_end >= n_rows:
                    break
                train_start = 0 if self.anchored else max(0, train_end - self.min_train_size)
                splits.append((train_start, train_end, test_end))

        return splits

    def run(
        self,
        df: pd.DataFrame,
        model_factory: Callable[[], SupportsTrainPredict],
    ) -> BacktestReport:
        splits = self._build_splits(len(df))
        if not splits:
            raise ValueError("Dataset too small for walk-forward backtest")

        roundtrip_cost = (2 * self.fees_bps + self.spread_bps + self.slippage_bps) / 10_000

        all_hits = 0
        all_total = 0
        all_gross = 0.0
        all_net = 0.0
        all_net_returns: list[float] = []
        fold_results: list[FoldResult] = []

        for fold_idx, (train_start, train_end, test_end) in enumerate(splits):
            train_df = df.iloc[train_start:train_end]
            test_df = df.iloc[train_end:test_end]
            if test_df.empty or len(train_df) < 2:
                continue

            model: SupportsTrainPredict = model_factory()
            model.train(train_df, target_col="close")

            fold_hits = 0
            fold_total = 0
            fold_gross_ret = 0.0
            fold_net_ret = 0.0
            fold_net_returns: list[float] = []

            for idx in range(1, len(test_df)):
                window = pd.concat([train_df, test_df.iloc[:idx]])
                pred = model.predict(window)
                realized = (
                    test_df["close"].iloc[idx] - test_df["close"].iloc[idx - 1]
                ) / test_df["close"].iloc[idx - 1]
                expected_sign = 1 if pred.expected_return >= 0 else -1
                realized_sign = 1 if realized >= 0 else -1
                gross = expected_sign * realized
                net = gross - roundtrip_cost

                fold_hits += int(expected_sign == realized_sign)
                fold_total += 1
                fold_gross_ret += gross
                fold_net_ret += net
                fold_net_returns.append(net)

            if fold_total == 0:
                continue

            fold_acc = fold_hits / fold_total
            fold_sharpe = _sharpe_ratio(fold_net_returns)
            fold_dd = _max_drawdown(fold_net_returns)
            fold_pf = _profit_factor(fold_net_returns)

            fold_results.append(
                FoldResult(
                    fold=fold_idx,
                    train_size=len(train_df),
                    test_size=len(test_df),
                    accuracy=fold_acc,
                    total_return=fold_gross_ret,
                    total_net_return=fold_net_ret,
                    sharpe=fold_sharpe,
                    max_drawdown=fold_dd,
                    profit_factor=fold_pf,
                )
            )

            all_hits += fold_hits
            all_total += fold_total
            all_gross += fold_gross_ret
            all_net += fold_net_ret
            all_net_returns.extend(fold_net_returns)

        if all_total == 0:
            raise ValueError("No evaluation points generated")

        stability = _stability_score(fold_results)

        report = BacktestReport(
            folds=len(fold_results),
            accuracy=all_hits / all_total,
            avg_return=all_gross / all_total,
            total_return=all_gross,
            avg_net_return=all_net / all_total,
            total_net_return=all_net,
            stability_score=stability,
            sharpe=_sharpe_ratio(all_net_returns),
            max_drawdown=_max_drawdown(all_net_returns),
            profit_factor=_profit_factor(all_net_returns),
            fold_results=fold_results,
        )
        logger.info(
            "walk_forward_completed",
            folds=report.folds,
            accuracy=report.accuracy,
            total_net_return=report.total_net_return,
            stability_score=report.stability_score,
            sharpe=report.sharpe,
            max_drawdown=report.max_drawdown,
        )
        return report
