"""Purged time-series cross-validation utilities."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import combinations
from typing import Protocol

import numpy as np
import pandas as pd

from bot_cripto.models.base import PredictionOutput


class SupportsTrainPredict(Protocol):
    def train(self, df: pd.DataFrame, target_col: str = "close") -> object: ...

    def predict(self, df: pd.DataFrame) -> PredictionOutput: ...


@dataclass(frozen=True)
class PurgedFoldResult:
    fold: int
    train_size: int
    test_size: int
    accuracy: float
    total_return: float
    total_net_return: float


@dataclass(frozen=True)
class PurgedCVReport:
    folds: int
    purge_size: int
    embargo_size: int
    accuracy: float
    avg_return: float
    total_return: float
    avg_net_return: float
    total_net_return: float
    fold_results: list[PurgedFoldResult] = field(default_factory=list)


@dataclass(frozen=True)
class CPCVFoldResult:
    combo_id: int
    test_groups: tuple[int, ...]
    train_size: int
    test_size: int
    accuracy: float
    total_return: float
    total_net_return: float
    sharpe: float
    is_sharpe: float = 0.0  # in-sample Sharpe (for overfitting detection)


@dataclass(frozen=True)
class CPCVReport:
    n_groups: int
    n_test_groups: int
    combinations_total: int
    purge_size: int
    embargo_size: int
    accuracy_mean: float
    total_return_mean: float
    total_net_return_mean: float
    total_net_return_p5: float
    sharpe_mean: float
    sharpe_p5: float
    is_sharpe_mean: float = 0.0   # mean IS Sharpe across folds
    is_oos_ratio_mean: float = 0.0  # IS/OOS Sharpe ratio: > 3 → likely overfitting
    fold_results: list[CPCVFoldResult] = field(default_factory=list)


def _sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns, dtype=float)
    std = float(np.std(arr, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.mean(arr)) / std


def build_purged_kfold_splits(
    n_samples: int,
    n_splits: int,
    purge_size: int = 0,
    embargo_size: int = 0,
) -> list[tuple[list[int], list[int]]]:
    """Build contiguous K-Fold splits with purge and embargo around each test fold."""
    if n_samples <= 1:
        return []
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if n_splits > n_samples:
        raise ValueError("n_splits must be <= n_samples")

    fold_len = int(math.ceil(n_samples / n_splits))
    out: list[tuple[list[int], list[int]]] = []

    for fold in range(n_splits):
        test_start = fold * fold_len
        test_end = min(n_samples, test_start + fold_len)
        if test_start >= test_end:
            continue

        test_idx = list(range(test_start, test_end))
        left_end = max(0, test_start - purge_size)
        right_start = min(n_samples, test_end + embargo_size)
        train_idx = list(range(0, left_end)) + list(range(right_start, n_samples))
        out.append((train_idx, test_idx))
    return out


def build_cpcv_splits(
    n_samples: int,
    n_groups: int = 6,
    n_test_groups: int = 2,
    purge_size: int = 0,
    embargo_size: int = 0,
) -> list[tuple[list[int], list[int], tuple[int, ...]]]:
    """Build combinatorial purged splits over contiguous time groups."""
    if n_samples <= 1:
        return []
    if n_groups < 2:
        raise ValueError("n_groups must be >= 2")
    if n_test_groups < 1 or n_test_groups >= n_groups:
        raise ValueError("n_test_groups must be in [1, n_groups-1]")
    if n_groups > n_samples:
        raise ValueError("n_groups must be <= n_samples")

    group_len = int(math.ceil(n_samples / n_groups))
    boundaries: list[tuple[int, int]] = []
    for g in range(n_groups):
        start = g * group_len
        end = min(n_samples, start + group_len)
        if start < end:
            boundaries.append((start, end))
    n_groups_eff = len(boundaries)
    if n_test_groups >= n_groups_eff:
        raise ValueError("effective number of groups too small for n_test_groups")

    all_idx = set(range(n_samples))
    out: list[tuple[list[int], list[int], tuple[int, ...]]] = []
    for combo in combinations(range(n_groups_eff), n_test_groups):
        test_set: set[int] = set()
        purge_set: set[int] = set()
        for g in combo:
            test_start, test_end = boundaries[g]
            test_set.update(range(test_start, test_end))
            purge_left_start = max(0, test_start - purge_size)
            purge_set.update(range(purge_left_start, test_start))
            purge_right_end = min(n_samples, test_end + embargo_size)
            purge_set.update(range(test_end, purge_right_end))

        train_set = all_idx - test_set - purge_set
        out.append((sorted(train_set), sorted(test_set), combo))
    return out


def run_purged_cv_backtest(
    df: pd.DataFrame,
    model_factory: Callable[[], SupportsTrainPredict],
    target_col: str = "close",
    n_splits: int = 5,
    purge_size: int = 5,
    embargo_size: int = 5,
    fees_bps: int = 10,
    spread_bps: int = 2,
    slippage_bps: int = 3,
) -> PurgedCVReport:
    """Evaluate model robustness with purged temporal K-Fold."""
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    splits = build_purged_kfold_splits(
        n_samples=len(df),
        n_splits=n_splits,
        purge_size=purge_size,
        embargo_size=embargo_size,
    )
    if not splits:
        raise ValueError("Dataset too small for purged cross-validation")

    roundtrip_cost = (2 * fees_bps + spread_bps + slippage_bps) / 10_000

    fold_results: list[PurgedFoldResult] = []
    global_hits = 0
    global_total = 0
    global_gross = 0.0
    global_net = 0.0

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        if len(train_idx) < 50 or len(test_idx) < 2:
            continue

        model = model_factory()
        train_df = df.iloc[train_idx]
        model.train(train_df, target_col=target_col)

        fold_hits = 0
        fold_total = 0
        fold_gross = 0.0
        fold_net = 0.0

        for idx in test_idx:
            if idx + 1 >= len(df):
                continue

            row_df = df.iloc[[idx]]
            pred = model.predict(row_df)
            close_now = float(df[target_col].iloc[idx])
            close_next = float(df[target_col].iloc[idx + 1])
            realized = (close_next - close_now) / close_now if close_now != 0 else 0.0

            expected_sign = 1 if pred.expected_return >= 0 else -1
            realized_sign = 1 if realized >= 0 else -1

            gross = expected_sign * realized
            net = gross - roundtrip_cost

            fold_hits += int(expected_sign == realized_sign)
            fold_total += 1
            fold_gross += gross
            fold_net += net

        if fold_total == 0:
            continue

        fold_acc = fold_hits / fold_total
        fold_results.append(
            PurgedFoldResult(
                fold=fold_idx,
                train_size=len(train_idx),
                test_size=len(test_idx),
                accuracy=fold_acc,
                total_return=fold_gross,
                total_net_return=fold_net,
            )
        )

        global_hits += fold_hits
        global_total += fold_total
        global_gross += fold_gross
        global_net += fold_net

    if not fold_results or global_total == 0:
        raise ValueError("No valid folds produced results")

    return PurgedCVReport(
        folds=len(fold_results),
        purge_size=purge_size,
        embargo_size=embargo_size,
        accuracy=global_hits / global_total,
        avg_return=global_gross / len(fold_results),
        total_return=global_gross,
        avg_net_return=global_net / len(fold_results),
        total_net_return=global_net,
        fold_results=fold_results,
    )


def run_cpcv_backtest(
    df: pd.DataFrame,
    model_factory: Callable[[], SupportsTrainPredict],
    target_col: str = "close",
    n_groups: int = 6,
    n_test_groups: int = 2,
    purge_size: int = 5,
    embargo_size: int = 5,
    fees_bps: int = 10,
    spread_bps: int = 2,
    slippage_bps: int = 3,
) -> CPCVReport:
    """Run combinatorial purged CV (CPCV-lite) for robustness distribution."""
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    splits = build_cpcv_splits(
        n_samples=len(df),
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        purge_size=purge_size,
        embargo_size=embargo_size,
    )
    if not splits:
        raise ValueError("Dataset too small for CPCV")

    roundtrip_cost = (2 * fees_bps + spread_bps + slippage_bps) / 10_000
    fold_results: list[CPCVFoldResult] = []

    for combo_id, (train_idx, test_idx, combo) in enumerate(splits):
        if len(train_idx) < 50 or len(test_idx) < 2:
            continue
        train_df = df.iloc[train_idx]
        model = model_factory()
        model.train(train_df, target_col=target_col)

        hits = 0
        total = 0
        gross_ret = 0.0
        net_ret = 0.0
        net_returns: list[float] = []
        for idx in test_idx:
            if idx + 1 >= len(df):
                continue
            row_df = df.iloc[[idx]]
            pred = model.predict(row_df)
            close_now = float(df[target_col].iloc[idx])
            close_next = float(df[target_col].iloc[idx + 1])
            realized = (close_next - close_now) / close_now if close_now != 0 else 0.0

            expected_sign = 1 if pred.expected_return >= 0 else -1
            realized_sign = 1 if realized >= 0 else -1
            gross = expected_sign * realized
            net = gross - roundtrip_cost
            hits += int(expected_sign == realized_sign)
            total += 1
            gross_ret += gross
            net_ret += net
            net_returns.append(net)

        if total == 0:
            continue

        # IS Sharpe: evaluate model on a subsample of training points.
        # Capped at 500 to bound compute; use the most-recent train samples
        # (highest temporal relevance).
        _IS_MAX = 500
        is_sample = train_idx[-_IS_MAX:] if len(train_idx) > _IS_MAX else train_idx
        is_net_returns: list[float] = []
        for is_idx in is_sample:
            if is_idx + 1 >= len(df):
                continue
            is_pred = model.predict(df.iloc[[is_idx]])
            c0 = float(df[target_col].iloc[is_idx])
            c1 = float(df[target_col].iloc[is_idx + 1])
            r = (c1 - c0) / c0 if c0 != 0 else 0.0
            sign = 1 if is_pred.expected_return >= 0 else -1
            is_net_returns.append(sign * r - roundtrip_cost)

        fold_is_sharpe = _sharpe(is_net_returns)
        fold_oos_sharpe = _sharpe(net_returns)

        fold_results.append(
            CPCVFoldResult(
                combo_id=combo_id,
                test_groups=combo,
                train_size=len(train_idx),
                test_size=len(test_idx),
                accuracy=hits / total,
                total_return=gross_ret,
                total_net_return=net_ret,
                sharpe=fold_oos_sharpe,
                is_sharpe=fold_is_sharpe,
            )
        )

    if not fold_results:
        raise ValueError("No valid CPCV combinations produced results")

    acc_mean = sum(f.accuracy for f in fold_results) / len(fold_results)
    ret_mean = sum(f.total_return for f in fold_results) / len(fold_results)
    net_mean = sum(f.total_net_return for f in fold_results) / len(fold_results)
    net_sorted = sorted(f.total_net_return for f in fold_results)
    p5 = float(np.percentile(np.array(net_sorted, dtype=float), 5))
    sharpe_vals = [f.sharpe for f in fold_results]
    sharpe_mean = float(np.mean(np.array(sharpe_vals, dtype=float)))
    sharpe_p5 = float(np.percentile(np.array(sharpe_vals, dtype=float), 5))

    # IS Sharpe and IS/OOS ratio (overfitting diagnostic)
    is_sharpe_vals = np.array([f.is_sharpe for f in fold_results], dtype=float)
    is_sharpe_mean = float(np.mean(is_sharpe_vals))
    # Ratio: IS/OOS.  Avoid division by zero; clamp OOS to ±0.01 floor.
    ratios = []
    for f in fold_results:
        oos = f.sharpe if abs(f.sharpe) > 0.01 else (0.01 if f.sharpe >= 0 else -0.01)
        ratios.append(f.is_sharpe / oos)
    is_oos_ratio_mean = float(np.mean(ratios)) if ratios else 0.0

    return CPCVReport(
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        combinations_total=len(fold_results),
        purge_size=purge_size,
        embargo_size=embargo_size,
        accuracy_mean=acc_mean,
        total_return_mean=ret_mean,
        total_net_return_mean=net_mean,
        total_net_return_p5=p5,
        sharpe_mean=sharpe_mean,
        sharpe_p5=sharpe_p5,
        is_sharpe_mean=is_sharpe_mean,
        is_oos_ratio_mean=is_oos_ratio_mean,
        fold_results=fold_results,
    )
