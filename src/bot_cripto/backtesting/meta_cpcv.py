"""CPCV-like validation utilities for meta-model classification."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from bot_cripto.backtesting.purged_cv import build_cpcv_splits
from bot_cripto.models.meta import MetaModel


@dataclass(frozen=True)
class MetaCPCVFoldResult:
    combo_id: int
    test_groups: tuple[int, ...]
    train_size: int
    test_size: int
    threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float


@dataclass(frozen=True)
class MetaCPCVReport:
    n_groups: int
    n_test_groups: int
    combinations_total: int
    purge_size: int
    embargo_size: int
    f1_mean: float
    f1_p5: float
    precision_mean: float
    recall_mean: float
    accuracy_mean: float
    fold_results: list[MetaCPCVFoldResult] = field(default_factory=list)


def _metrics(probs: np.ndarray, y_true: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (probs >= threshold).astype(int)
    yt = y_true.astype(int)
    tp = int(((pred == 1) & (yt == 1)).sum())
    fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum())
    tn = int(((pred == 0) & (yt == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def run_meta_cpcv_validation(
    x_meta: pd.DataFrame,
    y_meta: pd.Series,
    n_groups: int = 6,
    n_test_groups: int = 2,
    purge_size: int = 5,
    embargo_size: int = 5,
    threshold_min: float = 0.50,
    threshold_max: float = 0.80,
    threshold_step: float = 0.01,
    min_positive_predictions: int = 5,
) -> MetaCPCVReport:
    """Run CPCV temporal validation on meta-model features/labels."""
    if len(x_meta) != len(y_meta):
        raise ValueError("x_meta and y_meta must have equal length")
    if len(x_meta) < 120:
        raise ValueError("Need at least 120 samples for stable meta CPCV")

    x = x_meta.reset_index(drop=True)
    y = y_meta.reset_index(drop=True).astype(int)
    splits = build_cpcv_splits(
        n_samples=len(x),
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        purge_size=purge_size,
        embargo_size=embargo_size,
    )
    fold_rows: list[MetaCPCVFoldResult] = []
    for combo_id, (train_idx, test_idx, combo) in enumerate(splits):
        if len(train_idx) < 80 or len(test_idx) < 20:
            continue
        x_train = x.iloc[train_idx]
        y_train = y.iloc[train_idx]
        x_test = x.iloc[test_idx]
        y_test = y.iloc[test_idx]
        if int(y_train.nunique()) < 2 or int(y_test.nunique()) < 2:
            continue

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42 + combo_id,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        probs_train = model.predict_proba(x_train)[:, 1]
        best = MetaModel.optimize_threshold(
            probs=probs_train,
            labels=y_train.to_numpy(dtype=int),
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            threshold_step=threshold_step,
            min_positive_predictions=min_positive_predictions,
        )
        thr = float(best["threshold"])
        probs_test = model.predict_proba(x_test)[:, 1]
        m = _metrics(probs_test, y_test.to_numpy(dtype=int), threshold=thr)
        fold_rows.append(
            MetaCPCVFoldResult(
                combo_id=combo_id,
                test_groups=combo,
                train_size=len(train_idx),
                test_size=len(test_idx),
                threshold=thr,
                precision=m["precision"],
                recall=m["recall"],
                f1=m["f1"],
                accuracy=m["accuracy"],
            )
        )

    if not fold_rows:
        raise ValueError("No valid CPCV folds for meta validation")

    f1_vals = np.array([f.f1 for f in fold_rows], dtype=float)
    precision_vals = np.array([f.precision for f in fold_rows], dtype=float)
    recall_vals = np.array([f.recall for f in fold_rows], dtype=float)
    acc_vals = np.array([f.accuracy for f in fold_rows], dtype=float)

    return MetaCPCVReport(
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        combinations_total=len(fold_rows),
        purge_size=purge_size,
        embargo_size=embargo_size,
        f1_mean=float(np.mean(f1_vals)),
        f1_p5=float(np.percentile(f1_vals, 5)),
        precision_mean=float(np.mean(precision_vals)),
        recall_mean=float(np.mean(recall_vals)),
        accuracy_mean=float(np.mean(acc_vals)),
        fold_results=fold_rows,
    )
