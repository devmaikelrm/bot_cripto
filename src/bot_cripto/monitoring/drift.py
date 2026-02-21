"""Performance and data drift monitoring.

Provides three complementary drift detection methods:
1. **Relative drop** — simple mean comparison between baseline and recent windows.
2. **KS test** — Kolmogorov-Smirnov test on per-step metric distributions.
3. **Feature drift** — KS test on each feature column between two DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from bot_cripto.core.logging import get_logger

logger = get_logger("monitoring.drift")


# ---------------------------------------------------------------------------
# 1. Performance drift (original + KS enhancement)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftResult:
    drift_detected: bool
    baseline_mean: float
    recent_mean: float
    relative_drop: float
    ks_statistic: float = 0.0
    ks_pvalue: float = 1.0


def detect_performance_drift(
    history: list[float],
    baseline_window: int = 30,
    recent_window: int = 10,
    relative_drop_threshold: float = 0.2,
    ks_alpha: float = 0.05,
) -> DriftResult:
    """Detect performance drift using relative drop AND KS test.

    Drift is flagged when **either** the relative drop exceeds the threshold
    **or** the KS test rejects the null hypothesis that both windows come
    from the same distribution (p < ``ks_alpha``).
    """
    if len(history) < baseline_window + recent_window:
        return DriftResult(False, 0.0, 0.0, 0.0)

    baseline = history[-(baseline_window + recent_window) : -recent_window]
    recent = history[-recent_window:]

    baseline_mean = sum(baseline) / len(baseline)
    recent_mean = sum(recent) / len(recent)

    if baseline_mean == 0:
        relative_drop = 0.0
        drop_drift = False
    else:
        relative_drop = (baseline_mean - recent_mean) / abs(baseline_mean)
        drop_drift = relative_drop >= relative_drop_threshold

    # KS test: are the two windows drawn from the same distribution?
    # Only count as drift when performance is degrading (recent_mean < baseline_mean).
    # A statistically significant *improvement* should not trigger a retrain.
    ks_stat, ks_pvalue = stats.ks_2samp(baseline, recent)
    ks_drift = (ks_pvalue < ks_alpha) and (recent_mean < baseline_mean)

    drift = bool(drop_drift or ks_drift)

    logger.info(
        "performance_drift_check",
        drift_detected=drift,
        relative_drop=round(relative_drop, 4),
        ks_statistic=round(float(ks_stat), 4),
        ks_pvalue=round(float(ks_pvalue), 4),
    )

    return DriftResult(
        drift_detected=drift,
        baseline_mean=baseline_mean,
        recent_mean=recent_mean,
        relative_drop=relative_drop,
        ks_statistic=float(ks_stat),
        ks_pvalue=float(ks_pvalue),
    )


# ---------------------------------------------------------------------------
# 2. Feature-level data drift (KS test per column)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureDriftResult:
    """Drift result for a single feature column."""

    feature: str
    ks_statistic: float
    ks_pvalue: float
    drifted: bool


@dataclass(frozen=True)
class DataDriftReport:
    """Aggregate drift report across all features."""

    total_features: int
    drifted_features: int
    drift_ratio: float
    drift_detected: bool
    results: list[FeatureDriftResult] = field(default_factory=list)


def detect_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str] | None = None,
    ks_alpha: float = 0.05,
    drift_ratio_threshold: float = 0.30,
) -> DataDriftReport:
    """Detect data drift across feature distributions.

    Runs a two-sample KS test on each feature column between ``reference``
    (training data) and ``current`` (recent production data).  Drift is
    flagged when the fraction of drifted features exceeds
    ``drift_ratio_threshold``.

    Parameters
    ----------
    reference : pd.DataFrame
        Historical / training feature DataFrame.
    current : pd.DataFrame
        Recent / production feature DataFrame.
    features : list[str] | None
        Columns to test.  If *None*, uses all shared numeric columns.
    ks_alpha : float
        Significance level for per-feature KS test.
    drift_ratio_threshold : float
        If more than this fraction of features drift, overall drift is True.
    """
    if features is None:
        shared = sorted(set(reference.columns) & set(current.columns))
        features = [
            c
            for c in shared
            if pd.api.types.is_numeric_dtype(reference[c])
            and pd.api.types.is_numeric_dtype(current[c])
        ]

    results: list[FeatureDriftResult] = []
    for feat in features:
        ref_vals = reference[feat].dropna().values
        cur_vals = current[feat].dropna().values
        if len(ref_vals) < 10 or len(cur_vals) < 10:
            results.append(FeatureDriftResult(feat, 0.0, 1.0, False))
            continue
        ks_stat, ks_pvalue = stats.ks_2samp(ref_vals, cur_vals)
        results.append(
            FeatureDriftResult(
                feature=feat,
                ks_statistic=float(ks_stat),
                ks_pvalue=float(ks_pvalue),
                drifted=ks_pvalue < ks_alpha,
            )
        )

    total = len(results)
    drifted = sum(1 for r in results if r.drifted)
    ratio = drifted / total if total > 0 else 0.0
    drift_detected = ratio >= drift_ratio_threshold

    logger.info(
        "feature_drift_check",
        total_features=total,
        drifted_features=drifted,
        drift_ratio=round(ratio, 4),
        drift_detected=drift_detected,
    )

    return DataDriftReport(
        total_features=total,
        drifted_features=drifted,
        drift_ratio=ratio,
        drift_detected=drift_detected,
        results=results,
    )
