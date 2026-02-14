"""Online learning system with multi-trigger retrain policy.

Evaluates three independent retrain triggers and produces a recommendation:
1. **Time-based** — retrain every N hours regardless of performance.
2. **Performance degradation** — accuracy/return drop detected via relative
   comparison and KS test on the performance history.
3. **Data drift** — KS test on feature distributions between training data
   and recent production data.

Usage::

    system = OnlineLearningSystem(settings)
    recommendation = system.evaluate(
        performance_history=[0.55, 0.52, 0.48, ...],
        reference_features=train_df,
        current_features=recent_df,
    )
    if recommendation.should_retrain:
        # launch retrain pipeline
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from bot_cripto.core.config import Settings, get_settings
from bot_cripto.core.logging import get_logger
from bot_cripto.monitoring.drift import (
    DataDriftReport,
    DriftResult,
    detect_feature_drift,
    detect_performance_drift,
)

logger = get_logger("adaptive.online_learner")


@dataclass(frozen=True)
class TriggerResult:
    """Result of a single retrain trigger evaluation."""

    name: str
    fired: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrainRecommendation:
    """Aggregate recommendation from all triggers."""

    should_retrain: bool
    urgency: str  # "none", "low", "medium", "high"
    triggers_fired: int
    triggers_total: int
    results: list[TriggerResult] = field(default_factory=list)


class OnlineLearningSystem:
    """Evaluates retrain triggers and produces a recommendation.

    Parameters
    ----------
    settings : Settings | None
        Bot settings (for paths). Uses ``get_settings()`` if *None*.
    retrain_interval_hours : float
        Maximum hours between retrains before time trigger fires.
    perf_baseline_window : int
        Number of recent performance points for baseline window.
    perf_recent_window : int
        Number of most-recent performance points for comparison.
    perf_drop_threshold : float
        Relative drop in performance to trigger retrain.
    perf_ks_alpha : float
        KS test significance level for performance drift.
    feature_ks_alpha : float
        KS test significance level for per-feature drift.
    feature_drift_ratio : float
        Fraction of features that must drift to trigger retrain.
    last_retrain_path : Path | None
        File that stores the timestamp of the last retrain.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        retrain_interval_hours: float = 24.0,
        perf_baseline_window: int = 30,
        perf_recent_window: int = 10,
        perf_drop_threshold: float = 0.20,
        perf_ks_alpha: float = 0.05,
        feature_ks_alpha: float = 0.05,
        feature_drift_ratio: float = 0.30,
        last_retrain_path: Path | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retrain_interval_hours = retrain_interval_hours
        self.perf_baseline_window = perf_baseline_window
        self.perf_recent_window = perf_recent_window
        self.perf_drop_threshold = perf_drop_threshold
        self.perf_ks_alpha = perf_ks_alpha
        self.feature_ks_alpha = feature_ks_alpha
        self.feature_drift_ratio = feature_drift_ratio
        self.last_retrain_path = last_retrain_path or (
            self.settings.logs_dir / "last_retrain.txt"
        )

    # ------------------------------------------------------------------
    # Trigger 1: Time-based
    # ------------------------------------------------------------------

    def _check_time_trigger(self) -> TriggerResult:
        now = datetime.now(tz=UTC)
        last_retrain = self._read_last_retrain()

        if last_retrain is None:
            return TriggerResult(
                name="time",
                fired=True,
                reason="No previous retrain timestamp found",
                details={"hours_since_retrain": None},
            )

        elapsed = (now - last_retrain).total_seconds() / 3600.0
        fired = elapsed >= self.retrain_interval_hours

        return TriggerResult(
            name="time",
            fired=fired,
            reason=(
                f"Last retrain {elapsed:.1f}h ago (threshold: {self.retrain_interval_hours}h)"
            ),
            details={
                "hours_since_retrain": round(elapsed, 2),
                "threshold_hours": self.retrain_interval_hours,
                "last_retrain": last_retrain.isoformat(),
            },
        )

    def _read_last_retrain(self) -> datetime | None:
        if not self.last_retrain_path.exists():
            return None
        try:
            text = self.last_retrain_path.read_text(encoding="utf-8").strip()
            return datetime.fromisoformat(text)
        except (OSError, ValueError):
            return None

    def record_retrain(self) -> None:
        """Write current UTC timestamp as last retrain time."""
        self.last_retrain_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_retrain_path.write_text(
            datetime.now(tz=UTC).isoformat(), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Trigger 2: Performance degradation
    # ------------------------------------------------------------------

    def _check_performance_trigger(
        self, history: list[float]
    ) -> TriggerResult:
        if len(history) < self.perf_baseline_window + self.perf_recent_window:
            return TriggerResult(
                name="performance",
                fired=False,
                reason=f"Insufficient history ({len(history)} points, need {self.perf_baseline_window + self.perf_recent_window})",
                details={"history_length": len(history)},
            )

        result: DriftResult = detect_performance_drift(
            history=history,
            baseline_window=self.perf_baseline_window,
            recent_window=self.perf_recent_window,
            relative_drop_threshold=self.perf_drop_threshold,
            ks_alpha=self.perf_ks_alpha,
        )

        return TriggerResult(
            name="performance",
            fired=result.drift_detected,
            reason=(
                f"Relative drop: {result.relative_drop:.2%}, "
                f"KS p-value: {result.ks_pvalue:.4f}"
            ),
            details={
                "baseline_mean": round(result.baseline_mean, 6),
                "recent_mean": round(result.recent_mean, 6),
                "relative_drop": round(result.relative_drop, 4),
                "ks_statistic": round(result.ks_statistic, 4),
                "ks_pvalue": round(result.ks_pvalue, 4),
            },
        )

    # ------------------------------------------------------------------
    # Trigger 3: Feature / data drift
    # ------------------------------------------------------------------

    def _check_data_drift_trigger(
        self,
        reference: pd.DataFrame | None,
        current: pd.DataFrame | None,
    ) -> TriggerResult:
        if reference is None or current is None:
            return TriggerResult(
                name="data_drift",
                fired=False,
                reason="Reference or current features not provided",
            )

        if reference.empty or current.empty:
            return TriggerResult(
                name="data_drift",
                fired=False,
                reason="Reference or current features are empty",
            )

        report: DataDriftReport = detect_feature_drift(
            reference=reference,
            current=current,
            ks_alpha=self.feature_ks_alpha,
            drift_ratio_threshold=self.feature_drift_ratio,
        )

        drifted_names = [r.feature for r in report.results if r.drifted]

        return TriggerResult(
            name="data_drift",
            fired=report.drift_detected,
            reason=(
                f"{report.drifted_features}/{report.total_features} features drifted "
                f"({report.drift_ratio:.0%}, threshold: {self.feature_drift_ratio:.0%})"
            ),
            details={
                "total_features": report.total_features,
                "drifted_features": report.drifted_features,
                "drift_ratio": round(report.drift_ratio, 4),
                "drifted_names": drifted_names[:20],  # cap for readability
            },
        )

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        performance_history: list[float] | None = None,
        reference_features: pd.DataFrame | None = None,
        current_features: pd.DataFrame | None = None,
    ) -> RetrainRecommendation:
        """Evaluate all triggers and produce a recommendation.

        Parameters
        ----------
        performance_history : list[float] | None
            Per-inference accuracy or return values.
        reference_features : pd.DataFrame | None
            Training-time feature DataFrame (or a representative sample).
        current_features : pd.DataFrame | None
            Recent production feature DataFrame.

        Returns
        -------
        RetrainRecommendation
        """
        results: list[TriggerResult] = []

        # 1. Time trigger
        results.append(self._check_time_trigger())

        # 2. Performance trigger
        if performance_history is not None:
            results.append(self._check_performance_trigger(performance_history))

        # 3. Data drift trigger
        results.append(
            self._check_data_drift_trigger(reference_features, current_features)
        )

        fired = [r for r in results if r.fired]
        n_fired = len(fired)
        n_total = len(results)

        # Urgency heuristic
        if n_fired == 0:
            urgency = "none"
        elif n_fired == 1:
            # Time-only is low urgency; performance or data drift alone is medium
            urgency = "low" if fired[0].name == "time" else "medium"
        elif n_fired == 2:
            urgency = "medium"
        else:
            urgency = "high"

        recommendation = RetrainRecommendation(
            should_retrain=n_fired > 0,
            urgency=urgency,
            triggers_fired=n_fired,
            triggers_total=n_total,
            results=results,
        )

        logger.info(
            "retrain_evaluation",
            should_retrain=recommendation.should_retrain,
            urgency=recommendation.urgency,
            triggers_fired=n_fired,
            triggers=[r.name for r in fired],
        )

        return recommendation
