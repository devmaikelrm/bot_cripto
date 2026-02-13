"""Performance drift monitoring."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DriftResult:
    drift_detected: bool
    baseline_mean: float
    recent_mean: float
    relative_drop: float


def detect_performance_drift(
    history: list[float],
    baseline_window: int = 30,
    recent_window: int = 10,
    relative_drop_threshold: float = 0.2,
) -> DriftResult:
    if len(history) < baseline_window + recent_window:
        return DriftResult(False, 0.0, 0.0, 0.0)

    baseline = history[-(baseline_window + recent_window) : -recent_window]
    recent = history[-recent_window:]

    baseline_mean = sum(baseline) / len(baseline)
    recent_mean = sum(recent) / len(recent)

    if baseline_mean == 0:
        relative_drop = 0.0
        drift = False
    else:
        relative_drop = (baseline_mean - recent_mean) / abs(baseline_mean)
        drift = relative_drop >= relative_drop_threshold

    return DriftResult(drift, baseline_mean, recent_mean, relative_drop)
