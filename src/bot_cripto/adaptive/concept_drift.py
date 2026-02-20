"""Concept drift detection utilities (ADWIN/PageHinkley with safe fallback)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConceptDriftResult:
    drift_detected: bool
    adwin_detected: bool
    pagehinkley_detected: bool
    fallback_detected: bool
    detector_backend: str
    details: dict[str, float]


def _fallback_shift_detector(
    history: list[float],
    short_window: int = 20,
    long_window: int = 80,
    min_shift: float = 0.15,
) -> tuple[bool, dict[str, float]]:
    if len(history) < max(short_window + long_window, 40):
        return False, {"short_mean": 0.0, "long_mean": 0.0, "relative_shift": 0.0}

    arr = np.asarray(history, dtype=float)
    short_mean = float(arr[-short_window:].mean())
    long_mean = float(arr[-(short_window + long_window) : -short_window].mean())
    if abs(long_mean) <= 1e-12:
        shift = 0.0
    else:
        shift = float((long_mean - short_mean) / abs(long_mean))
    return shift >= min_shift, {
        "short_mean": short_mean,
        "long_mean": long_mean,
        "relative_shift": shift,
    }


def detect_concept_drift(
    history: list[float],
    adwin_delta: float = 0.002,
    ph_delta: float = 0.005,
    ph_threshold: float = 50.0,
) -> ConceptDriftResult:
    """Detect concept drift from scalar performance history.

    Uses river's ADWIN and PageHinkley when available, else fallback mean-shift detector.
    """
    fallback_flag, fallback_details = _fallback_shift_detector(history)

    try:
        from river import drift  # type: ignore

        adwin = drift.ADWIN(delta=adwin_delta)
        ph = drift.PageHinkley(delta=ph_delta, threshold=ph_threshold)
        adwin_flag = False
        ph_flag = False
        for value in history:
            x = float(value)
            adwin.update(x)
            ph.update(x)
            adwin_flag = adwin_flag or bool(getattr(adwin, "drift_detected", False))
            ph_flag = ph_flag or bool(getattr(ph, "drift_detected", False))

        drift_detected = bool(adwin_flag or ph_flag or fallback_flag)
        return ConceptDriftResult(
            drift_detected=drift_detected,
            adwin_detected=bool(adwin_flag),
            pagehinkley_detected=bool(ph_flag),
            fallback_detected=bool(fallback_flag),
            detector_backend="river",
            details={
                **fallback_details,
                "adwin_delta": float(adwin_delta),
                "ph_delta": float(ph_delta),
                "ph_threshold": float(ph_threshold),
            },
        )
    except Exception:
        return ConceptDriftResult(
            drift_detected=bool(fallback_flag),
            adwin_detected=False,
            pagehinkley_detected=False,
            fallback_detected=bool(fallback_flag),
            detector_backend="fallback",
            details=fallback_details,
        )
