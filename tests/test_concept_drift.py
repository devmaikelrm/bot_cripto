from __future__ import annotations

from bot_cripto.adaptive.concept_drift import detect_concept_drift


def test_detect_concept_drift_fallback_positive() -> None:
    history = [0.60] * 120 + [0.30] * 30
    out = detect_concept_drift(history)
    assert out.drift_detected is True


def test_detect_concept_drift_fallback_negative() -> None:
    history = [0.55 + (i % 3) * 0.001 for i in range(200)]
    out = detect_concept_drift(history)
    assert out.fallback_detected is False
