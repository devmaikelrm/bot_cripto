import numpy as np

from bot_cripto.monitoring.drift import detect_performance_drift


def test_detect_performance_drift_positive() -> None:
    history = [0.60] * 40 + [0.35] * 10
    res = detect_performance_drift(history=history, baseline_window=30, recent_window=10)
    assert res.drift_detected is True


def test_detect_performance_drift_negative() -> None:
    # Use slight variance so KS test sees overlapping distributions
    rng = np.random.default_rng(42)
    baseline = (0.60 + rng.normal(0, 0.02, 40)).tolist()
    recent = (0.59 + rng.normal(0, 0.02, 10)).tolist()
    history = baseline + recent
    res = detect_performance_drift(history=history, baseline_window=30, recent_window=10)
    assert res.drift_detected is False
