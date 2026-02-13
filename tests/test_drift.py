from bot_cripto.monitoring.drift import detect_performance_drift


def test_detect_performance_drift_positive() -> None:
    history = [0.60] * 40 + [0.35] * 10
    res = detect_performance_drift(history=history, baseline_window=30, recent_window=10)
    assert res.drift_detected is True


def test_detect_performance_drift_negative() -> None:
    history = [0.60] * 40 + [0.58] * 10
    res = detect_performance_drift(history=history, baseline_window=30, recent_window=10)
    assert res.drift_detected is False
