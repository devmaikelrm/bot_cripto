import numpy as np

from bot_cripto.models.calibration import ProbabilityCalibrator


def test_isotonic_calibrator_fit_predict() -> None:
    raw = np.array([0.05, 0.1, 0.2, 0.25, 0.3, 0.7, 0.75, 0.8, 0.9, 0.95] * 4)
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 4)
    calibrator = ProbabilityCalibrator(method="isotonic")
    metrics = calibrator.fit(raw_probs=raw, labels=labels)
    preds = calibrator.predict(np.array([0.2, 0.8]))

    assert metrics.samples == len(raw)
    assert 0.0 <= preds[0] <= 1.0
    assert 0.0 <= preds[1] <= 1.0
    assert preds[0] <= preds[1]


def test_platt_calibrator_save_load(tmp_path) -> None:
    raw = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9] * 5)
    labels = np.array([0, 0, 0, 1, 1, 1] * 5)
    calibrator = ProbabilityCalibrator(method="platt")
    calibrator.fit(raw_probs=raw, labels=labels)

    path = tmp_path / "cal.joblib"
    calibrator.save(path)

    loaded = ProbabilityCalibrator()
    loaded.load(path)
    preds = loaded.predict(np.array([0.15, 0.85]))

    assert 0.0 <= preds[0] <= 1.0
    assert 0.0 <= preds[1] <= 1.0
    assert preds[0] <= preds[1]
