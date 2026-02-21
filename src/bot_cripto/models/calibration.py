"""Probability calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

CalibrationMethod = Literal["isotonic", "platt"]


@dataclass(frozen=True)
class CalibrationMetrics:
    samples: int
    brier_before: float
    brier_after: float


class ProbabilityCalibrator:
    """Calibrate raw probabilities to improve reliability."""

    def __init__(self, method: CalibrationMethod = "isotonic") -> None:
        self.method = method
        self._model: IsotonicRegression | LogisticRegression | None = None

    @staticmethod
    def _clip(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, 1e-6, 1 - 1e-6)
        return cast(np.ndarray, clipped)

    # Isotonic regression overfits with < 200 samples (non-parametric, too many knots).
    # Below this threshold we fall back to Platt scaling (logistic regression).
    _ISOTONIC_MIN_SAMPLES = 200

    def fit(self, raw_probs: np.ndarray, labels: np.ndarray) -> CalibrationMetrics:
        probs = self._clip(raw_probs.astype(float).ravel())
        y = labels.astype(int).ravel()
        if len(probs) != len(y):
            raise ValueError("raw_probs and labels must have same length")
        if len(probs) < 20 or len(np.unique(y)) < 2:
            raise ValueError("insufficient data for probability calibration")

        brier_before = float(brier_score_loss(y, probs))

        # Auto-downgrade isotonic → Platt when sample count is too small to fit
        # a reliable monotone regression without severe overfitting.
        effective_method = self.method
        if self.method == "isotonic" and len(probs) < self._ISOTONIC_MIN_SAMPLES:
            effective_method = "platt"

        if effective_method == "isotonic":
            model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            model.fit(probs, y)
            calibrated = model.predict(probs)
            self._model = model
        else:
            model = LogisticRegression()
            model.fit(probs.reshape(-1, 1), y)
            calibrated = model.predict_proba(probs.reshape(-1, 1))[:, 1]
            self._model = model

        brier_after = float(brier_score_loss(y, self._clip(calibrated)))
        return CalibrationMetrics(
            samples=len(probs),
            brier_before=brier_before,
            brier_after=brier_after,
        )

    def predict(self, raw_probs: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValueError("calibrator not fitted")
        probs = self._clip(raw_probs.astype(float).ravel())

        if isinstance(self._model, IsotonicRegression):
            calibrated = self._model.predict(probs)
        else:
            calibrated = self._model.predict_proba(probs.reshape(-1, 1))[:, 1]
        return self._clip(calibrated)

    def save(self, path: Path) -> None:
        if self._model is None:
            raise ValueError("calibrator not fitted")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"method": self.method, "model": self._model}, path)

    def load(self, path: Path) -> None:
        payload = joblib.load(path)
        method = payload.get("method")
        model = payload.get("model")
        if method not in {"isotonic", "platt"}:
            raise ValueError("invalid calibrator payload")
        self.method = method
        self._model = model
