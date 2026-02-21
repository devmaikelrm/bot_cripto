"""Baseline model implementation using scikit-learn."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

from bot_cripto.core.config import get_settings
from bot_cripto.core.logging import get_logger
from bot_cripto.models.base import BasePredictor, ModelMetadata, PredictionOutput
from bot_cripto.models.calibration import CalibrationMethod, ProbabilityCalibrator

logger = get_logger("models.baseline")


class BaselineModel(BasePredictor):
    """RandomForest baseline with optional single-objective training mode."""

    def __init__(self, objective: Literal["multi", "trend", "return", "risk"] = "multi") -> None:
        self.settings = get_settings()
        self.horizon = self.settings.pred_horizon_steps
        self.objective = objective

        self.direction_model: BaseEstimator = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
        )
        self.return_model: BaseEstimator = RandomForestRegressor(
            n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
        )
        self.risk_model: BaseEstimator = RandomForestRegressor(
            n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
        )
        self.features: list[str] = []
        self.probability_calibrator: ProbabilityCalibrator | None = None

    def _prepare_targets(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        frame = df.copy()
        use_tb_label = "tb_label" in frame.columns and self.objective in {"trend"}
        use_tb_ret = "tb_ret" in frame.columns and self.objective in {"return", "trend"}

        future_close = frame[target_col].shift(-self.horizon)
        frame["target_return"] = (future_close - frame[target_col]) / frame[target_col]
        if use_tb_ret:
            frame["target_return"] = frame["tb_ret"].astype(float)

        # Direction label with an "edge band" to reduce label noise.
        # For trend/multi objectives we drop samples where the future move is too small to matter.
        if use_tb_label:
            # Triple-barrier label: +1 => up, -1 => down, 0 => no edge (drop)
            direction = np.where(
                frame["tb_label"] > 0,
                1,
                np.where(frame["tb_label"] < 0, 0, np.nan),
            )
            frame["target_direction"] = direction.astype(float)
        else:
            edge = float(self.settings.label_edge_return)
            if self.objective in {"multi", "trend"} and edge > 0:
                direction = np.where(
                    frame["target_return"] > edge,
                    1,
                    np.where(frame["target_return"] < -edge, 0, np.nan),
                )
                frame["target_direction"] = direction
            else:
                frame["target_direction"] = (frame["target_return"] > 0).astype(int)

        if "log_ret" not in frame.columns:
            frame["log_ret"] = np.log(frame[target_col] / frame[target_col].shift(1))

        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=self.horizon)
        # Forward volatility proxy: std of log-returns in the horizon window.
        # This makes target_risk interpretable as a sigma, so p10/p90 computed from mu +/- z*sigma is consistent.
        frame["target_risk"] = frame["log_ret"].rolling(window=indexer).std(ddof=0)
        return frame.dropna()

    def _fit_models(
        self,
        x_train: pd.DataFrame,
        y_dir: pd.Series,
        y_ret: pd.Series,
        y_risk: pd.Series,
    ) -> None:
        if self.objective == "trend":
            self.direction_model.fit(x_train, y_dir)
            self.return_model = DummyRegressor(strategy="mean")
            self.return_model.fit(x_train, y_ret)
            self.risk_model = DummyRegressor(strategy="mean")
            self.risk_model.fit(x_train, y_risk)
            return

        if self.objective == "return":
            self.direction_model = DummyClassifier(strategy="prior")
            self.direction_model.fit(x_train, y_dir)
            self.return_model.fit(x_train, y_ret)
            self.risk_model = DummyRegressor(strategy="mean")
            self.risk_model.fit(x_train, y_risk)
            return

        if self.objective == "risk":
            self.direction_model = DummyClassifier(strategy="prior")
            self.direction_model.fit(x_train, y_dir)
            self.return_model = DummyRegressor(strategy="mean")
            self.return_model.fit(x_train, y_ret)
            self.risk_model.fit(x_train, y_risk)
            return

        self.direction_model.fit(x_train, y_dir)
        self.return_model.fit(x_train, y_ret)
        self.risk_model.fit(x_train, y_risk)

    def train(self, df: pd.DataFrame, target_col: str = "close") -> ModelMetadata:
        log = logger.bind(rows=len(df), horizon=self.horizon, objective=self.objective)
        log.info("baseline_train_start")

        exclude = {
            "timestamp",
            "date",
            "open_time",
            "close_time",
            "target_return",
            "target_direction",
            "target_risk",
            target_col,
        }
        self.features = [
            col
            for col in df.columns
            if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
        ]

        train_df = self._prepare_targets(df, target_col)
        if train_df.empty:
            raise ValueError("No hay datos suficientes para entrenar (targets generaron NaNs)")

        x_all = train_df[self.features]
        y_dir_all = train_df["target_direction"].astype(int)
        y_ret_all = train_df["target_return"]
        y_risk_all = train_df["target_risk"]

        # Time-based split to avoid look-ahead bias in evaluation/calibration.
        split = int(len(train_df) * 0.8)
        if split < 200 or (len(train_df) - split) < 50:
            split = len(train_df)

        x_train = x_all.iloc[:split]
        y_dir = y_dir_all.iloc[:split]
        y_ret = y_ret_all.iloc[:split]
        y_risk = y_risk_all.iloc[:split]

        x_val = x_all.iloc[split:] if split < len(train_df) else None
        y_dir_val = y_dir_all.iloc[split:] if split < len(train_df) else None
        y_ret_val = y_ret_all.iloc[split:] if split < len(train_df) else None

        self._fit_models(x_train, y_dir, y_ret, y_risk)
        calibration_metrics: dict[str, float] = {}
        if (
            self.settings.enable_probability_calibration
            and self.objective in {"multi", "trend"}
            and x_val is not None
            and y_dir_val is not None
            and len(x_val) >= 200
            and int(y_dir_val.nunique()) >= 2
        ):
            calibration_metrics = self._fit_probability_calibrator(x_val, y_dir_val)

        acc_train = accuracy_score(y_dir, self.direction_model.predict(x_train))
        mae_train = mean_absolute_error(y_ret, self.return_model.predict(x_train))
        metrics: dict[str, float] = {
            # Backwards-compatible metric keys (tests + existing dashboards).
            "accuracy_in_sample": float(acc_train),
            "mae_return_in_sample": float(mae_train),
            # Also expose explicit "train" names.
            "accuracy_train": float(acc_train),
            "mae_return_train": float(mae_train),
            "objective_code": float(
                {"multi": 0, "trend": 1, "return": 2, "risk": 3}[self.objective]
            ),
            "using_tb_label": float(1.0 if ("tb_label" in train_df.columns and self.objective == "trend") else 0.0),
            "using_tb_return": float(
                1.0 if ("tb_ret" in train_df.columns and self.objective in {"trend", "return"}) else 0.0
            ),
            **calibration_metrics,
        }

        if x_val is not None and y_dir_val is not None and y_ret_val is not None and len(x_val) > 0:
            try:
                acc_val = accuracy_score(y_dir_val, self.direction_model.predict(x_val))
                mae_val = mean_absolute_error(y_ret_val, self.return_model.predict(x_val))
                metrics["accuracy_val"] = float(acc_val)
                metrics["mae_return_val"] = float(mae_val)
            except Exception:
                # Keep training robust if a dummy model or edge case fails metrics.
                pass

        log.info("baseline_train_done", **metrics)
        return ModelMetadata.create(model_type="baseline_rf", version="0.1.0", metrics=metrics)

    def _fit_probability_calibrator(
        self, x_train: pd.DataFrame, y_dir: pd.Series
    ) -> dict[str, float]:
        if len(x_train) < 80 or int(y_dir.nunique()) < 2:
            return {}

        probs = self.direction_model.predict_proba(x_train)
        if probs.shape[1] < 2:
            return {}

        class_one = np.where(self.direction_model.classes_ == 1)[0]
        if len(class_one) == 0:
            return {}
        raw_probs = probs[:, int(class_one[0])]
        labels = y_dir.to_numpy(dtype=int)
        method: CalibrationMethod = (
            "isotonic" if self.settings.probability_calibration_method != "platt" else "platt"
        )
        calibrator = ProbabilityCalibrator(method=method)
        try:
            metrics = calibrator.fit(raw_probs=raw_probs, labels=labels)
        except ValueError:
            return {}
        self.probability_calibrator = calibrator
        return {
            "calibration_samples": float(metrics.samples),
            "brier_before": metrics.brier_before,
            "brier_after": metrics.brier_after,
        }

    def predict(self, df: pd.DataFrame) -> PredictionOutput:
        if not self.features:
            raise ValueError("Modelo no entrenado")

        last_row = df.iloc[[-1]][self.features]
        proba = self.direction_model.predict_proba(last_row)

        if proba.shape[1] == 1:
            learned_class = self.direction_model.classes_[0]
            prob_up = 1.0 if learned_class == 1 else 0.0
        else:
            idx_1 = np.where(self.direction_model.classes_ == 1)[0]
            prob_up = proba[0][int(idx_1[0])] if len(idx_1) > 0 else 0.0

        expected_ret = float(self.return_model.predict(last_row)[0])
        pred_risk = float(self.risk_model.predict(last_row)[0])

        risk_score = min(max(pred_risk / self.settings.model_risk_vol_ref, 0.0), 1.0)

        # BTC returns have fat tails (empirical kurtosis >> 3); using the Normal
        # quantile multiplier 1.28 systematically underestimates the 10th/90th
        # percentile spread.  A Student-t with df=4 gives ≈1.53σ at the 90th
        # percentile, which is more conservative and better calibrated for crypto.
        sigma = pred_risk
        _FAT_TAIL_MULT = 1.53  # t(df=4).ppf(0.90), vs 1.28 for Normal
        p10 = expected_ret - _FAT_TAIL_MULT * sigma
        p50 = expected_ret
        p90 = expected_ret + _FAT_TAIL_MULT * sigma

        if self.probability_calibrator is not None:
            prob_up = float(self.probability_calibrator.predict(np.array([prob_up]))[0])

        return PredictionOutput(
            prob_up=float(prob_up),
            expected_return=float(expected_ret),
            p10=float(p10),
            p50=float(p50),
            p90=float(p90),
            risk_score=float(risk_score),
        )

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.direction_model, path / "direction.joblib")
        joblib.dump(self.return_model, path / "return.joblib")
        joblib.dump(self.risk_model, path / "risk.joblib")
        joblib.dump(self.features, path / "features.joblib")
        joblib.dump(self.objective, path / "objective.joblib")
        if self.probability_calibrator is not None:
            self.probability_calibrator.save(path / "probability_calibrator.joblib")
        logger.info("baseline_saved", path=str(path))

    def load(self, path: Path) -> None:
        if not (path / "features.joblib").exists():
            raise FileNotFoundError(f"No se encontro modelo en {path}")

        self.direction_model = joblib.load(path / "direction.joblib")
        self.return_model = joblib.load(path / "return.joblib")
        self.risk_model = joblib.load(path / "risk.joblib")
        self.features = joblib.load(path / "features.joblib")
        objective_path = path / "objective.joblib"
        if objective_path.exists():
            loaded = joblib.load(objective_path)
            if loaded in {"multi", "trend", "return", "risk"}:
                self.objective = cast(Literal["multi", "trend", "return", "risk"], loaded)
        calibrator_path = path / "probability_calibrator.joblib"
        if calibrator_path.exists():
            calibrator = ProbabilityCalibrator()
            calibrator.load(calibrator_path)
            self.probability_calibrator = calibrator
        else:
            self.probability_calibrator = None
        logger.info("baseline_loaded", path=str(path), objective=self.objective)
