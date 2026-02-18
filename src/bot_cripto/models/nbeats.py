"""N-BEATS (Neural Basis Expansion Analysis) predictor.

Implements the N-BEATS architecture from pytorch-forecasting as a
``BasePredictor``.  N-BEATS is a univariate model that operates on
the ``close`` price series and outputs point forecasts with quantile
estimates derived from multiple stacks (trend + seasonality + generic).

The model uses the same TimeSeriesDataSet / Lightning training pipeline
as the TFT predictor for consistency.
"""

from __future__ import annotations

import warnings
from math import erf, sqrt
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

# MONKEYPATCH: PyTorch 2.6+ force weights_only=False to allow complex objects in checkpoints
import torch.serialization
original_load = torch.load
def patched_load(*args, **kwargs):
    if "weights_only" in kwargs:
        kwargs["weights_only"] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import NBeats, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE

from bot_cripto.core.config import get_settings
from bot_cripto.core.logging import get_logger
from bot_cripto.models.base import BasePredictor, ModelMetadata, PredictionOutput
from bot_cripto.models.calibration import CalibrationMethod, ProbabilityCalibrator

logger = get_logger("models.nbeats")

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*defined but not used.*")
warnings.filterwarnings("ignore", ".*isinstance\\(treespec, LeafSpec\\) is deprecated.*")


class NBeatsPredictor(BasePredictor):
    """Predictor based on the N-BEATS architecture.

    N-BEATS is a pure time-series model that decomposes the forecast
    into interpretable components (trend, seasonality) plus a generic
    stack for residual patterns.  It operates on a single target
    column (``close`` by default).

    Key differences from TFT:
    - Univariate: only uses the target column (no exogenous features).
    - Faster training and inference due to simpler architecture.
    - Interpretable decomposition into trend/seasonality components.
    - Complementary to TFT in ensemble — captures different patterns.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.settings = settings
        self.horizon = settings.pred_horizon_steps
        self.context_length = 60  # lookback window (shorter than TFT's 96)
        self.batch_size = 128
        self.max_epochs = 30
        self.learning_rate = 1e-3
        self.dropout = 0.1
        self.num_blocks = [3, 3, 3]  # per stack
        self.num_block_layers = [4, 4, 4]  # layers per block
        self.widths = [256, 256, 256]  # hidden size per stack
        self.expansion_coefficient_lengths = [5, 5, 5]
        self.backcast_loss_ratio = 0.1  # regularization via backcast
        self.num_workers = 2

        self.model: NBeats | None = None
        self.dataset_params: dict[str, Any] = {}
        self.residual_std: float = 0.01  # estimated from validation residuals
        self.probability_calibrator: ProbabilityCalibrator | None = None
        self.enable_probability_calibration = settings.enable_probability_calibration
        self.trainer_params: dict[str, Any] = {
            "accelerator": "auto",
            "devices": 1,
            "enable_progress_bar": False,
            "logger": False,
            "enable_checkpointing": True,
            "log_every_n_steps": 10,
            "gradient_clip_val": 0.5,
        }

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_data(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        data = df[[target_col]].copy().sort_index()
        data["time_idx"] = np.arange(len(data), dtype=int)
        data["group_id"] = "0"
        return data

    # ------------------------------------------------------------------
    # Probability helpers (same approach as TFT)
    # ------------------------------------------------------------------

    @staticmethod
    def _normal_cdf(x: float, mu: float, sigma: float) -> float:
        if sigma <= 1e-9:
            return 1.0 if x >= mu else 0.0
        z = (x - mu) / (sigma * sqrt(2.0))
        return 0.5 * (1.0 + erf(z))

    def _raw_probability_from_returns(
        self,
        p10_ret: float,
        p50_ret: float,
        p90_ret: float,
    ) -> float:
        sigma = max((p90_ret - p10_ret) / 2.5631031311, 1e-6)
        prob_up = 1.0 - self._normal_cdf(0.0, p50_ret, sigma)
        return float(min(max(prob_up, 0.0), 1.0))

    # ------------------------------------------------------------------
    # Quantile extraction
    # ------------------------------------------------------------------

    def _extract_point_prediction(self, data: pd.DataFrame) -> float:
        """Get point forecast for the last horizon step."""
        pred_dataset = TimeSeriesDataSet.from_parameters(
            self.dataset_params,
            data,
            predict=True,
            stop_randomization=True,
        )
        pred_loader = pred_dataset.to_dataloader(
            train=False,
            batch_size=1,
            num_workers=self.num_workers,
        )
        raw_predictions = self.model.predict(  # type: ignore[union-attr]
            pred_loader,
            mode="prediction",
            return_x=False,
            trainer_kwargs={
                "logger": False,
                "enable_progress_bar": False,
            },
        )
        preds = raw_predictions[0]
        return float(preds[-1])  # last horizon step

    def _derive_quantiles(self, point_pred: float) -> tuple[float, float, float]:
        """Derive p10/p50/p90 from point prediction using residual std."""
        sigma = self.residual_std
        p10 = point_pred - 1.2816 * sigma  # z=1.2816 for 10th percentile
        p50 = point_pred
        p90 = point_pred + 1.2816 * sigma
        return p10, p50, p90

    def _estimate_residual_std(
        self,
        data: pd.DataFrame,
        target_col: str,
        start_idx: int,
        max_samples: int = 50,
    ) -> float:
        """Estimate prediction residual std from validation window."""
        frame = data.copy()
        limit = len(frame) - self.horizon - 1
        start = max(self.context_length + self.horizon, start_idx)
        if limit <= start + 5:
            return 0.01

        step = max(1, (limit - start) // max_samples)
        residuals: list[float] = []

        for idx in range(start, limit + 1, step):
            window = frame.iloc[: idx + 1]
            prepared = self._prepare_data(window, target_col)
            try:
                pred_price = self._extract_point_prediction(prepared)
            except Exception:
                continue
            actual_price = float(frame[target_col].iloc[idx + self.horizon])
            residuals.append(pred_price - actual_price)

        if len(residuals) < 3:
            return 0.01
        return float(np.std(residuals))

    # ------------------------------------------------------------------
    # Probability calibration
    # ------------------------------------------------------------------

    def _fit_probability_calibrator(
        self,
        raw_df: pd.DataFrame,
        target_col: str,
        start_idx: int,
    ) -> dict[str, float]:
        if not self.enable_probability_calibration:
            return {}
        if self.model is None:
            return {}

        frame = raw_df.copy().sort_index()
        limit = len(frame) - self.horizon - 1
        start = max(self.context_length, start_idx)
        if limit <= start + 20:
            return {}

        max_samples = self.settings.tft_calibration_max_samples
        step = max(1, (limit - start) // max_samples)
        raw_probs: list[float] = []
        labels: list[int] = []

        for idx in range(start, limit + 1, step):
            window = frame.iloc[: idx + 1]
            prepared = self._prepare_data(window, target_col)
            try:
                point_pred = self._extract_point_prediction(prepared)
                p10_price, p50_price, p90_price = self._derive_quantiles(point_pred)
            except Exception as exc:
                logger.debug("nbeats_calibration_sample_skip", error=str(exc))
                continue

            current_price = float(window[target_col].iloc[-1])
            if abs(current_price) < 1e-9:
                continue
            p10_ret = (p10_price - current_price) / abs(current_price)
            p50_ret = (p50_price - current_price) / abs(current_price)
            p90_ret = (p90_price - current_price) / abs(current_price)
            raw_probs.append(self._raw_probability_from_returns(p10_ret, p50_ret, p90_ret))

            future_price = float(frame[target_col].iloc[idx + self.horizon])
            actual_price = float(frame[target_col].iloc[idx])
            labels.append(1 if future_price > actual_price else 0)

        if len(raw_probs) < 20 or len(set(labels)) < 2:
            return {}

        method: CalibrationMethod = (
            "isotonic" if self.settings.probability_calibration_method != "platt" else "platt"
        )
        calibrator = ProbabilityCalibrator(method=method)
        try:
            metrics = calibrator.fit(
                raw_probs=np.array(raw_probs, dtype=float),
                labels=np.array(labels, dtype=int),
            )
        except ValueError:
            return {}
        self.probability_calibrator = calibrator
        return {
            "calibration_samples": float(metrics.samples),
            "brier_before": metrics.brier_before,
            "brier_after": metrics.brier_after,
        }

    # ------------------------------------------------------------------
    # BasePredictor interface
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame, target_col: str = "close") -> ModelMetadata:
        log = logger.bind(rows=len(df))
        log.info("nbeats_training_start")

        data = self._prepare_data(df, target_col)
        min_length = self.context_length + self.horizon + 1
        if len(data) < min_length:
            raise ValueError(
                f"Insufficient data for N-BEATS. Need: {min_length}, Got: {len(data)}"
            )
        self.probability_calibrator = None

        # Split: training | calibration holdout
        holdout_count = int(len(data) * self.settings.tft_calibration_holdout_ratio)
        calibration_start_idx = max(
            self.context_length + self.horizon + 1,
            len(data) - holdout_count,
        )
        training_data = data.iloc[:calibration_start_idx].copy()
        training_cutoff = int(training_data["time_idx"].max()) - self.horizon

        if training_cutoff <= self.context_length:
            raise ValueError("Insufficient data for N-BEATS temporal split")

        # Build TimeSeriesDataSet (univariate — only target column)
        # GroupNormalizer handles target scaling internally (no external scaler)
        training_dataset = TimeSeriesDataSet(
            training_data[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target=target_col,
            group_ids=["group_id"],
            min_encoder_length=self.context_length,
            max_encoder_length=self.context_length,
            min_prediction_length=self.horizon,
            max_prediction_length=self.horizon,
            time_varying_known_reals=[],
            time_varying_unknown_reals=[target_col],
            target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
            add_relative_time_idx=False,
            add_target_scales=False,
            add_encoder_length=False,
            allow_missing_timesteps=True,
        )

        self.dataset_params = training_dataset.get_parameters()
        train_loader = training_dataset.to_dataloader(
            train=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        validation = TimeSeriesDataSet.from_dataset(
            training_dataset,
            data,
            predict=True,
            stop_randomization=True,
        )
        val_loader = validation.to_dataloader(
            train=False,
            batch_size=self.batch_size * 10,
            num_workers=self.num_workers,
        )

        # Build N-BEATS model with interpretable stacks
        self.model = NBeats.from_dataset(
            training_dataset,
            stack_types=["trend", "seasonality", "generic"],
            num_blocks=self.num_blocks,
            num_block_layers=self.num_block_layers,
            widths=self.widths,
            expansion_coefficient_lengths=self.expansion_coefficient_lengths,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            backcast_loss_ratio=self.backcast_loss_ratio,
            loss=MAE(),
            log_interval=10,
            reduce_on_plateau_patience=5,
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min",
        )

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=[early_stop],
            **self.trainer_params,
        )
        trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        val_loss = float(trainer.callback_metrics.get("val_loss", torch.tensor(0.0)).item())

        # Estimate residual std from validation window for quantile derivation
        self.residual_std = self._estimate_residual_std(
            data, target_col, start_idx=calibration_start_idx
        )

        calibration_metrics = self._fit_probability_calibrator(
            df, target_col, start_idx=calibration_start_idx
        )
        log.info(
            "nbeats_training_done",
            val_loss=val_loss,
            residual_std=round(self.residual_std, 6),
            **calibration_metrics,
        )

        return ModelMetadata.create(
            model_type="nbeats",
            version="0.1.0",
            metrics={"val_loss": val_loss, **calibration_metrics},
        )

    def predict(self, df: pd.DataFrame) -> PredictionOutput:
        if self.model is None or not self.dataset_params:
            raise ValueError("N-BEATS model not trained/loaded")

        target_col = "close"
        data = self._prepare_data(df, target_col)
        if len(data) < self.context_length:
            raise ValueError(
                f"Insufficient data for prediction. Need: {self.context_length}, Got: {len(data)}"
            )

        current_price = float(df[target_col].iloc[-1])
        point_pred = self._extract_point_prediction(data)
        p10_price, p50_price, p90_price = self._derive_quantiles(point_pred)

        def calc_ret(price: float) -> float:
            return float((price - current_price) / current_price)

        expected_ret = calc_ret(p50_price)
        p10_ret = calc_ret(p10_price)
        p50_ret = expected_ret
        p90_ret = calc_ret(p90_price)

        prob_up = self._raw_probability_from_returns(p10_ret, p50_ret, p90_ret)
        if self.probability_calibrator is not None:
            prob_up = float(self.probability_calibrator.predict(np.array([prob_up]))[0])

        spread = p90_ret - p10_ret
        risk_score = min(max(spread / self.settings.model_risk_spread_ref, 0.0), 1.0)

        return PredictionOutput(
            prob_up=float(prob_up),
            expected_return=float(expected_ret),
            p10=float(p10_ret),
            p50=float(p50_ret),
            p90=float(p90_ret),
            risk_score=float(risk_score),
        )

    def save(self, path: Path) -> None:
        if self.model is None:
            raise ValueError("N-BEATS model not trained.")

        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.dataset_params, path / "dataset_params.joblib")
        joblib.dump(self.residual_std, path / "residual_std.joblib")
        if self.probability_calibrator is not None:
            self.probability_calibrator.save(path / "probability_calibrator.joblib")
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "hparams": self.model.hparams,
            },
            path / "model.pt",
        )
        logger.info("nbeats_model_saved", path=str(path))

    def load(self, path: Path) -> None:
        params_path = path / "dataset_params.joblib"
        calibrator_path = path / "probability_calibrator.joblib"
        model_path = path / "model.pt"

        if not params_path.exists() or not model_path.exists():
            raise FileNotFoundError(f"N-BEATS artifacts not found in {path}")

        self.dataset_params = joblib.load(params_path)
        residual_path = path / "residual_std.joblib"
        if residual_path.exists():
            self.residual_std = joblib.load(residual_path)
        if calibrator_path.exists():
            calibrator = ProbabilityCalibrator()
            calibrator.load(calibrator_path)
            self.probability_calibrator = calibrator
        else:
            self.probability_calibrator = None

        checkpoint = torch.load(model_path, weights_only=False)
        hparams = checkpoint["hparams"]
        state_dict = checkpoint["state_dict"]
        self.model = NBeats(**hparams)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info("nbeats_model_loaded", path=str(path))
