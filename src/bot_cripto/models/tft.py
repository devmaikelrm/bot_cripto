"""Temporal Fusion Transformer predictor."""

from __future__ import annotations

import os
import warnings
from math import erf, sqrt
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.serialization

from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, TorchNormalizer
from pytorch_forecasting.metrics import QuantileLoss, MultiHorizonMetric
from sklearn.preprocessing import RobustScaler

# Add GroupNormalizer to safe globals for torch.load (PyTorch 2.6+ requirement)
if hasattr(torch.serialization, "add_safe_globals"):
    import sklearn.preprocessing
    import pandas.core.internals.managers
    import numpy as np
    import pandas._libs.internals
    torch.serialization.add_safe_globals([
        GroupNormalizer, 
        pd.DataFrame, 
        pd.Series,
        pd.Index,
        sklearn.preprocessing.RobustScaler,
        pandas.core.internals.managers.BlockManager,
        pandas._libs.internals._unpickle_block,
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        np.core.multiarray.scalar,
    ])

from bot_cripto.core.config import get_settings
from bot_cripto.core.logging import get_logger
from bot_cripto.models.base import BasePredictor, ModelMetadata, PredictionOutput
from bot_cripto.models.calibration import CalibrationMethod, ProbabilityCalibrator

logger = get_logger("models.tft")

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*defined but not used.*")
warnings.filterwarnings("ignore", ".*Attribute 'loss' is an instance of `nn.Module`.*")
warnings.filterwarnings("ignore", ".*Attribute 'logging_metrics' is an instance of `nn.Module`.*")
warnings.filterwarnings("ignore", ".*isinstance\\(treespec, LeafSpec\\) is deprecated.*")


class SharpeAwareLoss(MultiHorizonMetric):
    """Custom loss function optimized for Sharpe Ratio."""
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2, **kwargs):
        """
        Args:
            alpha: Weight for MAE (Mean Absolute Error)
            beta: Weight for Directional Accuracy
            gamma: Weight for Risk-Adjusted Returns (Volatility penalty)
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def loss(self, y_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
        # y_pred shape: [batch, horizon, quantiles] - we use median (index 1) for main prediction
        # y_actual shape: [batch, horizon]
        
        # 1. Mean Absolute Error (Precision)
        pred_median = y_pred[..., 1] # Assuming quantile 0.5 is at index 1
        mae = torch.mean(torch.abs(pred_median - y_actual))
        
        # 2. Directional Accuracy Loss
        # We want predicted direction to match actual direction relative to previous step
        # Since y_actual is raw value (price/return), we need diffs.
        # Approximation: sign(pred - mean) vs sign(actual - mean) or just raw values if standardized
        # Better: Soft directional penalty using tanh or sigmoid
        target_direction = torch.sign(y_actual) 
        pred_direction = torch.tanh(pred_median * 10) # Soft sign
        # Loss is 0 if signs match, >0 if mismatch. 
        # (1 - product) / 2 ranges from 0 (match) to 1 (mismatch)
        dir_loss = torch.mean(1 - (target_direction * pred_direction))
        
        # 3. Risk-Adjusted Penalty (Volatility)
        # Penalize high variance in predictions when target is stable
        # or simply punish negative returns more (Sortino-like)
        # Here we penalize volatility of the error residuals
        residuals = pred_median - y_actual
        vol_penalty = torch.std(residuals)
        
        total_loss = (self.alpha * mae) + (self.beta * dir_loss) + (self.gamma * vol_penalty)
        return total_loss


class TFTPredictor(BasePredictor):
    """Predictor based on Temporal Fusion Transformer."""

    def __init__(self) -> None:
        settings = get_settings()
        self.settings = settings
        self.horizon = settings.pred_horizon_steps
        self.encoder_length = int(getattr(settings, "tft_encoder_length", 288))
        self.batch_size = int(getattr(settings, "tft_batch_size", 256))
        self.max_epochs = 30 # Increased for better convergence
        self.learning_rate = 1e-3 # Slightly lower for stability with custom loss
        self.hidden_size = 128
        self.attention_head_size = 8
        self.dropout = 0.2 # Increased dropout for regularization
        self.hidden_continuous_size = 64
        self.lstm_layers = 3 # New parameter: Deeper network
        self.quantiles = [0.1, 0.5, 0.9]
        self.num_workers = int(getattr(settings, "tft_num_workers", 4))
        self.early_stopping_patience = int(getattr(settings, "tft_early_stopping_patience", 5))
        self.early_stopping_min_delta = float(getattr(settings, "tft_early_stopping_min_delta", 1e-4))

        if torch.cuda.is_available():
            try:
                mem_gb = float(torch.cuda.get_device_properties(0).total_memory) / (1024**3)
                if mem_gb < 16:
                    self.batch_size = min(self.batch_size, 128)
                    self.num_workers = min(self.num_workers, 4)
                elif mem_gb < 24:
                    self.batch_size = min(self.batch_size, 256)
                    self.num_workers = min(self.num_workers, 6)
            except Exception:
                pass
        else:
            self.batch_size = min(self.batch_size, 64)
            self.num_workers = min(self.num_workers, 2)

        self.batch_size = int(os.getenv("TFT_BATCH_SIZE", str(self.batch_size)))
        self.num_workers = int(os.getenv("TFT_NUM_WORKERS", str(self.num_workers)))
        self.early_stopping_patience = int(
            os.getenv("TFT_EARLY_STOPPING_PATIENCE", str(self.early_stopping_patience))
        )
        self.early_stopping_min_delta = float(
            os.getenv("TFT_EARLY_STOPPING_MIN_DELTA", str(self.early_stopping_min_delta))
        )
        accelerator = str(getattr(settings, "tft_accelerator", "auto"))
        if accelerator == "auto":
            accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        precision = str(getattr(settings, "tft_precision", "bf16-mixed"))
        if accelerator == "cpu" and precision != "32-true":
            precision = "32-true"

        self.model: TemporalFusionTransformer | None = None
        self.dataset_params: dict[str, Any] = {}
        self.trainer_params: dict[str, Any] = {
            "accelerator": accelerator,
            "devices": 1,
            "precision": precision,
            "enable_progress_bar": True,
            "logger": True,
            "enable_checkpointing": True,
            "log_every_n_steps": 10,
            "gradient_clip_val": 0.1,
        }
        self.feature_scalers: dict[str, RobustScaler] = {}
        self.scaled_columns: list[str] = []
        self.probability_calibrator: ProbabilityCalibrator | None = None
        self.enable_probability_calibration = settings.enable_probability_calibration

    def _prepare_data(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        _ = target_col
        data = df.copy().sort_index()
        data["time_idx"] = np.arange(len(data), dtype=int)
        data["group_id"] = "0"
        if "day_of_week" in data.columns:
            data["day_of_week"] = data["day_of_week"].astype(str).astype("category")
        if "sentiment_regime" in data.columns:
            data["sentiment_regime"] = data["sentiment_regime"].astype(str).astype("category")
        else:
            data["sentiment_regime"] = "APATHY"
            data["sentiment_regime"] = data["sentiment_regime"].astype("category")
        return data

    def _fit_scalers(
        self,
        data: pd.DataFrame,
        cutoff_idx: int,
        target_col: str,
        columns: list[str],
    ) -> None:
        self.feature_scalers = {}
        self.scaled_columns = []
        training_frame = data[data["time_idx"] <= cutoff_idx]
        for column in columns:
            if column == target_col:
                continue
            scaler = RobustScaler()
            train_values = training_frame[[column]].astype(float).to_numpy()
            if len(train_values) == 0:
                continue
            scaler.fit(train_values)
            self.feature_scalers[column] = scaler
            self.scaled_columns.append(column)

    def _apply_scalers(self, data: pd.DataFrame) -> pd.DataFrame:
        out = data.copy()
        for column, scaler in self.feature_scalers.items():
            if column in out.columns:
                out[column] = scaler.transform(out[[column]].astype(float).to_numpy()).ravel()
        return out

    @staticmethod
    def _normal_cdf(x: float, mu: float, sigma: float) -> float:
        if sigma <= 1e-9:
            return 1.0 if x >= mu else 0.0
        z = (x - mu) / (sigma * sqrt(2.0))
        return 0.5 * (1.0 + erf(z))

    def _extract_quantile_prediction(self, data: pd.DataFrame) -> tuple[float, float, float]:
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
            mode="quantiles",
            return_x=False,
            trainer_kwargs={
                "logger": False,
                "enable_progress_bar": False,
            },
        )
        preds = raw_predictions[0]
        step_idx = -1
        p10_price = float(preds[step_idx, 0])
        p50_price = float(preds[step_idx, 1])
        p90_price = float(preds[step_idx, 2])
        return p10_price, p50_price, p90_price

    def _raw_probability_from_returns(
        self,
        p10_ret: float,
        p50_ret: float,
        p90_ret: float,
    ) -> float:
        sigma = max((p90_ret - p10_ret) / 2.5631031311, 1e-6)
        prob_up = 1.0 - self._normal_cdf(0.0, p50_ret, sigma)
        return float(min(max(prob_up, 0.0), 1.0))

    def _fit_probability_calibrator(
        self,
        raw_df: pd.DataFrame,
        start_idx: int,
    ) -> dict[str, float]:
        if not self.enable_probability_calibration:
            return {}
        if self.model is None:
            return {}

        frame = raw_df.copy().sort_index()
        limit = len(frame) - self.horizon - 1
        start = max(self.encoder_length, start_idx)
        if limit <= start + 20:
            return {}

        max_samples = self.settings.tft_calibration_max_samples
        step = max(1, (limit - start) // max_samples)
        raw_probs: list[float] = []
        labels: list[int] = []

        for idx in range(start, limit + 1, step):
            window = frame.iloc[: idx + 1]
            target_name = str(self.dataset_params.get("target", "close"))
            prepared = self._prepare_data(window, target_name)
            prepared = self._apply_scalers(prepared)
            try:
                p10_pred, p50_pred, p90_pred = self._extract_quantile_prediction(prepared)
            except Exception as exc:
                logger.debug("tft_calibration_sample_skip", error=str(exc))
                continue

            current_price = float(window["close"].iloc[-1])
            if target_name == "log_ret":
                p10_ret = float(np.expm1(p10_pred))
                p50_ret = float(np.expm1(p50_pred))
                p90_ret = float(np.expm1(p90_pred))
            else:
                p10_ret = (p10_pred - current_price) / current_price
                p50_ret = (p50_pred - current_price) / current_price
                p90_ret = (p90_pred - current_price) / current_price
            raw_probs.append(self._raw_probability_from_returns(p10_ret, p50_ret, p90_ret))

            future_price = float(frame["close"].iloc[idx + self.horizon])
            labels.append(1 if future_price > current_price else 0)

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

    def train_with_checkpoint(
        self, df: pd.DataFrame, checkpoint_path: str, target_col: str = "close"
    ) -> ModelMetadata:
        """Resume training from a checkpoint."""
        return self.train(df, target_col=target_col, resume_from=checkpoint_path)

    def train(
        self, df: pd.DataFrame, target_col: str = "close", resume_from: str | None = None
    ) -> ModelMetadata:
        torch.set_num_threads(6)
        if torch.cuda.is_available():
            # Better Tensor Core utilization on modern NVIDIA GPUs (e.g., RTX 4090).
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        log = logger.bind(rows=len(df))
        log.info("iniciando_entrenamiento_tft_mejorado")

        target_mode = str(getattr(self.settings, "tft_target_mode", "log_return")).lower()
        train_target = "log_ret" if target_mode == "log_return" and "log_ret" in df.columns else target_col
        data = self._prepare_data(df, train_target)
        min_length = self.encoder_length + self.horizon + 1
        if len(data) < min_length:
            raise ValueError(
                f"Datos insuficientes para TFT. Req: {min_length}, Actual: {len(data)}"
            )
        self.probability_calibrator = None

        valid_reals = {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "rsi",
            "volatility",
            "macd",
            "atr",
            "log_ret",
            "rel_vol",
            "obi_score",
            "whale_score",
            "sentiment_score",
            "macro_spy_close",
            "macro_qqq_close",
            "macro_dx_y_nyb_close",
            "macro_gc_f_close",
            "micro_volume_imbalance",
            "micro_price_impact",
            "micro_amihud",
            "micro_kyle_lambda",
            "micro_parkinson_vol",
            "micro_gk_vol",
            "micro_rs_vol",
            "micro_roll_spread",
            "micro_jump_score",
            "micro_upper_wick_ratio",
            "micro_lower_wick_ratio",
            "micro_vwap_deviation",
        }
        unknown_reals = [c for c in df.columns if c in valid_reals and c != train_target]
        if train_target not in unknown_reals:
            unknown_reals.append(train_target)

        holdout_count = int(len(data) * self.settings.tft_calibration_holdout_ratio)
        calibration_start_idx = max(
            self.encoder_length + self.horizon + 1,
            len(data) - holdout_count,
        )
        training_data = data.iloc[:calibration_start_idx].copy()
        training_cutoff = int(training_data["time_idx"].max()) - self.horizon

        if training_cutoff <= self.encoder_length:
            raise ValueError("Datos insuficientes para split temporal de calibracion TFT")

        self._fit_scalers(training_data, training_cutoff, target_col, unknown_reals)
        training_data = self._apply_scalers(training_data)
        data_scaled = self._apply_scalers(data)
        known_cats = []
        if "day_of_week" in data.columns:
            known_cats.append("day_of_week")
        if "sentiment_regime" in data.columns:
            known_cats.append("sentiment_regime")

        training_dataset = TimeSeriesDataSet(
            training_data[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target=train_target,
            group_ids=["group_id"],
            min_encoder_length=self.encoder_length // 2,
            max_encoder_length=self.encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.horizon,
            static_categoricals=[],
            static_reals=[],
            time_varying_known_categoricals=known_cats,
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=unknown_reals,
            target_normalizer=TorchNormalizer(method="standard"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
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
            data_scaled,
            predict=True,
            stop_randomization=True,
        )
        val_loader = validation.to_dataloader(
            train=False,
            batch_size=self.batch_size * 10,
            num_workers=self.num_workers,
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=False,
            filename="best-{epoch:02d}-{val_loss:.6f}",
        )
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
            verbose=True,
        )
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=[checkpoint_callback, early_stopping_callback],
            **self.trainer_params,
        )
        
        # Use QuantileLoss as main loss but keep structure prepared for Sharpe
        # Note: SharpeAwareLoss requires point predictions or specific quantile handling.
        # For this iteration, we stick to QuantileLoss for stability with the new architecture,
        # as SharpeLoss is harder to converge without extensive tuning on the exact return series structure.
        # The PDF suggests SharpeAwareLoss, but implementing it as a direct replacement for QuantileLoss
        # in TemporalFusionTransformer class is tricky because TFT output is strictly quantiles.
        # Strategy: Use QuantileLoss for training to ensure distributional learning,
        # but increase model capacity to capture the complexities needed for Sharpe optimization later.
        
        self.model = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            lstm_layers=self.lstm_layers,
            output_size=3,  # p10, p50, p90
            loss=QuantileLoss([0.1, 0.5, 0.9]),
            log_interval=10,
            reduce_on_plateau_patience=(
                int(self.settings.tft_lr_patience) if self.settings.tft_lr_reduce_on_plateau else 1000
            ),
            reduce_on_plateau_reduction=float(self.settings.tft_lr_reduction),
        )
        trainer.fit(
            self.model, 
            train_dataloaders=train_loader, 
            val_dataloaders=val_loader,
            ckpt_path=resume_from
        )
        best_ckpt = checkpoint_callback.best_model_path
        if best_ckpt:
            try:
                self.model = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)
                log.info("tft_best_checkpoint_loaded", path=best_ckpt)
            except Exception as exc:
                log.warning("tft_best_checkpoint_load_failed", path=best_ckpt, error=str(exc))
        if checkpoint_callback.best_model_score is not None:
            val_loss = float(checkpoint_callback.best_model_score.item())
        else:
            val_loss = float(trainer.callback_metrics.get("val_loss", torch.tensor(0.0)).item())
        calibration_metrics = self._fit_probability_calibrator(df, start_idx=calibration_start_idx)
        log.info("entrenamiento_completado", val_loss=val_loss, **calibration_metrics)
        return ModelMetadata.create(
            model_type="tft_pytorch_improved",
            version="0.2.0",
            metrics={"val_loss": val_loss, "target_mode": train_target, **calibration_metrics},
        )

    def predict(self, df: pd.DataFrame) -> PredictionOutput:
        if self.model is None or not self.dataset_params:
            raise ValueError("Modelo no entrenado/cargado")

        data = self._prepare_data(df, "close")
        if len(data) < self.encoder_length:
            raise ValueError(
                "Datos insuficientes para predecir. "
                f"Req: {self.encoder_length}, Actual: {len(data)}"
            )

        data = self._apply_scalers(data)
        p10_pred, p50_pred, p90_pred = self._extract_quantile_prediction(data)
        target_name = str(self.dataset_params.get("target", "close"))
        if target_name == "log_ret":
            p10_ret = float(np.expm1(p10_pred))
            p50_ret = float(np.expm1(p50_pred))
            p90_ret = float(np.expm1(p90_pred))
            expected_ret = p50_ret
        else:
            current_price = float(df["close"].iloc[-1])

            def calc_ret(price: float) -> float:
                return float((price - current_price) / current_price)

            expected_ret = calc_ret(p50_pred)
            p10_ret = calc_ret(p10_pred)
            p50_ret = expected_ret
            p90_ret = calc_ret(p90_pred)

        # Sanitize: ensure p10 <= p50 <= p90 to prevent Quantile Crossing errors
        p10_ret, p50_ret, p90_ret = sorted([p10_ret, p50_ret, p90_ret])

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
            raise ValueError("Modelo TFT no entrenado.")

        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.dataset_params, path / "dataset_params.joblib")
        joblib.dump(self.scaled_columns, path / "scaled_columns.joblib")
        joblib.dump(self.feature_scalers, path / "feature_scalers.joblib")
        if self.probability_calibrator is not None:
            self.probability_calibrator.save(path / "probability_calibrator.joblib")
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "hparams": self.model.hparams,
            },
            path / "model.pt",
        )
        logger.info("modelo_tft_guardado", path=str(path))

    def load(self, path: Path) -> None:
        params_path = path / "dataset_params.joblib"
        scaled_columns_path = path / "scaled_columns.joblib"
        feature_scalers_path = path / "feature_scalers.joblib"
        calibrator_path = path / "probability_calibrator.joblib"
        model_path = path / "model.pt"
        if not params_path.exists() or not model_path.exists():
            raise FileNotFoundError(f"Artefactos TFT no encontrados en {path}")

        self.dataset_params = joblib.load(params_path)
        if scaled_columns_path.exists() and feature_scalers_path.exists():
            self.scaled_columns = joblib.load(scaled_columns_path)
            self.feature_scalers = joblib.load(feature_scalers_path)
        else:
            self.scaled_columns = []
            self.feature_scalers = {}
        if calibrator_path.exists():
            calibrator = ProbabilityCalibrator()
            calibrator.load(calibrator_path)
            self.probability_calibrator = calibrator
        else:
            self.probability_calibrator = None
        checkpoint = torch.load(model_path, weights_only=False)
        hparams = checkpoint["hparams"]
        state_dict = checkpoint["state_dict"]

        self.model = TemporalFusionTransformer(**hparams)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info("modelo_tft_cargado")
