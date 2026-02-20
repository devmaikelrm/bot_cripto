"""Optional NeuralForecast adapters (iTransformer / PatchTST)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd

from bot_cripto.core.config import get_settings
from bot_cripto.models.base import BasePredictor, ModelMetadata, PredictionOutput


class NeuralForecastAdapter(BasePredictor):
    """BasePredictor wrapper for optional neuralforecast models."""

    def __init__(
        self,
        model_name: Literal["itransformer", "patchtst"],
    ) -> None:
        self.settings = get_settings()
        self.model_name = model_name
        self.horizon = int(self.settings.pred_horizon_steps)
        self.input_size = max(16, int(self.settings.encoder_length))
        self.max_steps = 200
        self.nf: Any = None
        self.residual_std: float = 0.01

    @staticmethod
    def _to_nf_df(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        out = pd.DataFrame()
        if isinstance(df.index, pd.DatetimeIndex):
            out["ds"] = df.index.tz_localize(None) if df.index.tz is not None else df.index
        else:
            out["ds"] = pd.date_range(start="2020-01-01", periods=len(df), freq="5min")
        out["unique_id"] = "0"
        out["y"] = df[target_col].astype(float).values
        return out

    @staticmethod
    def _resolve_model_cls(model_name: str) -> Any:
        from neuralforecast.models import PatchTST

        if model_name == "patchtst":
            return PatchTST

        # iTransformer naming differs across versions.
        try:
            from neuralforecast.models import iTransformer

            return iTransformer
        except Exception:
            from neuralforecast.models import ITransformer  # type: ignore

            return ITransformer

    def train(self, df: pd.DataFrame, target_col: str = "close") -> ModelMetadata:
        from neuralforecast import NeuralForecast

        nf_df = self._to_nf_df(df, target_col=target_col)
        model_cls = self._resolve_model_cls(self.model_name)
        model = model_cls(
            h=self.horizon,
            input_size=self.input_size,
            max_steps=self.max_steps,
        )
        self.nf = NeuralForecast(models=[model], freq="5min")
        self.nf.fit(df=nf_df)

        # Crude residual estimate from one-shot in-sample forecast.
        try:
            pred_df = self.nf.predict(df=nf_df)
            col = next((c for c in pred_df.columns if c.lower() != "ds"), None)
            if col is not None:
                yhat = pred_df[col].to_numpy(dtype=float)
                y = nf_df["y"].tail(len(yhat)).to_numpy(dtype=float)
                if len(y) > 5:
                    self.residual_std = float(np.std(y - yhat))
        except Exception:
            self.residual_std = 0.01

        return ModelMetadata.create(
            model_type=f"neuralforecast_{self.model_name}",
            version="0.1.0",
            metrics={"residual_std": float(self.residual_std)},
        )

    def predict(self, df: pd.DataFrame) -> PredictionOutput:
        if self.nf is None:
            raise ValueError("Model not trained")

        nf_df = self._to_nf_df(df, target_col="close")
        pred_df = self.nf.predict(df=nf_df)
        col = next((c for c in pred_df.columns if c.lower() != "ds"), None)
        if col is None:
            raise ValueError("No forecast column from neuralforecast")

        last_pred = float(pred_df[col].iloc[-1])
        last_close = float(df["close"].iloc[-1])
        expected_ret = (last_pred - last_close) / last_close if last_close != 0 else 0.0

        sigma = max(self.residual_std / max(abs(last_close), 1e-9), 1e-4)
        p50 = expected_ret
        p10 = p50 - 1.28 * sigma
        p90 = p50 + 1.28 * sigma
        prob_up = float(np.clip(0.5 + (expected_ret / (3.0 * sigma)), 0.0, 1.0))
        risk_score = float(np.clip(abs(expected_ret) / max(self.settings.model_risk_spread_ref, 1e-6), 0.0, 1.0))

        return PredictionOutput(
            prob_up=prob_up,
            expected_return=expected_ret,
            p10=p10,
            p50=p50,
            p90=p90,
            risk_score=risk_score,
        )

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_name": self.model_name,
                "horizon": self.horizon,
                "input_size": self.input_size,
                "max_steps": self.max_steps,
                "residual_std": self.residual_std,
                "nf": self.nf,
            },
            path / "neuralforecast_adapter.joblib",
        )

    def load(self, path: Path) -> None:
        payload = joblib.load(path / "neuralforecast_adapter.joblib")
        self.model_name = str(payload["model_name"])
        self.horizon = int(payload["horizon"])
        self.input_size = int(payload["input_size"])
        self.max_steps = int(payload["max_steps"])
        self.residual_std = float(payload.get("residual_std", 0.01))
        self.nf = payload["nf"]
