"""Forex model layer wrappers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bot_cripto.models.base import BasePredictor, ModelMetadata, PredictionOutput
from bot_cripto.models.baseline import BaselineModel


class ForexTrendModel(BasePredictor):
    """Forex-side predictor wrapper (conservative baseline objective)."""

    def __init__(self) -> None:
        self.model = BaselineModel(objective="trend")

    def train(self, df: pd.DataFrame, target_col: str = "close") -> ModelMetadata:
        return self.model.train(df, target_col=target_col)

    def predict(self, df: pd.DataFrame) -> PredictionOutput:
        return self.model.predict(df)

    def save(self, path: Path) -> None:
        self.model.save(path)

    def load(self, path: Path) -> None:
        self.model.load(path)
