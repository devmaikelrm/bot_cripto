"""Meta-model to refine TFT predictions based on historical accuracy."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from bot_cripto.core.logging import get_logger

logger = get_logger("models.meta")

class MetaModel:
    """Secondary model that predicts if the primary model's signal will be profitable."""

    def __init__(self) -> None:
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.is_fitted = False

    def _prepare_meta_features(self, tft_pred: dict, regime_str: str, quant_signals: dict) -> np.ndarray:
        """Combine primary model outputs with context for meta-prediction."""
        return np.array([[
            tft_pred.get("prob_up", 0.5),
            tft_pred.get("expected_return", 0.0),
            tft_pred.get("risk_score", 0.5),
            tft_pred.get("confidence", 0.0),
            1.0 if regime_str == "BULL_TREND" else 0.0,
            quant_signals.get("funding_rate", 0.0),
            quant_signals.get("fear_greed", 0.5)
        ]])

    def fit(self, X_meta: pd.DataFrame, y_real: pd.Series) -> None:
        """Train the meta-model on primary model's historical errors."""
        if len(X_meta) < 100:
            logger.warning("meta_model_train_skip_low_data")
            return
            
        self.model.fit(X_meta, y_real)
        self.is_fitted = True
        logger.info("meta_model_fitted")

    def should_filter(self, tft_pred: dict, regime_str: str, quant_signals: dict) -> bool:
        """Returns True if the meta-model suggests blocking the trade."""
        if not self.is_fitted:
            return False # Default to trust primary if meta not ready
            
        X = self._prepare_meta_features(tft_pred, regime_str, quant_signals)
        # Probability of the trade being a 'success' (label 1)
        prob_success = self.model.predict_proba(X)[0][1]
        
        # If meta-model is less than 55% sure of success, filter it
        is_filtered = prob_success < 0.55
        if is_filtered:
            logger.info("meta_model_blocking_trade", prob_success=prob_success)
        return is_filtered

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "meta_model.joblib")

    def load(self, path: Path) -> None:
        self.model = joblib.load(path / "meta_model.joblib")
        self.is_fitted = True
