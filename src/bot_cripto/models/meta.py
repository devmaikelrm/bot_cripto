"""Meta-model to filter low-quality primary signals."""

from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from bot_cripto.core.logging import get_logger

logger = get_logger("models.meta")


class MetaModel:
    """Secondary classifier that predicts if a signal is likely to succeed."""

    FEATURE_COLUMNS: tuple[str, ...] = (
        "prob_up",
        "expected_return",
        "abs_expected_return",
        "risk_score",
        "confidence",
        "regime_bull",
        "regime_bear",
        "regime_range",
        "funding_rate",
        "fear_greed",
        "fear_greed_deviation",
        "orderbook_imbalance",
        "social_sentiment",
        "social_sentiment_anomaly",
        "macro_risk_off_score",
        "corr_btc_sp500",
        "corr_btc_dxy",
        "volatility",
        "rel_vol",
        "adx",
        "funding_x_confidence",
    )

    def __init__(self, min_prob_success: float = 0.55) -> None:
        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42, n_jobs=-1
        )
        self.is_fitted = False
        self.min_prob_success = float(min_prob_success)

    def _prepare_meta_features(self, tft_pred: dict, regime_str: str, quant_signals: dict) -> np.ndarray:
        """Combine primary outputs and context into a fixed feature vector."""
        confidence = float(tft_pred.get("confidence", abs(float(tft_pred.get("prob_up", 0.5)) - 0.5) * 2.0))
        expected_return = float(tft_pred.get("expected_return", 0.0))
        funding_rate = float(quant_signals.get("funding_rate", 0.0))
        fear_greed = float(quant_signals.get("fear_greed", 0.5))
        orderbook = float(quant_signals.get("orderbook_imbalance", 0.0))
        social = float(quant_signals.get("social_sentiment", 0.5))
        social_anomaly = float(quant_signals.get("social_sentiment_anomaly", 0.0))
        macro_risk_off = float(quant_signals.get("macro_risk_off_score", 0.5))
        corr_spx = float(quant_signals.get("corr_btc_sp500", 0.0))
        corr_dxy = float(quant_signals.get("corr_btc_dxy", 0.0))
        volatility = float(quant_signals.get("volatility", 0.0))
        rel_vol = float(quant_signals.get("rel_vol", 1.0))
        adx = float(quant_signals.get("adx", 0.0))
        regime_bull = 1.0 if regime_str == "BULL_TREND" else 0.0
        regime_bear = 1.0 if regime_str == "BEAR_TREND" else 0.0
        regime_range = 1.0 if regime_str == "RANGE_SIDEWAYS" else 0.0
        return np.array([[
            tft_pred.get("prob_up", 0.5),
            expected_return,
            abs(expected_return),
            tft_pred.get("risk_score", 0.5),
            confidence,
            regime_bull,
            regime_bear,
            regime_range,
            funding_rate,
            fear_greed,
            abs(fear_greed - 0.5),
            orderbook,
            social,
            social_anomaly,
            macro_risk_off,
            corr_spx,
            corr_dxy,
            volatility,
            rel_vol,
            adx,
            funding_rate * confidence,
        ]], dtype=float)

    @classmethod
    def ensure_feature_columns(cls, x: pd.DataFrame) -> pd.DataFrame:
        """Guarantee expected feature set, filling missing columns with neutral defaults."""
        out = x.copy()
        defaults: dict[str, float] = {
            "prob_up": 0.5,
            "expected_return": 0.0,
            "abs_expected_return": 0.0,
            "risk_score": 0.5,
            "confidence": 0.0,
            "regime_bull": 0.0,
            "regime_bear": 0.0,
            "regime_range": 1.0,
            "funding_rate": 0.0,
            "fear_greed": 0.5,
            "fear_greed_deviation": 0.0,
            "orderbook_imbalance": 0.0,
            "social_sentiment": 0.5,
            "social_sentiment_anomaly": 0.0,
            "macro_risk_off_score": 0.5,
            "corr_btc_sp500": 0.0,
            "corr_btc_dxy": 0.0,
            "volatility": 0.0,
            "rel_vol": 1.0,
            "adx": 0.0,
            "funding_x_confidence": 0.0,
        }
        for c in cls.FEATURE_COLUMNS:
            if c not in out.columns:
                out[c] = defaults[c]
        return out.loc[:, list(cls.FEATURE_COLUMNS)]

    def fit(self, X_meta: pd.DataFrame, y_real: pd.Series) -> None:
        """Train the meta-model on historical correctness labels."""
        if len(X_meta) < 100:
            logger.warning("meta_model_train_skip_low_data")
            return

        X_fit = self.ensure_feature_columns(X_meta)

        if int(pd.Series(y_real).nunique()) < 2:
            logger.warning("meta_model_train_skip_single_class")
            return

        self.model.fit(X_fit, y_real)
        self.is_fitted = True
        logger.info("meta_model_fitted", samples=len(X_fit))

    def predict_success_prob_batch(self, X_meta: pd.DataFrame) -> np.ndarray:
        """Return success probabilities for a batch of meta features."""
        if not self.is_fitted:
            return np.ones(len(X_meta), dtype=float)
        X = self.ensure_feature_columns(X_meta)
        probs = self.model.predict_proba(X)
        if probs.shape[1] < 2:
            return np.ones(len(X), dtype=float)
        return probs[:, 1].astype(float)

    @staticmethod
    def _threshold_metrics(
        probs: np.ndarray, labels: np.ndarray, threshold: float
    ) -> dict[str, float]:
        preds = (probs >= threshold).astype(int)
        labels_i = labels.astype(int)
        tp = int(((preds == 1) & (labels_i == 1)).sum())
        fp = int(((preds == 1) & (labels_i == 0)).sum())
        fn = int(((preds == 0) & (labels_i == 1)).sum())
        tn = int(((preds == 0) & (labels_i == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        accuracy = (tp + tn) / max(1, (tp + tn + fp + fn))
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        support_pred_pos = float((preds == 1).sum())
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "support_pred_pos": support_pred_pos,
        }

    @classmethod
    def optimize_threshold(
        cls,
        probs: np.ndarray,
        labels: np.ndarray,
        threshold_min: float = 0.50,
        threshold_max: float = 0.80,
        threshold_step: float = 0.01,
        min_positive_predictions: int = 5,
    ) -> dict[str, float]:
        """Pick threshold maximizing F1 with precision/accuracy tie-breakers."""
        if probs.size == 0 or labels.size == 0:
            return {"threshold": 0.55, "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

        best = {
            "threshold": 0.55,
            "precision": 0.0,
            "recall": 0.0,
            "f1": -1.0,
            "accuracy": 0.0,
        }
        t = float(threshold_min)
        while t <= float(threshold_max) + 1e-12:
            m = cls._threshold_metrics(probs=probs, labels=labels, threshold=t)
            if m["support_pred_pos"] < float(min_positive_predictions):
                t += float(threshold_step)
                continue

            better = False
            if m["f1"] > best["f1"]:
                better = True
            elif m["f1"] == best["f1"] and m["precision"] > best["precision"]:
                better = True
            elif (
                m["f1"] == best["f1"]
                and m["precision"] == best["precision"]
                and m["accuracy"] > best["accuracy"]
            ):
                better = True

            if better:
                best = {
                    "threshold": float(t),
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "accuracy": m["accuracy"],
                }
            t += float(threshold_step)

        if best["f1"] < 0:
            # Fallback: no threshold produced enough positive predictions.
            fallback = cls._threshold_metrics(probs=probs, labels=labels, threshold=0.55)
            return {
                "threshold": 0.55,
                "precision": fallback["precision"],
                "recall": fallback["recall"],
                "f1": fallback["f1"],
                "accuracy": fallback["accuracy"],
            }
        return best

    def predict_success_prob(self, tft_pred: dict, regime_str: str, quant_signals: dict) -> float:
        """Return probability that the proposed trade will be successful."""
        if not self.is_fitted:
            return 1.0
        x_arr = self._prepare_meta_features(tft_pred, regime_str, quant_signals)[0]
        X = pd.DataFrame([x_arr], columns=list(self.FEATURE_COLUMNS))
        probs = self.model.predict_proba(X)
        if probs.shape[1] < 2:
            return 1.0
        return float(probs[0][1])

    def should_filter(self, tft_pred: dict, regime_str: str, quant_signals: dict) -> bool:
        """Return True if confidence in success is below threshold."""
        if not self.is_fitted:
            return False

        prob_success = self.predict_success_prob(tft_pred, regime_str, quant_signals)
        is_filtered = prob_success < self.min_prob_success
        if is_filtered:
            logger.info(
                "meta_model_blocking_trade",
                prob_success=prob_success,
                threshold=self.min_prob_success,
            )
        return is_filtered

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "meta_model.joblib")
        (path / "meta_config.json").write_text(
            json.dumps({"min_prob_success": self.min_prob_success}, indent=2),
            encoding="utf-8",
        )

    def load(self, path: Path) -> None:
        self.model = joblib.load(path / "meta_model.joblib")
        cfg = path / "meta_config.json"
        if cfg.exists():
            try:
                payload = json.loads(cfg.read_text(encoding="utf-8"))
                self.min_prob_success = float(payload.get("min_prob_success", self.min_prob_success))
            except Exception:
                pass
        self.is_fitted = True
