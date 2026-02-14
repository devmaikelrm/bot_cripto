"""Market Regime detection using Unsupervised Machine Learning."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from bot_cripto.core.logging import get_logger

logger = get_logger("regime.ml")

class MLRegimeEngine:
    """Detects market regimes using K-Means clustering on volatility and momentum."""

    def __init__(self, n_regimes: int = 4) -> None:
        self.n_regimes = n_regimes
        self.model = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.is_fitted = False
        # Map clusters to human names after fitting
        self.regime_map: dict[int, str] = {}

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features relevant for regime detection."""
        feats = pd.DataFrame(index=df.index)
        feats["vol_std"] = df["log_ret"].rolling(50).std()
        feats["mom_100"] = df["close"].pct_change(100)
        feats["range_pct"] = (df["high"] - df["low"]) / df["close"]
        feats["gap_short_long"] = (df["close"].ewm(span=20).mean() - df["close"].ewm(span=100).mean()) / df["close"]
        return feats.dropna()

    def fit(self, df: pd.DataFrame) -> None:
        """Train the clustering model on historical data."""
        X = self._extract_features(df)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        
        # Analyze clusters to assign names
        clusters = self.model.predict(X_scaled)
        X["cluster"] = clusters
        
        # Heuristic to name regimes:
        # High Volatility -> highest vol_std
        # Bull -> highest mom_100
        # Bear -> lowest mom_100
        stats = X.groupby("cluster").mean()
        
        high_vol_cluster = stats["vol_std"].idxmax()
        bull_cluster = stats["mom_100"].idxmax()
        bear_cluster = stats["mom_100"].idxmin()
        
        for c in range(self.n_regimes):
            if c == high_vol_cluster:
                self.regime_map[c] = "CRISIS_HIGH_VOL"
            elif c == bull_cluster:
                self.regime_map[c] = "BULL_TREND"
            elif c == bear_cluster:
                self.regime_map[c] = "BEAR_TREND"
            else:
                self.regime_map[c] = "RANGE_SIDEWAYS"
        
        self.is_fitted = True
        logger.info("regime_ml_fitted", mapping=self.regime_map)

    def predict(self, df: pd.DataFrame) -> str:
        """Detect current regime for the last row."""
        if not self.is_fitted:
            return "UNKNOWN"
            
        X = self._extract_features(df).tail(1)
        if X.empty:
            return "UNKNOWN"
            
        X_scaled = self.scaler.transform(X)
        cluster = int(self.model.predict(X_scaled)[0])
        return self.regime_map.get(cluster, "RANGE_SIDEWAYS")

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "regime_model.joblib")
        joblib.dump(self.scaler, path / "regime_scaler.joblib")
        joblib.dump(self.regime_map, path / "regime_map.joblib")

    def load(self, path: Path) -> None:
        self.model = joblib.load(path / "regime_model.joblib")
        self.scaler = joblib.load(path / "regime_scaler.joblib")
        self.regime_map = joblib.load(path / "regime_map.joblib")
        self.is_fitted = True
