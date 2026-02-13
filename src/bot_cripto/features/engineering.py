"""Feature Engineering — Indicadores técnicos y pipeline de transformación.

Incluye implementación vectorizada de RSI, MACD, ATR, Bollinger Bands.
Pipeline para limpieza, winsorization y generación de dataset final.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bot_cripto.core.logging import get_logger

logger = get_logger("features.engineering")


class TechnicalAnalysis:
    """Librería de indicadores técnicos vectorizados."""

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index (RSI)."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Moving Average Convergence Divergence (MACD)."""
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram})

    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
        """Bollinger Bands."""
        ma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        return pd.DataFrame({"bb_upper": upper, "bb_middle": ma, "bb_lower": lower})

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range (ATR)."""
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def log_returns(series: pd.Series) -> pd.Series:
        """Logarithmic Returns."""
        return (series / series.shift(1)).apply(np.log)

    @staticmethod
    def realized_volatility(series: pd.Series, window: int = 20) -> pd.Series:
        """Realized Volatility (Rolling Std Dev of Returns)."""
        return series.rolling(window=window).std()


class FeaturePipeline:
    """Pipeline de transformación de datos crudos a features."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica indicadores y limpieza."""
        log = logger.bind(rows=len(df))
        log.info("iniciando_feature_engineering")

        if df.empty:
            log.warning("dataframe_vacio")
            return df

        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("FeaturePipeline expects a DatetimeIndex.")
        dt_index = df.index

        # 1. Indicadores básicos
        df["log_ret"] = TechnicalAnalysis.log_returns(df["close"])
        df["volatility"] = TechnicalAnalysis.realized_volatility(df["log_ret"])
        df["volatility_100"] = TechnicalAnalysis.realized_volatility(df["log_ret"], window=100)
        df["ret_5"] = df["log_ret"].rolling(window=5).sum()
        df["ret_10"] = df["log_ret"].rolling(window=10).sum()
        df["ret_20"] = df["log_ret"].rolling(window=20).sum()

        df["rsi"] = TechnicalAnalysis.rsi(df["close"])
        df["rsi_delta"] = df["rsi"].diff()

        macd_df = TechnicalAnalysis.macd(df["close"])
        df = pd.concat([df, macd_df], axis=1)
        df["macd_hist_delta"] = df["macd_hist"].diff()

        bb_df = TechnicalAnalysis.bollinger_bands(df["close"])
        df = pd.concat([df, bb_df], axis=1)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

        df["atr"] = TechnicalAnalysis.atr(df["high"], df["low"], df["close"])
        df["atr_pct"] = df["atr"] / df["close"]

        # 2. Volume Features
        # Relative Volume (volumen actual / media movil 20)
        # FX providers may have volume==0. Avoid NaNs/infs wiping the dataset.
        vol_ma = df["volume"].rolling(20).mean()
        df["rel_vol"] = np.where(vol_ma > 0, df["volume"] / vol_ma, 0.0)
        df["log_volume"] = np.log1p(df["volume"])
        vol_mean = df["log_ret"].rolling(100).mean()
        vol_std = df["log_ret"].rolling(100).std().replace(0.0, np.nan)
        df["vol_z"] = ((df["log_ret"] - vol_mean) / vol_std).fillna(0.0)

        # 3. Time Features
        # Seno/Coseno de hora y dia para ciclicidad
        df["hour_sin"] = np.sin(2 * np.pi * dt_index.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * dt_index.hour / 24)
        df["day_of_week"] = dt_index.dayofweek

        # 4. Limpieza (dropna por ventanas rolling)
        # El indicador con mayor ventana es MACD slow (26) + signal (9) ~= 35
        # O Bollinger (20). RSI (14).
        # Eliminamos las primeras N filas que tienen NaN
        original_len = len(df)
        df.dropna(inplace=True)
        dropped = original_len - len(df)

        log.info(
            "features_calculados",
            dropped_rows=dropped,
            final_rows=len(df),
            features=list(df.columns),
        )

        return df
