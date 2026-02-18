"""Feature Engineering — Indicadores técnicos y pipeline de transformación.

Incluye implementación vectorizada de RSI, MACD, ATR, Bollinger Bands.
Pipeline para limpieza, winsorization y generación de dataset final.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bot_cripto.core.logging import get_logger
from bot_cripto.data.macro import MacroFetcher
from bot_cripto.core.config import get_settings
from bot_cripto.features.microstructure import MicrostructureFeatures

logger = get_logger("features.engineering")


class TechnicalAnalysis:
# ... (rest of class)
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


class MacroMerger:
    """Merges macro-economic data into the high-frequency crypto dataset."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def merge(self, crypto_df: pd.DataFrame) -> pd.DataFrame:
        """Joins macro data (daily) into crypto data (e.g. 5m) via forward fill."""
        fetcher = MacroFetcher(self.settings)
        macro_df = fetcher.load_macro_all()
        
        if macro_df.empty:
            logger.warning("no_macro_data_found_skipping_merge")
            return crypto_df

        # Ensure crypto index is sorted and TZ-aware (UTC)
        crypto_df = crypto_df.sort_index()
        
        # Merge macro data. Since macro is daily, we join and then forward fill.
        # We use merge_asof for efficiency or a simple join + ffill.
        merged = pd.merge_asof(
            crypto_df,
            macro_df,
            left_index=True,
            right_index=True,
            direction="backward"
        )
        
        # Fill any initial NaNs if macro data started after crypto data
        merged = merged.ffill().bfill()
        
        logger.info("macro_data_merged", columns=list(macro_df.columns))
        return merged


class FeaturePipeline:
    """Pipeline de transformación de datos crudos a features."""

    @staticmethod
    def _merge_micro_snapshots(df: pd.DataFrame) -> pd.DataFrame:
        """Load microstructure snapshots and merge into OHLCV via forward-fill."""
        settings = get_settings()
        # Detect symbol from parquet filename convention or use default
        symbol = settings.symbols_list[0].replace("/", "_")
        snap_path = settings.data_dir_raw / f"{symbol}_micro_snapshots.parquet"

        micro_cols = ["obi", "whale_pressure", "sentiment"]
        if not snap_path.exists():
            for col in micro_cols:
                df[col] = 0.0
            return df

        try:
            snaps = pd.read_parquet(snap_path)
            snaps = snaps.sort_index()
            # merge_asof: assign each OHLCV bar the most recent snapshot
            merged = pd.merge_asof(
                df, snaps[micro_cols],
                left_index=True, right_index=True, direction="backward",
            )
            # Fill any leading NaNs (before first snapshot) with 0
            for col in micro_cols:
                merged[col] = merged[col].fillna(0.0)
            return merged
        except Exception as exc:
            logger.warning("micro_snapshot_merge_failed", error=str(exc))
            for col in micro_cols:
                df[col] = 0.0
            return df

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

        # 0. Merge Macro Data
        merger = MacroMerger()
        df = merger.merge(df)

        # 0.5 Microstructure & Sentiment Features (from snapshot file)
        df = self._merge_micro_snapshots(df)
        df["obi_score"] = df["obi"].rolling(window=3).mean().fillna(0.0)
        df["whale_score"] = df["whale_pressure"].rolling(window=5).mean().fillna(0.0)
        df["sentiment_score"] = df["sentiment"].rolling(window=12).mean().fillna(0.0)

        # 1. Indicadores básicos
        # Log Returns & Volatility
        df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
        
        # Volatilidad realizada (Rolling Std Dev) - 20, 50, 100
        df["volatility_20"] = df["log_ret"].rolling(window=20, min_periods=20).std()
        df["volatility_50"] = df["log_ret"].rolling(window=50, min_periods=20).std()
        df["volatility_100"] = df["log_ret"].rolling(window=100, min_periods=20).std()
        # Backward compatibility with legacy feature contracts/tests.
        df["volatility"] = df["volatility_20"]
        
        # Retornos acumulados (Momentum a corto plazo)
        df["ret_1"] = df["log_ret"]  # 1 periodo
        df["ret_3"] = df["log_ret"].rolling(window=3).sum()
        df["ret_5"] = df["log_ret"].rolling(window=5).sum()
        df["ret_10"] = df["log_ret"].rolling(window=10).sum()
        df["ret_20"] = df["log_ret"].rolling(window=20).sum()

        # RSI & Delta
        rsi_series = TechnicalAnalysis.rsi(df["close"], period=14)
        df["rsi"] = rsi_series
        df["rsi_delta"] = rsi_series.diff()

        # MACD
        macd_df = TechnicalAnalysis.macd(df["close"], fast=12, slow=26, signal=9)
        df = pd.concat([df, macd_df], axis=1)
        df["macd_hist_delta"] = df["macd_hist"].diff()

        # Bollinger Bands
        bb_df = TechnicalAnalysis.bollinger_bands(df["close"], period=20, std_dev=2)
        df = pd.concat([df, bb_df], axis=1)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

        # ATR & ATR%
        atr_series = TechnicalAnalysis.atr(df["high"], df["low"], df["close"], period=14)
        df["atr"] = atr_series
        df["atr_pct"] = atr_series / df["close"]

        # EMA Slopes (Tendencia)
        ema_9 = df["close"].ewm(span=9, adjust=False).mean()
        ema_21 = df["close"].ewm(span=21, adjust=False).mean()
        df["ema_slope_9"] = (ema_9 - ema_9.shift(1)) / ema_9.shift(1)
        df["ema_slope_21"] = (ema_21 - ema_21.shift(1)) / ema_21.shift(1)

        # 2. Volume Features
        # Rolling Volume Stats (20 periodos)
        vol_window = 20
        df["vol_mean_20"] = df["volume"].rolling(window=vol_window).mean()
        df["vol_std_20"] = df["volume"].rolling(window=vol_window).std()
        
        # Relative Volume (volumen actual / media movil 20)
        # Avoid division by zero
        df["rel_vol"] = np.where(df["vol_mean_20"] > 0, df["volume"] / df["vol_mean_20"], 0.0)
        
        # Log Volume
        df["log_volume"] = np.log1p(df["volume"])
        
        # Z-Score de Volatilidad (para detectar regímenes extremos)
        # Usamos volatility_100 como base de largo plazo
        vol_base_mean = df["volatility_100"].rolling(window=100, min_periods=20).mean()
        vol_base_std = df["volatility_100"].rolling(window=100, min_periods=20).std()
        df["vol_z"] = ((df["volatility_100"] - vol_base_mean) / vol_base_std).fillna(0.0)

        # 2.5 Microstructure Features (OHLCV-derived)
        df = MicrostructureFeatures.compute_all(df, window=20)

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
