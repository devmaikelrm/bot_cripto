"""Tests para Feature Engineering."""

import numpy as np
import pandas as pd
import pytest

from bot_cripto.features.engineering import FeaturePipeline, TechnicalAnalysis


@pytest.fixture
def sample_df():
    """DataFrame de prueba con tendencia alcista."""
    # Generar 100 velas
    dates = pd.date_range(start="2023-01-01", periods=100, freq="5min")
    close = np.linspace(100, 200, 100)  # Tendencia lineal perfecta
    # Agregar un poco de ruido para volatilidad
    np.random.seed(42)
    noise = np.random.normal(0, 1, 100)
    close += noise

    high = close + 2
    low = close - 2
    open_ = close - 0.5
    volume = np.random.randint(100, 1000, 100)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )
    return df


class TestTechnicalAnalysis:
    """Tests para indicadores técnicos."""

    def test_log_returns(self, sample_df):
        """Calcula retornos logarítmicos correctamente."""
        ret = TechnicalAnalysis.log_returns(sample_df["close"])
        assert len(ret) == len(sample_df)
        assert np.isnan(ret.iloc[0])  # Primer valor es NaN
        assert not np.isnan(ret.iloc[1])

    def test_rsi_calculation(self):
        """RSI debe estar entre 0 y 100."""
        # Serie oscilante
        vals = [10, 12, 11, 13, 15, 18, 16, 15, 14, 12, 11, 10, 12, 14, 16, 18, 20]
        s = pd.Series(vals)
        rsi = TechnicalAnalysis.rsi(s, period=5)

        # Eliminar NaNs iniciales
        rsi_vals = rsi.dropna()
        assert not rsi_vals.empty
        assert ((rsi_vals >= 0) & (rsi_vals <= 100)).all()

    def test_macd_structure(self, sample_df):
        """MACD devuelve DataFrame con 3 columnas."""
        macd = TechnicalAnalysis.macd(sample_df["close"])
        assert isinstance(macd, pd.DataFrame)
        assert set(macd.columns) == {"macd", "macd_signal", "macd_hist"}
        assert len(macd) == len(sample_df)

    def test_bollinger_structure(self, sample_df):
        """Bollinger devuelve bandas upper/middle/lower."""
        bb = TechnicalAnalysis.bollinger_bands(sample_df["close"])
        assert {"bb_upper", "bb_middle", "bb_lower"}.issubset(bb.columns)
        # Upper >= Middle >= Lower (salvo casos raros de NaN, pero en general)
        valid = bb.dropna()
        assert (valid["bb_upper"] >= valid["bb_middle"]).all()
        assert (valid["bb_middle"] >= valid["bb_lower"]).all()


class TestFeaturePipeline:
    """Tests para el pipeline de transformación."""

    def test_transform_adds_features(self, sample_df):
        """Transform agrega columnas y limpia NaNs."""
        pipeline = FeaturePipeline()
        initial_rows = len(sample_df)

        processed = pipeline.transform(sample_df)

        # Debe haber eliminado filas iniciales por rolling windows (min 35 aprox)
        assert len(processed) < initial_rows
        assert len(processed) > 0

        # Columnas esperadas
        expected = {
            "log_ret",
            "volatility",
            "rsi",
            "macd",
            "macd_signal",
            "macd_hist",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "atr",
            "rel_vol",
            "hour_sin",
            "hour_cos",
            "day_of_week",
        }
        assert expected.issubset(processed.columns)

        # No debe haber NaNs
        assert not processed.isna().any().any()

    def test_empty_dataframe(self):
        """Maneja DF vacío sin error."""
        pipeline = FeaturePipeline()
        empty = pd.DataFrame()
        res = pipeline.transform(empty)
        assert res.empty
