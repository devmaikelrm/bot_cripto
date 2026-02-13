"""Tests de integración para Ingestión de Datos."""

from unittest.mock import patch

import pandas as pd
import pytest

from bot_cripto.core.config import Settings
from bot_cripto.data.ingestion import BinanceFetcher


@pytest.fixture
def temp_settings(tmp_path):
    """Configuración temporal con directorios en tmp."""
    s = Settings(
        data_dir_raw=tmp_path / "raw",
        data_dir_processed=tmp_path / "processed",
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
    )
    s.ensure_dirs()
    return s


@pytest.fixture
def mock_ccxt():
    """Mock de la librería ccxt."""
    with patch("bot_cripto.data.adapters.ccxt") as mock:
        yield mock


class TestBinanceFetcher:
    """Tests para BinanceFetcher."""

    def test_init(self, temp_settings, mock_ccxt):
        """Inicializa correctamente el exchange."""
        fetcher = BinanceFetcher(temp_settings)
        assert fetcher.exchange is not None
        mock_ccxt.binance.assert_called_once()

    def test_fetch_history_mocked(self, temp_settings, mock_ccxt):
        """Simula descarga de datos y verifica paginación."""
        fetcher = BinanceFetcher(temp_settings)

        # Mock fetch_ohlcv to return 2 batches of data then empty
        # Batch 1: 5 candles
        batch1 = [[1600000000000 + i * 300000, 100, 105, 95, 102, 1000] for i in range(5)]
        # Batch 2: 5 candles
        last_ts = batch1[-1][0]
        batch2 = [[last_ts + (i + 1) * 300000, 102, 107, 98, 105, 1200] for i in range(5)]

        # Configure mock return values
        fetcher.exchange.fetch_ohlcv.side_effect = [batch1, batch2, []]
        fetcher.exchange.parse_timeframe.return_value = 300  # 5m = 300s
        fetcher.exchange.rateLimit = 100

        df = fetcher.fetch_history("BTC/USDT", "5m", days=1)

        assert not df.empty
        assert len(df) == 10
        assert "open" in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)
        # Verificar que se llamó 3 veces (2 batches + 1 empty)
        assert fetcher.exchange.fetch_ohlcv.call_count == 3

    def test_save_data(self, temp_settings, mock_ccxt):
        """Verifica guardado en Parquet."""
        fetcher = BinanceFetcher(temp_settings)

        # Crear DF dummy
        data = {
            "timestamp": [1600000000000, 1600000300000],
            "open": [100.0, 101.0],
            "high": [105.0, 106.0],
            "low": [95.0, 96.0],
            "close": [102.0, 103.0],
            "volume": [1000.0, 1100.0],
        }
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("date", inplace=True)

        path = fetcher.save_data(df, "BTC/USDT", "5m")

        assert path.exists()
        assert path.name == "BTC_USDT_5m.parquet"

        # Leer y verificar
        saved_df = pd.read_parquet(path)
        assert len(saved_df) == 2
        assert saved_df.iloc[0]["close"] == 102.0

    def test_fill_gaps_inserts_missing_candle(self, temp_settings, mock_ccxt):
        fetcher = BinanceFetcher(temp_settings)
        fetcher.exchange.parse_timeframe.return_value = 300

        idx = pd.to_datetime(
            ["2024-01-01T00:00:00Z", "2024-01-01T00:05:00Z", "2024-01-01T00:15:00Z"],
            utc=True,
        )
        df = pd.DataFrame(
            {
                "timestamp": [1704067200000, 1704067500000, 1704068100000],
                "open": [100.0, 101.0, 103.0],
                "high": [101.0, 102.0, 104.0],
                "low": [99.0, 100.0, 102.0],
                "close": [100.5, 101.5, 103.5],
                "volume": [10.0, 11.0, 13.0],
            },
            index=idx,
        )

        out = fetcher.fill_gaps(df, "5m")
        assert len(out) == 4
        missing_row = out.loc[pd.Timestamp("2024-01-01T00:10:00Z")]
        assert float(missing_row["volume"]) == 0.0
        assert float(missing_row["close"]) == 101.5

    @pytest.mark.integration
    def test_fetch_live_integration(self, temp_settings):
        """Test de integración real con Binance (1 día).

        Requiere conexión a internet. Corre solo si se marca explícitamente
        o en entorno con acceso.
        """
        try:
            fetcher = BinanceFetcher(temp_settings)
            # Descargar muy poco historial (1 hora = 12 velas de 5m)
            # Para no golpear rate limits en tests
            days = 0.05  # ~1.2 horas

            df = fetcher.fetch_history("BTC/USDT", "5m", days=days)

            if df.empty:
                pytest.skip("No se pudieron descargar datos de Binance (posible problema de red)")

            assert not df.empty
            assert "close" in df.columns
            assert len(df) > 0

            # Guardar
            path = fetcher.save_data(df, "BTC/USDT", "5m")
            assert path.exists()

        except Exception as e:
            pytest.fail(f"Fallo integración con Binance: {e}")
