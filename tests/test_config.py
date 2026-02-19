"""Tests para configuración centralizada."""

from __future__ import annotations

from bot_cripto.core.config import Settings


class TestSettings:
    """Tests para Settings."""

    def test_defaults(self) -> None:
        """Verifica que los defaults cargan correctamente."""
        settings = Settings()
        assert settings.exchange == "binance"
        assert settings.symbols == "BTC/USDT"
        assert settings.timeframe == "5m"
        assert settings.pred_horizon_steps == 5
        assert settings.encoder_length == 60
        assert settings.paper_mode is True
        assert settings.live_mode is False
        assert settings.stream_snapshot_interval_seconds == 5
        assert settings.stream_orderbook_depth == 20
        assert settings.stream_retention_days == 7
        assert settings.social_sentiment_reliability_enabled is True
        assert settings.social_sentiment_reliability_min_weight == 0.10

    def test_symbols_list(self) -> None:
        """Verifica parsing de symbols comma-separated."""
        settings = Settings(symbols="BTC/USDT, ETH/USDT, SOL/USDT")
        assert settings.symbols_list == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    def test_fees_decimal_conversion(self) -> None:
        """Verifica conversión de bps a decimal."""
        settings = Settings(fees_bps=10)
        assert settings.fees_decimal == 0.001

    def test_fees_zero(self) -> None:
        """Fees cero no rompe nada."""
        settings = Settings(fees_bps=0)
        assert settings.fees_decimal == 0.0

    def test_risk_bounds(self) -> None:
        """Risk max se mantiene en rango."""
        settings = Settings(risk_max=0.5)
        assert 0.0 <= settings.risk_max <= 1.0

    def test_ensure_dirs(self, tmp_path: object) -> None:
        """ensure_dirs crea los directorios necesarios."""
        from pathlib import Path

        base = Path(str(tmp_path))
        settings = Settings(
            data_dir_raw=base / "raw",
            data_dir_processed=base / "processed",
            models_dir=base / "models",
            logs_dir=base / "logs",
        )
        settings.ensure_dirs()
        assert settings.data_dir_raw.exists()
        assert settings.data_dir_processed.exists()
        assert settings.models_dir.exists()
        assert settings.logs_dir.exists()

    def test_macro_event_parsing(self) -> None:
        settings = Settings(
            macro_event_crisis_windows_utc="13:20-14:10,23:30-00:30",
            macro_event_crisis_weekdays="0,2,4",
        )
        assert settings.macro_event_weekdays == [0, 2, 4]
        assert len(settings.macro_event_windows) == 2

    def test_telegram_sentiment_chat_ids_parsing(self) -> None:
        settings = Settings(telegram_sentiment_chat_ids="-1001, -1002,")
        assert settings.telegram_sentiment_chat_ids_list == ["-1001", "-1002"]
