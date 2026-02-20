from __future__ import annotations

import pandas as pd

from bot_cripto.core.config import Settings
from bot_cripto.jobs import inference


def _settings(tmp_path) -> Settings:
    settings = Settings(
        data_dir_raw=tmp_path / "raw",
        data_dir_processed=tmp_path / "processed",
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
        watchtower_db_path=tmp_path / "logs" / "watchtower.db",
    )
    settings.ensure_dirs()
    return settings


def test_fetch_quant_signals_forex_skips_crypto_endpoints(monkeypatch, tmp_path) -> None:
    calls: dict[str, int] = {"funding": 0, "oi": 0, "lsr": 0, "obi": 0}

    class DummyFetcher:
        def __init__(self, settings):
            self.settings = settings

        def fetch_funding_rate(self, symbol):
            calls["funding"] += 1
            return 0.1

        def fetch_open_interest(self, symbol):
            calls["oi"] += 1
            return 10.0

        def fetch_long_short_ratio(self, symbol):
            calls["lsr"] += 1
            return 1.2

        def fetch_orderbook_imbalance(self, symbol):
            calls["obi"] += 1
            return 0.2

        def fetch_fear_and_greed(self):
            return 0.5

        def fetch_social_sentiment_bundle(self, symbol):
            return {
                "social_sentiment": 0.5,
                "social_sentiment_raw": 0.5,
                "social_sentiment_anomaly": 0.0,
                "social_sentiment_zscore": 0.0,
                "social_sentiment_velocity": 0.0,
                "social_sentiment_x": 0.5,
                "social_sentiment_news": 0.5,
                "social_sentiment_telegram": 0.5,
                "social_sentiment_reliability_x": 1.0,
                "social_sentiment_reliability_news": 1.0,
                "social_sentiment_reliability_telegram": 1.0,
            }

        def fetch_macro_context(self, close_series):
            return {
                "sp500_ret_1d": 0.0,
                "dxy_ret_1d": 0.0,
                "corr_btc_sp500": 0.0,
                "corr_btc_dxy": 0.0,
                "macro_risk_off_score": 0.5,
            }

        def save_signals(self, *args, **kwargs):
            return None

    monkeypatch.setattr(inference, "QuantSignalFetcher", DummyFetcher)
    settings = _settings(tmp_path)
    df = pd.DataFrame({"close": [1.0, 1.01, 1.0, 1.02]})

    out = inference._fetch_quant_signals_safe(settings=settings, target="EUR/USD", df=df)
    assert out["funding_rate"] == 0.0
    assert out["open_interest"] == 0.0
    assert out["long_short_ratio"] == 1.0
    assert out["orderbook_imbalance"] == 0.0
    assert calls == {"funding": 0, "oi": 0, "lsr": 0, "obi": 0}

