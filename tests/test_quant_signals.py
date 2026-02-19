from __future__ import annotations

from bot_cripto.core.config import Settings
from bot_cripto.data import quant_signals as qs
from bot_cripto.data.quant_signals import QuantSignalFetcher


def test_social_sentiment_source_x_is_used(monkeypatch, tmp_path) -> None:
    qs._cache.clear()
    settings = Settings(
        data_dir_raw=tmp_path / "raw",
        social_sentiment_source="x",
    )
    settings.ensure_dirs()
    fetcher = QuantSignalFetcher(settings)

    monkeypatch.setattr(fetcher, "_fetch_social_sentiment_x", lambda symbol: 0.8)
    monkeypatch.setattr(fetcher, "_fetch_social_sentiment_endpoint", lambda symbol: None)
    monkeypatch.setattr(fetcher, "_fetch_social_sentiment_telegram", lambda symbol: None)
    monkeypatch.setattr(fetcher, "_fetch_social_sentiment_cryptopanic", lambda symbol: None)
    monkeypatch.setattr(fetcher, "_fetch_social_sentiment_local", lambda symbol: None)

    score = fetcher.fetch_social_sentiment("BTC/USDT")
    assert score == 0.9


def test_social_sentiment_auto_falls_back_to_local(monkeypatch, tmp_path) -> None:
    qs._cache.clear()
    settings = Settings(
        data_dir_raw=tmp_path / "raw",
        social_sentiment_source="auto",
        social_sentiment_endpoint="",
        x_bearer_token="",
        cryptopanic_api_key="",
    )
    settings.ensure_dirs()
    fetcher = QuantSignalFetcher(settings)

    monkeypatch.setattr(fetcher, "_fetch_social_sentiment_nlp", lambda symbol: None)
    monkeypatch.setattr(fetcher, "_fetch_social_sentiment_endpoint", lambda symbol: None)
    monkeypatch.setattr(fetcher, "_fetch_social_sentiment_x", lambda symbol: None)
    monkeypatch.setattr(fetcher, "_fetch_social_sentiment_telegram", lambda symbol: None)
    monkeypatch.setattr(fetcher, "_fetch_social_sentiment_cryptopanic", lambda symbol: None)
    monkeypatch.setattr(fetcher, "_fetch_social_sentiment_local", lambda symbol: 0.25)

    score = fetcher.fetch_social_sentiment("SOL/USDT")
    assert score == 0.625


def test_social_sentiment_falls_back_to_fng(monkeypatch, tmp_path) -> None:
    qs._cache.clear()
    settings = Settings(
        data_dir_raw=tmp_path / "raw",
        social_sentiment_source="api",
        social_sentiment_endpoint="https://example.invalid/sentiment",
    )
    settings.ensure_dirs()
    fetcher = QuantSignalFetcher(settings)

    monkeypatch.setattr(fetcher, "_fetch_social_sentiment_endpoint", lambda symbol: None)
    monkeypatch.setattr(fetcher, "fetch_fear_and_greed", lambda: 0.42)
    score = fetcher.fetch_social_sentiment("ETH/USDT")
    assert score == 0.42


def test_social_sentiment_source_nlp_is_used(monkeypatch, tmp_path) -> None:
    qs._cache.clear()
    settings = Settings(
        data_dir_raw=tmp_path / "raw",
        social_sentiment_source="nlp",
        social_sentiment_nlp_enabled=True,
    )
    settings.ensure_dirs()
    fetcher = QuantSignalFetcher(settings)

    monkeypatch.setattr(fetcher, "_fetch_social_sentiment_nlp", lambda symbol: -0.2)
    score = fetcher.fetch_social_sentiment("BTC/USDT")
    assert score == 0.4


def test_social_sentiment_blend_reweights_missing_sources(monkeypatch, tmp_path) -> None:
    qs._cache.clear()
    settings = Settings(
        data_dir_raw=tmp_path / "raw",
        social_sentiment_source="blend",
        social_sentiment_weight_x=0.5,
        social_sentiment_weight_news=0.3,
        social_sentiment_weight_telegram=0.2,
        social_sentiment_ema_alpha=1.0,
    )
    settings.ensure_dirs()
    fetcher = QuantSignalFetcher(settings)

    monkeypatch.setattr(fetcher, "_fetch_social_sentiment_x", lambda symbol: 0.6)
    monkeypatch.setattr(fetcher, "_fetch_social_sentiment_news", lambda symbol: -0.2)
    monkeypatch.setattr(fetcher, "_fetch_social_sentiment_telegram", lambda symbol: None)

    bundle = fetcher.fetch_social_sentiment_bundle("BTC/USDT")
    # Reweighted: (0.5*0.6 + 0.3*(-0.2)) / 0.8 = 0.3 -> normalized => 0.65
    assert round(bundle["social_sentiment"], 4) == 0.65
    assert round(bundle["social_sentiment_raw"], 4) == 0.65
    assert bundle["social_sentiment_telegram"] == 0.5


def test_social_sentiment_blend_ema_and_velocity(monkeypatch, tmp_path) -> None:
    qs._cache.clear()
    settings = Settings(
        data_dir_raw=tmp_path / "raw",
        social_sentiment_source="blend",
        social_sentiment_ema_alpha=0.5,
    )
    settings.ensure_dirs()
    fetcher = QuantSignalFetcher(settings, cache_ttl=0.0)

    monkeypatch.setattr(fetcher, "_fetch_social_sentiment_x", lambda symbol: 1.0)
    monkeypatch.setattr(fetcher, "_fetch_social_sentiment_news", lambda symbol: 0.0)
    monkeypatch.setattr(fetcher, "_fetch_social_sentiment_telegram", lambda symbol: 0.0)
    first = fetcher.fetch_social_sentiment_bundle("ETH/USDT")
    assert first["social_sentiment_velocity"] == 0.0

    qs._cache.clear()
    monkeypatch.setattr(fetcher, "_fetch_social_sentiment_x", lambda symbol: -1.0)
    monkeypatch.setattr(fetcher, "_fetch_social_sentiment_news", lambda symbol: 0.0)
    monkeypatch.setattr(fetcher, "_fetch_social_sentiment_telegram", lambda symbol: 0.0)
    second = fetcher.fetch_social_sentiment_bundle("ETH/USDT")
    assert second["social_sentiment"] < first["social_sentiment"]
    assert second["social_sentiment_velocity"] < 0.0
