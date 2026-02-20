from __future__ import annotations

from types import SimpleNamespace

from bot_cripto.core.config import Settings
from bot_cripto.data.sentiment_rss import RSSNewsSentimentFetcher


def test_rss_fetch_recent_texts_filters_symbol(monkeypatch, tmp_path) -> None:
    settings = Settings(
        data_dir_raw=tmp_path / "raw",
        social_sentiment_news_rss_enabled=True,
        social_sentiment_news_rss_urls="https://example.com/rss",
        social_sentiment_news_rss_max_items=10,
    )
    settings.ensure_dirs()
    fetcher = RSSNewsSentimentFetcher(settings)

    xml_payload = """
    <rss><channel>
      <item><title>Bitcoin jumps after ETF news</title><description>BTC momentum rises</description></item>
      <item><title>Gold market update</title><description>no crypto mention</description></item>
    </channel></rss>
    """.strip()

    def _fake_get(url: str, timeout: int) -> SimpleNamespace:  # noqa: ARG001
        return SimpleNamespace(content=xml_payload.encode("utf-8"), raise_for_status=lambda: None)

    monkeypatch.setattr("requests.get", _fake_get)
    texts = fetcher.fetch_recent_texts("BTC/USDT")
    assert len(texts) == 1
    assert "Bitcoin" in texts[0]
