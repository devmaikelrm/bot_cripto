"""RSS crypto news sentiment fetcher."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from statistics import mean

import requests

from bot_cripto.core.config import Settings
from bot_cripto.core.logging import get_logger
from bot_cripto.data.sentiment_lexicon import score_text

logger = get_logger("data.sentiment_rss")

_REQUEST_TIMEOUT = 6


class RSSNewsSentimentFetcher:
    """Fetch sentiment from configured RSS feeds using lexicon scoring."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _feed_urls(self) -> list[str]:
        raw = self.settings.social_sentiment_news_rss_urls or ""
        return [u.strip() for u in raw.split(",") if u.strip()]

    def fetch(self, symbol: str = "BTC/USDT") -> float | None:
        texts = self.fetch_recent_texts(symbol=symbol)
        if not texts:
            return None
        values = [s for s in (score_text(t) for t in texts) if s is not None]
        if not values:
            return None
        score = float(mean(values))
        logger.info("rss_news_sentiment_captured", symbol=symbol, samples=len(values), score=score)
        return score

    def fetch_recent_texts(self, symbol: str = "BTC/USDT") -> list[str]:
        if not self.settings.social_sentiment_news_rss_enabled:
            return []
        urls = self._feed_urls()
        if not urls:
            return []

        coin = symbol.split("/")[0].upper()
        keys = {coin, f"${coin}", f"#{coin}"}
        limit = int(self.settings.social_sentiment_news_rss_max_items)

        out: list[str] = []
        for url in urls:
            try:
                response = requests.get(url, timeout=_REQUEST_TIMEOUT)
                response.raise_for_status()
                root = ET.fromstring(response.content)
                for item in root.findall(".//item"):
                    title = (item.findtext("title") or "").strip()
                    desc = (item.findtext("description") or "").strip()
                    text = f"{title} {desc}".strip()
                    if not text:
                        continue
                    upper = text.upper()
                    if keys and not any(k in upper for k in keys):
                        continue
                    out.append(text)
                    if len(out) >= limit:
                        return out
            except Exception as exc:
                logger.warning("rss_feed_fetch_failed", url=url, error=str(exc))
                continue
        return out
