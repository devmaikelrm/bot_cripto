"""GNews sentiment fetcher for crypto headlines."""

from __future__ import annotations

from statistics import mean

import requests

from bot_cripto.core.config import Settings
from bot_cripto.core.logging import get_logger
from bot_cripto.data.sentiment_lexicon import score_text

logger = get_logger("data.sentiment_gnews")

_REQUEST_TIMEOUT = 6


class GNewsSentimentFetcher:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def fetch(self, symbol: str = "BTC/USDT") -> float | None:
        texts = self.fetch_recent_texts(symbol=symbol)
        if not texts:
            return None
        values = [s for s in (score_text(t) for t in texts) if s is not None]
        if not values:
            return None
        score = float(mean(values))
        logger.info("gnews_sentiment_captured", symbol=symbol, samples=len(values), score=score)
        return score

    def fetch_recent_texts(self, symbol: str = "BTC/USDT") -> list[str]:
        token = (self.settings.gnews_api_key or "").strip()
        if not token:
            return []
        coin = symbol.split("/")[0].upper()
        q = f"{coin} OR ${coin} OR #{coin}"
        response = requests.get(
            "https://gnews.io/api/v4/search",
            params={
                "q": q,
                "lang": "en",
                "max": int(self.settings.gnews_max_results),
                "token": token,
            },
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        articles = payload.get("articles", [])
        out: list[str] = []
        for a in articles:
            title = str(a.get("title", "")).strip()
            desc = str(a.get("description", "")).strip()
            text = f"{title} {desc}".strip()
            if text:
                out.append(text)
        return out
