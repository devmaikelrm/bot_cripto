"""Reddit sentiment fetcher using public search endpoint."""

from __future__ import annotations

from statistics import mean

import requests

from bot_cripto.core.config import Settings
from bot_cripto.core.logging import get_logger
from bot_cripto.data.sentiment_lexicon import score_text

logger = get_logger("data.sentiment_reddit")

_REQUEST_TIMEOUT = 6


class RedditSentimentFetcher:
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
        logger.info("reddit_sentiment_captured", symbol=symbol, samples=len(values), score=score)
        return score

    def fetch_recent_texts(self, symbol: str = "BTC/USDT") -> list[str]:
        coin = symbol.split("/")[0].upper()
        q = f"{coin} OR ${coin} OR #{coin}"
        response = requests.get(
            "https://www.reddit.com/search.json",
            params={"q": q, "sort": "new", "limit": int(self.settings.reddit_max_results)},
            headers={"User-Agent": self.settings.reddit_user_agent or "bot-cripto/0.1"},
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        children = payload.get("data", {}).get("children", [])
        out: list[str] = []
        for item in children:
            data = item.get("data", {})
            title = str(data.get("title", "")).strip()
            text = str(data.get("selftext", "")).strip()
            joined = f"{title} {text}".strip()
            if joined:
                out.append(joined)
        return out
