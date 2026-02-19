"""Native X (Twitter) sentiment fetcher using API v2 recent search."""

from __future__ import annotations

from statistics import mean

import requests

from bot_cripto.core.config import Settings
from bot_cripto.core.logging import get_logger
from bot_cripto.data.sentiment_lexicon import score_text

logger = get_logger("data.sentiment_x")

_REQUEST_TIMEOUT = 5


class XSentimentFetcher:
    """Fetch sentiment from X recent posts."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def fetch(self, symbol: str = "BTC/USDT") -> float | None:
        token = (self.settings.x_bearer_token or "").strip()
        if not token:
            return None

        coin = symbol.split("/")[0].upper()
        query = self.settings.x_query_template.format(symbol=symbol, coin=coin)

        response = requests.get(
            "https://api.twitter.com/2/tweets/search/recent",
            params={
                "query": query,
                "max_results": self.settings.x_max_results,
                "tweet.fields": "lang",
            },
            headers={"Authorization": f"Bearer {token}"},
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        tweets = payload.get("data", [])
        if not tweets:
            return None

        values: list[float] = []
        for item in tweets:
            text = str(item.get("text", ""))
            local = score_text(text)
            if local is not None:
                values.append(local)
        if not values:
            return None

        score = float(mean(values))
        logger.info("x_sentiment_captured", symbol=symbol, samples=len(values), score=score)
        return score
