"""Native Telegram sentiment fetcher using Bot API updates."""

from __future__ import annotations

from statistics import mean

import requests

from bot_cripto.core.config import Settings
from bot_cripto.core.logging import get_logger
from bot_cripto.data.sentiment_lexicon import score_text

logger = get_logger("data.sentiment_telegram")

_REQUEST_TIMEOUT = 5


class TelegramSentimentFetcher:
    """Read bot-visible messages and derive simple sentiment."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def fetch(self, symbol: str = "BTC/USDT") -> float | None:
        token = (self.settings.telegram_bot_token or "").strip()
        if not token:
            return None

        url = f"https://api.telegram.org/bot{token}/getUpdates"
        response = requests.get(
            url,
            params={"limit": self.settings.telegram_sentiment_lookback_limit, "timeout": 0},
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        if not bool(payload.get("ok")):
            return None

        allowed = set(self.settings.telegram_sentiment_chat_ids_list)
        coin = symbol.split("/")[0].upper()
        keys = {coin, f"${coin}", f"#{coin}", symbol.upper()}

        values: list[float] = []
        for item in payload.get("result", []):
            msg = item.get("message") or item.get("channel_post") or {}
            chat_id = str((msg.get("chat") or {}).get("id", ""))
            if allowed and chat_id not in allowed:
                continue

            text = str(msg.get("text") or msg.get("caption") or "")
            if not text:
                continue
            upper = text.upper()
            if not any(k in upper for k in keys):
                continue
            local = score_text(text)
            if local is not None:
                values.append(local)

        if not values:
            return None

        score = float(mean(values))
        logger.info("telegram_sentiment_captured", symbol=symbol, samples=len(values), score=score)
        return score
