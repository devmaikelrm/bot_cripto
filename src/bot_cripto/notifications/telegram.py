"""Telegram notifications with lightweight rate limiting."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from bot_cripto.core.config import Settings, get_settings
from bot_cripto.core.logging import get_logger

logger = get_logger("notifications.telegram")


class TelegramNotifier:
    """Simple Telegram sender with basic throttling."""

    def __init__(self, settings: Settings | None = None, min_interval_seconds: float = 1.0) -> None:
        self.settings = settings or get_settings()
        self.min_interval_seconds = min_interval_seconds
        self._last_sent_ts = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.settings.telegram_bot_token and self.settings.telegram_chat_id)

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_sent_ts
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)

    def _post(self, payload: dict[str, Any]) -> bool:
        if not self.enabled:
            logger.warning("telegram_disabled_missing_token_or_chat")
            return False

        token = self.settings.telegram_bot_token
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")  # noqa: S310
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        req.add_header("User-Agent", "bot-cripto/0.1.0")

        self._throttle()
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                body = resp.read().decode("utf-8")
                parsed = json.loads(body)
                ok = bool(parsed.get("ok"))
                if ok:
                    self._last_sent_ts = time.monotonic()
                    logger.info("telegram_sent")
                else:
                    logger.error("telegram_api_error", body=body)
                return ok
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            logger.error("telegram_send_error", error=str(exc))
            return False

    def send(self, text: str) -> bool:
        payload = {
            "chat_id": self.settings.telegram_chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        return self._post(payload)

    def send_markdown(self, text: str) -> bool:
        payload = {
            "chat_id": self.settings.telegram_chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        return self._post(payload)

    def notify_job_start(self, job_name: str) -> bool:
        return self.send(f"Starting job: `{job_name}`")

    def notify_job_end(self, job_name: str, status: str = "ok") -> bool:
        return self.send(f"Finished job: `{job_name}` status={status}")

    def notify_error(self, where: str, error: str) -> bool:
        return self.send_markdown(f"*Error* in `{where}`\n`{error}`")


def tg_send(text: str, settings: Settings | None = None) -> bool:
    """Send plain text Telegram message."""
    return TelegramNotifier(settings=settings).send(text)


def tg_send_markdown(text: str, settings: Settings | None = None) -> bool:
    """Send markdown-formatted Telegram message."""
    return TelegramNotifier(settings=settings).send_markdown(text)
