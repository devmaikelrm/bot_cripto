"""Unified smoke checks for external API connectors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any

import requests

from bot_cripto.core.config import Settings
from bot_cripto.data.adapters import build_adapter
from bot_cripto.data.quant_signals import QuantSignalFetcher


@dataclass(frozen=True)
class APICheckResult:
    name: str
    status: str  # ok|skip|error
    latency_ms: float
    detail: str
    payload: dict[str, Any]


def _result(name: str, started: float, status: str, detail: str, payload: dict[str, Any]) -> APICheckResult:
    return APICheckResult(
        name=name,
        status=status,
        latency_ms=(perf_counter() - started) * 1000.0,
        detail=detail,
        payload=payload,
    )


def _check_market_provider(provider: str, symbol: str, timeframe: str) -> APICheckResult:
    started = perf_counter()
    try:
        adapter = build_adapter(provider)
        since = int((datetime.now(tz=UTC) - timedelta(days=2)).timestamp() * 1000)
        rows = adapter.fetch_ohlcv(symbol, timeframe, since=since, limit=5)
        if not rows:
            return _result(provider, started, "error", "no_ohlcv_rows", {})
        return _result(
            provider,
            started,
            "ok",
            "ohlcv_ok",
            {"rows": len(rows), "first_ts": int(rows[0][0]), "last_ts": int(rows[-1][0])},
        )
    except Exception as exc:
        return _result(provider, started, "error", str(exc), {})


def _check_sentiment_source(fetcher: QuantSignalFetcher, source: str, symbol: str, needs_key: bool) -> APICheckResult:
    started = perf_counter()
    if needs_key:
        if source == "x" and not fetcher.settings.x_bearer_token.strip():
            return _result(source, started, "skip", "missing_x_bearer_token", {})
        if source == "telegram" and not fetcher.settings.telegram_bot_token.strip():
            return _result(source, started, "skip", "missing_telegram_bot_token", {})
        if source == "cryptopanic" and not fetcher.settings.cryptopanic_api_key.strip():
            return _result(source, started, "skip", "missing_cryptopanic_api_key", {})
        if source == "gnews" and not fetcher.settings.gnews_api_key.strip():
            return _result(source, started, "skip", "missing_gnews_api_key", {})

    try:
        local = fetcher.settings.model_copy(update={"social_sentiment_source": source})
        score = QuantSignalFetcher(local).fetch_social_sentiment(symbol)
        return _result(source, started, "ok", "sentiment_ok", {"score01": float(score)})
    except Exception as exc:
        return _result(source, started, "error", str(exc), {})


def _check_global_context(fetcher: QuantSignalFetcher) -> APICheckResult:
    started = perf_counter()
    try:
        payload = fetcher.fetch_global_market_context()
        return _result("global_context", started, "ok", "context_ok", payload)
    except Exception as exc:
        return _result("global_context", started, "error", str(exc), {})


def _check_fng(fetcher: QuantSignalFetcher) -> APICheckResult:
    started = perf_counter()
    try:
        value = fetcher.fetch_fear_and_greed()
        return _result("fear_greed", started, "ok", "fng_ok", {"score01": float(value)})
    except Exception as exc:
        return _result("fear_greed", started, "error", str(exc), {})


def _check_http(name: str, url: str, headers: dict[str, str] | None = None) -> APICheckResult:
    started = perf_counter()
    try:
        r = requests.get(url, headers=headers or {}, timeout=6)
        r.raise_for_status()
        return _result(name, started, "ok", "http_ok", {"status_code": int(r.status_code)})
    except Exception as exc:
        return _result(name, started, "error", str(exc), {})


def run_api_smoke(settings: Settings, symbol: str, timeframe: str) -> dict[str, Any]:
    checks: list[APICheckResult] = []
    fetcher = QuantSignalFetcher(settings)

    checks.append(_check_market_provider("binance", symbol, timeframe))
    checks.append(_check_market_provider("bybit", symbol, timeframe))

    checks.append(_check_sentiment_source(fetcher, "x", symbol, needs_key=True))
    checks.append(_check_sentiment_source(fetcher, "telegram", symbol, needs_key=True))
    checks.append(_check_sentiment_source(fetcher, "gnews", symbol, needs_key=True))
    checks.append(_check_sentiment_source(fetcher, "reddit", symbol, needs_key=False))
    checks.append(_check_sentiment_source(fetcher, "cryptopanic", symbol, needs_key=True))
    checks.append(_check_sentiment_source(fetcher, "rss", symbol, needs_key=False))
    checks.append(_check_sentiment_source(fetcher, "nlp", symbol, needs_key=False))

    checks.append(_check_fng(fetcher))
    checks.append(_check_global_context(fetcher))
    checks.append(_check_http("coingecko_ping", "https://api.coingecko.com/api/v3/ping"))
    checks.append(_check_http("coinpaprika_global", "https://api.coinpaprika.com/v1/global"))

    ok = [c for c in checks if c.status == "ok"]
    skipped = [c for c in checks if c.status == "skip"]
    errors = [c for c in checks if c.status == "error"]
    status = "ok" if not errors else ("partial" if ok else "error")

    ts_compact = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_symbol = symbol.replace("/", "_")
    out_file = settings.logs_dir / f"api_smoke_{safe_symbol}_{timeframe}_{ts_compact}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "ts": datetime.now(tz=UTC).isoformat(),
        "symbol": symbol,
        "timeframe": timeframe,
        "status": status,
        "summary": {
            "total": len(checks),
            "ok": len(ok),
            "skip": len(skipped),
            "error": len(errors),
        },
        "checks": [c.__dict__ for c in checks],
        "output_file": str(out_file),
    }
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
