"""Quantitative signals for sentiment and market context."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import requests
from filelock import FileLock

from bot_cripto.core.config import Settings
from bot_cripto.core.logging import get_logger
from bot_cripto.data.sentiment_lexicon import score_text
from bot_cripto.data.sentiment_telegram import TelegramSentimentFetcher
from bot_cripto.data.sentiment_x import XSentimentFetcher

logger = get_logger("data.quant_signals")

_cache: dict[str, tuple[float, float]] = {}  # key -> (value, expire_ts)
_DEFAULT_TTL = 300.0  # 5 minutes
_REQUEST_TIMEOUT = 5  # fail fast; never block inference for too long


class QuantSignalFetcher:
    """Fetchers for funding/sentiment/orderbook/macro quant signals."""

    def __init__(self, settings: Settings, cache_ttl: float = _DEFAULT_TTL) -> None:
        self.settings = settings
        self.data_dir = settings.data_dir_raw
        self.cache_ttl = cache_ttl

    @staticmethod
    def _cache_get(key: str) -> float | None:
        entry = _cache.get(key)
        if entry is None:
            return None
        value, expire_ts = entry
        if time.monotonic() > expire_ts:
            del _cache[key]
            return None
        return value

    def _cache_set(self, key: str, value: float) -> None:
        _cache[key] = (value, time.monotonic() + self.cache_ttl)

    def fetch_funding_rate(self, symbol: str = "BTC/USDT") -> float:
        cache_key = f"funding:{symbol}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            safe_symbol = symbol.replace("/", "")
            url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={safe_symbol}"
            response = requests.get(url, timeout=_REQUEST_TIMEOUT)
            response.raise_for_status()
            rate = float(response.json().get("lastFundingRate", 0.0))
            logger.info("funding_rate_captured", symbol=symbol, rate=rate)
            self._cache_set(cache_key, rate)
            return rate
        except Exception as exc:
            logger.warning("funding_rate_fetch_failed", symbol=symbol, error=str(exc))
            return 0.0

    def fetch_fear_and_greed(self) -> float:
        cache_key = "fng"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=_REQUEST_TIMEOUT)
            response.raise_for_status()
            value = float(response.json()["data"][0]["value"]) / 100.0
            logger.info("fear_greed_captured", value=value)
            self._cache_set(cache_key, value)
            return value
        except Exception as exc:
            logger.warning("fear_greed_fetch_failed", error=str(exc))
            return 0.5

    def fetch_open_interest(self, symbol: str = "BTC/USDT") -> float:
        cache_key = f"oi:{symbol}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            safe_symbol = symbol.replace("/", "")
            url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={safe_symbol}"
            response = requests.get(url, timeout=_REQUEST_TIMEOUT)
            response.raise_for_status()
            oi = float(response.json().get("openInterest", 0.0))
            logger.info("open_interest_captured", symbol=symbol, oi=oi)
            self._cache_set(cache_key, oi)
            return oi
        except Exception as exc:
            logger.warning("open_interest_fetch_failed", symbol=symbol, error=str(exc))
            return 0.0

    def fetch_long_short_ratio(self, symbol: str = "BTC/USDT") -> float:
        cache_key = f"lsr:{symbol}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            safe_symbol = symbol.replace("/", "")
            url = (
                "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
                f"?symbol={safe_symbol}&period=5m&limit=1"
            )
            response = requests.get(url, timeout=_REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            ratio = float(data[0].get("longShortRatio", 1.0)) if data else 1.0
            logger.info("long_short_ratio_captured", symbol=symbol, ratio=ratio)
            self._cache_set(cache_key, ratio)
            return ratio
        except Exception as exc:
            logger.warning("long_short_ratio_fetch_failed", symbol=symbol, error=str(exc))
            return 1.0

    def fetch_orderbook_imbalance(self, symbol: str = "BTC/USDT", depth: int = 50) -> float:
        """Return imbalance in [-1, 1]: positive => more bids than asks."""
        cache_key = f"obi:{symbol}:{depth}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            safe_symbol = symbol.replace("/", "")
            url = f"https://api.binance.com/api/v3/depth?symbol={safe_symbol}&limit={depth}"
            response = requests.get(url, timeout=_REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            bid_qty = float(sum(float(level[1]) for level in bids if len(level) >= 2))
            ask_qty = float(sum(float(level[1]) for level in asks if len(level) >= 2))
            total = bid_qty + ask_qty
            imbalance = 0.0 if total <= 0 else (bid_qty - ask_qty) / total
            logger.info("orderbook_imbalance_captured", symbol=symbol, depth=depth, imbalance=imbalance)
            self._cache_set(cache_key, imbalance)
            return imbalance
        except Exception as exc:
            logger.warning("orderbook_imbalance_fetch_failed", symbol=symbol, error=str(exc))
            return 0.0

    def fetch_social_sentiment(self, symbol: str = "BTC/USDT") -> float:
        """Read social sentiment using configured source with safe fallbacks.

        Source order in `auto` mode:
        1. `SOCIAL_SENTIMENT_ENDPOINT` (expects JSON with score)
        2. X API (if `X_BEARER_TOKEN` is set)
        3. Telegram Bot API updates (if `TELEGRAM_BOT_TOKEN` is set)
        4. CryptoPanic API (if `CRYPTOPANIC_API_KEY` is set)
        5. local file `data/raw/social_sentiment_<SYMBOL>.json`
        6. neutral 0.5
        """
        cache_key = f"social:{symbol}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        source = (self.settings.social_sentiment_source or "auto").strip().lower()
        source_order = (
            [source]
            if source in {"api", "x", "telegram", "cryptopanic", "local", "auto"}
            else ["auto"]
        )
        if source == "auto":
            source_order = ["api", "x", "telegram", "cryptopanic", "local"]

        for mode in source_order:
            try:
                if mode == "api":
                    score = self._fetch_social_sentiment_endpoint(symbol)
                elif mode == "x":
                    score = self._fetch_social_sentiment_x(symbol)
                elif mode == "telegram":
                    score = self._fetch_social_sentiment_telegram(symbol)
                elif mode == "cryptopanic":
                    score = self._fetch_social_sentiment_cryptopanic(symbol)
                else:
                    score = self._fetch_social_sentiment_local(symbol)
                if score is not None:
                    score = _normalize_sentiment_score(score)
                    logger.info("social_sentiment_captured", symbol=symbol, source=mode, score=score)
                    self._cache_set(cache_key, score)
                    return score
            except Exception as exc:
                logger.warning("social_sentiment_source_failed", symbol=symbol, source=mode, error=str(exc))

        # last fallback: fear/greed if available
        try:
            score = self.fetch_fear_and_greed()
            self._cache_set(cache_key, score)
            return score
        except Exception:
            return 0.5

    def _fetch_social_sentiment_endpoint(self, symbol: str) -> float | None:
        endpoint = (self.settings.social_sentiment_endpoint or "").strip()
        if not endpoint:
            return None
        url = endpoint
        if "{symbol}" in endpoint:
            url = endpoint.replace("{symbol}", symbol.replace("/", "_"))
        response = requests.get(url, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and "score" in payload:
            return float(payload["score"])
        if isinstance(payload, dict):
            values: list[float] = []
            for key in ("twitter", "telegram", "reddit", "news", "sentiment"):
                if key in payload:
                    values.append(float(payload[key]))
            if values:
                return float(np.mean(values))
        return None

    def _fetch_social_sentiment_local(self, symbol: str) -> float | None:
        path = self.data_dir / f"social_sentiment_{symbol.replace('/', '_')}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "score" in payload:
            return float(payload["score"])
        if isinstance(payload, dict):
            values: list[float] = []
            for key in ("twitter", "telegram", "reddit", "news"):
                if key in payload:
                    values.append(float(payload[key]))
            if values:
                return float(np.mean(values))
        return None

    def _fetch_social_sentiment_cryptopanic(self, symbol: str) -> float | None:
        key = (self.settings.cryptopanic_api_key or "").strip()
        if not key:
            return None
        coin = symbol.split("/")[0].upper()
        url = (
            "https://cryptopanic.com/api/developer/v2/posts/"
            f"?auth_token={key}&currencies={coin}&public=true&kind=news"
        )
        response = requests.get(url, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", [])
        if not results:
            return None

        score_sum = 0.0
        counted = 0
        for item in results[:50]:
            title = str(item.get("title", "")).lower()
            if not title:
                continue
            local = score_text(title)
            if local is None:
                continue
            score_sum += local
            counted += 1

        if counted == 0:
            return None
        return score_sum / float(counted)

    def _fetch_social_sentiment_x(self, symbol: str) -> float | None:
        return XSentimentFetcher(self.settings).fetch(symbol=symbol)

    def _fetch_social_sentiment_telegram(self, symbol: str) -> float | None:
        return TelegramSentimentFetcher(self.settings).fetch(symbol=symbol)

    @staticmethod
    def _fetch_yahoo_closes(ticker: str, interval: str = "1d", range_value: str = "6mo") -> pd.Series:
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            f"?interval={interval}&range={range_value}"
        )
        response = requests.get(url, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()["chart"]["result"][0]
        closes = data["indicators"]["quote"][0]["close"]
        timestamps = data["timestamp"]
        idx = pd.to_datetime(timestamps, unit="s", utc=True)
        return pd.Series(closes, index=idx, dtype="float64").dropna()

    def fetch_macro_context(self, btc_close: pd.Series) -> dict[str, float]:
        """Fetch SP500/DXY returns and rolling correlation vs BTC daily returns."""
        cache_key = "macro_context"
        spx_ret = self._cache_get(f"{cache_key}:spx_ret")
        dxy_ret = self._cache_get(f"{cache_key}:dxy_ret")
        corr_spx = self._cache_get(f"{cache_key}:corr_spx")
        corr_dxy = self._cache_get(f"{cache_key}:corr_dxy")
        if spx_ret is not None and dxy_ret is not None and corr_spx is not None and corr_dxy is not None:
            return {
                "sp500_ret_1d": spx_ret,
                "dxy_ret_1d": dxy_ret,
                "corr_btc_sp500": corr_spx,
                "corr_btc_dxy": corr_dxy,
                "macro_risk_off_score": float(max(0.0, min(1.0, (dxy_ret - spx_ret + 0.05) / 0.10))),
            }

        try:
            spx = self._fetch_yahoo_closes("^GSPC")
            dxy = self._fetch_yahoo_closes("DX-Y.NYB")

            btc = btc_close.astype(float).dropna().copy()
            if not isinstance(btc.index, pd.DatetimeIndex):
                btc.index = pd.to_datetime(btc.index, utc=True)
            elif btc.index.tz is None:
                btc.index = btc.index.tz_localize("UTC")
            else:
                btc.index = btc.index.tz_convert("UTC")

            btc_ret = btc.resample("1D").last().pct_change().dropna()
            spx_ret_series = spx.pct_change().dropna()
            dxy_ret_series = dxy.pct_change().dropna()

            align_spx = pd.concat([btc_ret, spx_ret_series], axis=1, join="inner").dropna()
            align_dxy = pd.concat([btc_ret, dxy_ret_series], axis=1, join="inner").dropna()

            corr_btc_sp500 = float(align_spx.iloc[:, 0].corr(align_spx.iloc[:, 1])) if len(align_spx) >= 20 else 0.0
            corr_btc_dxy = float(align_dxy.iloc[:, 0].corr(align_dxy.iloc[:, 1])) if len(align_dxy) >= 20 else 0.0
            sp500_ret_1d = float(spx_ret_series.iloc[-1]) if len(spx_ret_series) > 0 else 0.0
            dxy_ret_1d = float(dxy_ret_series.iloc[-1]) if len(dxy_ret_series) > 0 else 0.0
            macro_risk_off_score = float(max(0.0, min(1.0, (dxy_ret_1d - sp500_ret_1d + 0.05) / 0.10)))

            self._cache_set(f"{cache_key}:spx_ret", sp500_ret_1d)
            self._cache_set(f"{cache_key}:dxy_ret", dxy_ret_1d)
            self._cache_set(f"{cache_key}:corr_spx", corr_btc_sp500)
            self._cache_set(f"{cache_key}:corr_dxy", corr_btc_dxy)
            logger.info(
                "macro_context_captured",
                sp500_ret_1d=sp500_ret_1d,
                dxy_ret_1d=dxy_ret_1d,
                corr_btc_sp500=corr_btc_sp500,
                corr_btc_dxy=corr_btc_dxy,
                macro_risk_off_score=macro_risk_off_score,
            )
            return {
                "sp500_ret_1d": sp500_ret_1d,
                "dxy_ret_1d": dxy_ret_1d,
                "corr_btc_sp500": corr_btc_sp500,
                "corr_btc_dxy": corr_btc_dxy,
                "macro_risk_off_score": macro_risk_off_score,
            }
        except Exception as exc:
            logger.warning("macro_context_fetch_failed", error=str(exc))
            return {
                "sp500_ret_1d": 0.0,
                "dxy_ret_1d": 0.0,
                "corr_btc_sp500": 0.0,
                "corr_btc_dxy": 0.0,
                "macro_risk_off_score": 0.5,
            }

    def save_signals(
        self,
        symbol: str,
        funding: float,
        fng: float,
        open_interest: float = 0.0,
        long_short_ratio: float = 1.0,
        orderbook_imbalance: float = 0.0,
        social_sentiment: float = 0.5,
        sp500_ret_1d: float = 0.0,
        dxy_ret_1d: float = 0.0,
        corr_btc_sp500: float = 0.0,
        corr_btc_dxy: float = 0.0,
        macro_risk_off_score: float = 0.5,
    ) -> None:
        """Save signals to parquet for audit/feature merge."""
        path = self.data_dir / f"signals_{symbol.replace('/', '_')}.parquet"
        df = pd.DataFrame(
            {
                "funding_rate": [funding],
                "fear_greed": [fng],
                "open_interest": [open_interest],
                "long_short_ratio": [long_short_ratio],
                "orderbook_imbalance": [orderbook_imbalance],
                "social_sentiment": [social_sentiment],
                "sp500_ret_1d": [sp500_ret_1d],
                "dxy_ret_1d": [dxy_ret_1d],
                "corr_btc_sp500": [corr_btc_sp500],
                "corr_btc_dxy": [corr_btc_dxy],
                "macro_risk_off_score": [macro_risk_off_score],
            },
            index=[pd.Timestamp.now(tz="UTC")],
        )
        df.index.name = "date"

        lock = FileLock(str(path) + ".lock")
        with lock:
            if path.exists():
                existing = pd.read_parquet(path)
                df = pd.concat([existing, df]).sort_index()
                limit = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
                df = df[df.index >= limit]
            df.to_parquet(path)


def _normalize_sentiment_score(raw: float) -> float:
    if -1.0 <= raw <= 1.0:
        score = (raw + 1.0) / 2.0
    else:
        score = raw
    return float(min(1.0, max(0.0, score)))
