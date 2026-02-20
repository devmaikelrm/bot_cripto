"""Quantitative signals for sentiment and market context."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from filelock import FileLock

from bot_cripto.core.config import Settings
from bot_cripto.core.logging import get_logger
from bot_cripto.data.sentiment_lexicon import score_text
from bot_cripto.data.sentiment_gnews import GNewsSentimentFetcher
from bot_cripto.data.sentiment_nlp import NLPSentimentScorer
from bot_cripto.data.sentiment_reddit import RedditSentimentFetcher
from bot_cripto.data.sentiment_rss import RSSNewsSentimentFetcher
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
            # Fallback from coingecko/coinpaprika global context.
            macro = self.fetch_global_market_context()
            return float(macro.get("market_sentiment_proxy", 0.5))

    def fetch_global_market_context(self) -> dict[str, float]:
        """Fetch cross-provider global crypto context with resilient fallbacks."""
        cache_key = "global_market_context"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return {
                "btc_dominance": cached,
                "market_sentiment_proxy": max(0.0, min(1.0, 1.0 - cached / 100.0)),
            }

        # Provider A: CoinGecko (preferred)
        try:
            headers = {}
            if self.settings.coingecko_api_key:
                headers["x-cg-demo-api-key"] = self.settings.coingecko_api_key
            response = requests.get(
                "https://api.coingecko.com/api/v3/global",
                headers=headers,
                timeout=_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json().get("data", {})
            btc_dom = float(payload.get("market_cap_percentage", {}).get("btc", 50.0))
            self._cache_set(cache_key, btc_dom)
            return {
                "btc_dominance": btc_dom,
                "market_sentiment_proxy": max(0.0, min(1.0, 1.0 - btc_dom / 100.0)),
            }
        except Exception as exc:
            logger.warning("coingecko_global_context_failed", error=str(exc))

        # Provider B: Coinpaprika (secondary)
        try:
            headers = {}
            if self.settings.coinpaprika_api_key:
                headers["Authorization"] = f"Bearer {self.settings.coinpaprika_api_key}"
            response = requests.get(
                "https://api.coinpaprika.com/v1/global",
                headers=headers,
                timeout=_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
            btc_dom = float(payload.get("bitcoin_dominance_percentage", 50.0))
            self._cache_set(cache_key, btc_dom)
            return {
                "btc_dominance": btc_dom,
                "market_sentiment_proxy": max(0.0, min(1.0, 1.0 - btc_dom / 100.0)),
            }
        except Exception as exc:
            logger.warning("coinpaprika_global_context_failed", error=str(exc))

        return {"btc_dominance": 50.0, "market_sentiment_proxy": 0.5}

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
        """Return final normalized sentiment in [0,1] used by inference."""
        bundle = self.fetch_social_sentiment_bundle(symbol=symbol)
        return float(bundle["social_sentiment"])

    def fetch_social_sentiment_bundle(self, symbol: str = "BTC/USDT") -> dict[str, float]:
        """Return sentiment bundle with raw/ema/velocity and per-source components."""
        cache_key = f"social_bundle:{symbol}"
        cached_sent = self._cache_get(cache_key)
        if cached_sent is not None:
            return {
                "social_sentiment": cached_sent,
                "social_sentiment_raw": cached_sent,
                "social_sentiment_anomaly": 0.0,
                "social_sentiment_zscore": 0.0,
                "social_sentiment_velocity": 0.0,
                "social_sentiment_x": 0.5,
                "social_sentiment_news": 0.5,
                "social_sentiment_telegram": 0.5,
                "social_sentiment_reliability_x": 1.0,
                "social_sentiment_reliability_news": 1.0,
                "social_sentiment_reliability_telegram": 1.0,
            }

        source = (self.settings.social_sentiment_source or "auto").strip().lower()

        # Phase 2 blend: weighted x/news/telegram + automatic reweighting.
        if source in {"auto", "blend"}:
            bundle = self._fetch_social_sentiment_blend_bundle(symbol)
            if bundle is not None:
                self._cache_set(cache_key, float(bundle["social_sentiment"]))
                return bundle

        source_order = (
            [source]
            if source in {"nlp", "api", "x", "telegram", "cryptopanic", "rss", "gnews", "reddit", "local"}
            else ["nlp", "api", "x", "telegram", "gnews", "reddit", "cryptopanic", "rss", "local"]
        )
        for mode in source_order:
            try:
                if mode == "nlp":
                    score = self._fetch_social_sentiment_nlp(symbol)
                elif mode == "api":
                    score = self._fetch_social_sentiment_endpoint(symbol)
                elif mode == "x":
                    score = self._fetch_social_sentiment_x(symbol)
                elif mode == "telegram":
                    score = self._fetch_social_sentiment_telegram(symbol)
                elif mode == "cryptopanic":
                    score = self._fetch_social_sentiment_cryptopanic(symbol)
                elif mode == "rss":
                    score = self._fetch_social_sentiment_rss(symbol)
                elif mode == "gnews":
                    score = self._fetch_social_sentiment_gnews(symbol)
                elif mode == "reddit":
                    score = self._fetch_social_sentiment_reddit(symbol)
                else:
                    score = self._fetch_social_sentiment_local(symbol)
                if score is not None:
                    score01 = _normalize_sentiment_score(score)
                    self._cache_set(cache_key, score01)
                    return {
                        "social_sentiment": score01,
                        "social_sentiment_raw": score01,
                        "social_sentiment_anomaly": 0.0,
                        "social_sentiment_zscore": 0.0,
                        "social_sentiment_velocity": 0.0,
                        "social_sentiment_x": 0.5,
                        "social_sentiment_news": 0.5,
                        "social_sentiment_telegram": 0.5,
                        "social_sentiment_reliability_x": 1.0,
                        "social_sentiment_reliability_news": 1.0,
                        "social_sentiment_reliability_telegram": 1.0,
                    }
            except Exception as exc:
                logger.warning("social_sentiment_source_failed", symbol=symbol, source=mode, error=str(exc))

        # Last fallback: fear/greed.
        try:
            score01 = float(self.fetch_fear_and_greed())
        except Exception:
            score01 = 0.5
        self._cache_set(cache_key, score01)
        return {
            "social_sentiment": score01,
            "social_sentiment_raw": score01,
            "social_sentiment_anomaly": 0.0,
            "social_sentiment_zscore": 0.0,
            "social_sentiment_velocity": 0.0,
            "social_sentiment_x": 0.5,
            "social_sentiment_news": 0.5,
            "social_sentiment_telegram": 0.5,
            "social_sentiment_reliability_x": 1.0,
            "social_sentiment_reliability_news": 1.0,
            "social_sentiment_reliability_telegram": 1.0,
        }

    def _fetch_social_sentiment_blend_bundle(self, symbol: str) -> dict[str, float] | None:
        x_signed = self._fetch_social_sentiment_x(symbol)
        tg_signed = self._fetch_social_sentiment_telegram(symbol)
        news_signed = self._fetch_social_sentiment_news(symbol)

        weighted_raw = self._weighted_sentiment_signed(
            x_signed=x_signed,
            news_signed=news_signed,
            telegram_signed=tg_signed,
            symbol=symbol,
        )
        if weighted_raw is None:
            return None

        raw01 = _normalize_sentiment_score(weighted_raw)
        anomaly, zscore = self._compute_social_sentiment_anomaly(symbol=symbol, raw01=raw01)
        ema01, velocity = self._smooth_social_sentiment(symbol=symbol, raw01=raw01)
        bundle = {
            "social_sentiment": ema01,
            "social_sentiment_raw": raw01,
            "social_sentiment_anomaly": anomaly,
            "social_sentiment_zscore": zscore,
            "social_sentiment_velocity": velocity,
            "social_sentiment_x": _normalize_sentiment_score(x_signed) if x_signed is not None else 0.5,
            "social_sentiment_news": _normalize_sentiment_score(news_signed) if news_signed is not None else 0.5,
            "social_sentiment_telegram": _normalize_sentiment_score(tg_signed) if tg_signed is not None else 0.5,
            "social_sentiment_reliability_x": self._source_reliability("x", symbol, x_signed),
            "social_sentiment_reliability_news": self._source_reliability("news", symbol, news_signed),
            "social_sentiment_reliability_telegram": self._source_reliability("telegram", symbol, tg_signed),
        }
        logger.info(
            "social_sentiment_blend_captured",
            symbol=symbol,
            social_sentiment=bundle["social_sentiment"],
            social_sentiment_raw=bundle["social_sentiment_raw"],
            social_sentiment_anomaly=bundle["social_sentiment_anomaly"],
            social_sentiment_zscore=bundle["social_sentiment_zscore"],
            social_sentiment_velocity=bundle["social_sentiment_velocity"],
            source_x=bundle["social_sentiment_x"],
            source_news=bundle["social_sentiment_news"],
            source_telegram=bundle["social_sentiment_telegram"],
            reliability_x=bundle["social_sentiment_reliability_x"],
            reliability_news=bundle["social_sentiment_reliability_news"],
            reliability_telegram=bundle["social_sentiment_reliability_telegram"],
        )
        return bundle

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

    def _fetch_social_sentiment_nlp(self, symbol: str) -> float | None:
        if not self.settings.social_sentiment_nlp_enabled:
            return None

        texts: list[str] = []
        texts.extend(XSentimentFetcher(self.settings).fetch_recent_texts(symbol=symbol))
        texts.extend(TelegramSentimentFetcher(self.settings).fetch_recent_texts(symbol=symbol))
        if not texts:
            return None

        scorer = NLPSentimentScorer(self.settings)
        score = scorer.score_texts(texts)
        return score

    def _fetch_social_sentiment_rss(self, symbol: str) -> float | None:
        return RSSNewsSentimentFetcher(self.settings).fetch(symbol=symbol)

    def _fetch_social_sentiment_gnews(self, symbol: str) -> float | None:
        return GNewsSentimentFetcher(self.settings).fetch(symbol=symbol)

    def _fetch_social_sentiment_reddit(self, symbol: str) -> float | None:
        return RedditSentimentFetcher(self.settings).fetch(symbol=symbol)

    def _fetch_social_sentiment_news(self, symbol: str) -> float | None:
        """News proxy score in [-1,1] using API -> GNews -> CryptoPanic -> RSS -> local fallbacks."""
        try:
            score = self._fetch_social_sentiment_endpoint(symbol)
            if score is not None:
                return score
        except Exception:
            pass
        try:
            score = self._fetch_social_sentiment_gnews(symbol)
            if score is not None:
                return score
        except Exception:
            pass
        try:
            score = self._fetch_social_sentiment_cryptopanic(symbol)
            if score is not None:
                return score
        except Exception:
            pass
        try:
            score = self._fetch_social_sentiment_rss(symbol)
            if score is not None:
                return score
        except Exception:
            pass
        try:
            return self._fetch_social_sentiment_local(symbol)
        except Exception:
            return None

    def _weighted_sentiment_signed(
        self,
        x_signed: float | None,
        news_signed: float | None,
        telegram_signed: float | None,
        symbol: str = "BTC/USDT",
    ) -> float | None:
        rel_x = self._source_reliability("x", symbol, x_signed)
        rel_news = self._source_reliability("news", symbol, news_signed)
        rel_telegram = self._source_reliability("telegram", symbol, telegram_signed)
        weighted_terms: list[tuple[float, float]] = [
            (
                float(self.settings.social_sentiment_weight_x) * rel_x,
                x_signed if x_signed is not None else np.nan,
            ),
            (
                float(self.settings.social_sentiment_weight_news) * rel_news,
                news_signed if news_signed is not None else np.nan,
            ),
            (
                float(self.settings.social_sentiment_weight_telegram) * rel_telegram,
                telegram_signed if telegram_signed is not None else np.nan,
            ),
        ]
        valid = [(w, v) for w, v in weighted_terms if not np.isnan(v) and w > 0.0]
        if not valid:
            return None
        total_weight = float(sum(w for w, _ in valid))
        if total_weight <= 0.0:
            return None
        combined = float(sum(w * v for w, v in valid) / total_weight)
        return float(max(-1.0, min(1.0, combined)))

    def _source_reliability(self, source: str, symbol: str, value_signed: float | None) -> float:
        if not self.settings.social_sentiment_reliability_enabled:
            return 1.0
        min_w = float(self.settings.social_sentiment_reliability_min_weight)
        if value_signed is None:
            return min_w

        path = self.data_dir / f"signals_{symbol.replace('/', '_')}.parquet"
        col = f"social_sentiment_{source}"
        if not path.exists():
            return 1.0

        try:
            frame = pd.read_parquet(path, columns=[col])
        except Exception:
            return 1.0
        if frame.empty or col not in frame.columns:
            return 1.0

        window = int(self.settings.social_sentiment_reliability_window)
        series = pd.to_numeric(frame[col], errors="coerce").dropna().tail(window)
        if series.empty:
            return 1.0

        # Lower volatility in the source score means higher reliability.
        signed = (series.astype(float) * 2.0) - 1.0
        std = float(signed.std(ddof=0)) if len(signed) > 1 else 0.0
        stability = float(max(0.0, min(1.0, 1.0 - (std / 0.75))))
        sample_factor = float(max(0.2, min(1.0, len(signed) / float(window))))
        reliability = (0.6 * stability) + (0.4 * sample_factor)
        return float(max(min_w, min(1.0, reliability)))

    def _sentiment_state_path(self, symbol: str) -> Path:
        safe_symbol = symbol.replace("/", "_")
        return self.data_dir / f"sentiment_state_{safe_symbol}.json"

    def _load_sentiment_state(self, symbol: str) -> dict[str, float]:
        path = self._sentiment_state_path(symbol)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return {k: float(v) for k, v in payload.items() if isinstance(v, (int, float))}
            return {}
        except Exception:
            return {}

    def _save_sentiment_state(self, symbol: str, ema01: float, raw01: float) -> None:
        path = self._sentiment_state_path(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"ema01": float(ema01), "raw01": float(raw01), "ts": pd.Timestamp.now(tz="UTC").timestamp()}
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _smooth_social_sentiment(self, symbol: str, raw01: float) -> tuple[float, float]:
        state = self._load_sentiment_state(symbol)
        prev_ema = state.get("ema01")
        if prev_ema is None:
            ema = float(raw01)
            velocity = 0.0
        else:
            alpha = float(self.settings.social_sentiment_ema_alpha)
            ema = float(alpha * raw01 + (1.0 - alpha) * float(prev_ema))
            velocity = float(ema - float(prev_ema))
        self._save_sentiment_state(symbol, ema01=ema, raw01=raw01)
        return ema, velocity

    def _compute_social_sentiment_anomaly(self, symbol: str, raw01: float) -> tuple[float, float]:
        """Return anomaly score [0,1] + signed z-score over recent raw sentiment."""
        path = self.data_dir / f"signals_{symbol.replace('/', '_')}.parquet"
        if not path.exists():
            return 0.0, 0.0
        try:
            hist = pd.read_parquet(path, columns=["social_sentiment_raw"])
        except Exception:
            return 0.0, 0.0
        if hist.empty:
            return 0.0, 0.0

        series = pd.to_numeric(hist["social_sentiment_raw"], errors="coerce").dropna()
        window = int(self.settings.social_sentiment_anomaly_window)
        series = series.tail(window)
        if len(series) < 12:
            return 0.0, 0.0

        med = float(series.median())
        mad = float((series - med).abs().median())
        if mad > 1e-8:
            denom = 1.4826 * mad
        else:
            std = float(series.std(ddof=0))
            if std <= 1e-8:
                return 0.0, 0.0
            denom = std

        z = float((raw01 - med) / denom)
        clip = float(self.settings.social_sentiment_anomaly_z_clip)
        anomaly = float(max(0.0, min(1.0, abs(z) / clip)))
        return anomaly, z

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
        social_sentiment_raw: float = 0.5,
        social_sentiment_anomaly: float = 0.0,
        social_sentiment_zscore: float = 0.0,
        social_sentiment_velocity: float = 0.0,
        social_sentiment_x: float = 0.5,
        social_sentiment_news: float = 0.5,
        social_sentiment_telegram: float = 0.5,
        social_sentiment_reliability_x: float = 1.0,
        social_sentiment_reliability_news: float = 1.0,
        social_sentiment_reliability_telegram: float = 1.0,
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
                "social_sentiment_raw": [social_sentiment_raw],
                "social_sentiment_anomaly": [social_sentiment_anomaly],
                "social_sentiment_zscore": [social_sentiment_zscore],
                "social_sentiment_velocity": [social_sentiment_velocity],
                "social_sentiment_x": [social_sentiment_x],
                "social_sentiment_news": [social_sentiment_news],
                "social_sentiment_telegram": [social_sentiment_telegram],
                "social_sentiment_reliability_x": [social_sentiment_reliability_x],
                "social_sentiment_reliability_news": [social_sentiment_reliability_news],
                "social_sentiment_reliability_telegram": [social_sentiment_reliability_telegram],
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
