"""Crypto Sentiment Analysis using CryptoPanic API."""

from __future__ import annotations

import requests
import pandas as pd
from filelock import FileLock
from pathlib import Path
from bot_cripto.core.config import Settings
from bot_cripto.core.logging import get_logger

logger = get_logger("data.sentiment")

class SentimentFetcher:
    """Fetch and analyze market sentiment from news aggregators."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.api_token = "YOUR_CRYPTOPANIC_TOKEN" # Se puede configurar via ENV
        self.data_dir = settings.data_dir_raw

    def fetch_sentiment(self, symbol: str = "BTC") -> float:
        """
        Fetch recent news and return a score between -1 (Bearish) and 1 (Bullish).
        If no API token is provided, returns 0.
        """
        if self.api_token == "YOUR_CRYPTOPANIC_TOKEN":
            return 0.0

        url = f"https://cryptopanic.com/api/v1/posts/?auth_token={self.api_token}&currencies={symbol}&filter=hot"
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
            results = data.get("results", [])
            
            if not results:
                return 0.0
                
            scores = []
            for post in results:
                votes = post.get("votes", {})
                # Positive votes: liked, important, bullish
                pos = votes.get("liked", 0) + votes.get("important", 0) + votes.get("bullish", 0)
                # Negative votes: disliked, bearish
                neg = votes.get("disliked", 0) + votes.get("bearish", 0)
                
                total = pos + neg
                if total > 0:
                    scores.append((pos - neg) / total)
            
            if not scores:
                return 0.0
                
            avg_score = sum(scores) / len(scores)
            logger.info("sentiment_captured", symbol=symbol, score=avg_score, news_count=len(results))
            return avg_score

        except Exception as exc:
            logger.error("sentiment_fetch_failed", error=str(exc))
            return 0.0

    def save_sentiment(self, symbol: str, score: float) -> None:
        """Persist sentiment score to be merged later."""
        path = self.data_dir / f"sentiment_{symbol.replace('/', '_')}.parquet"
        df = pd.DataFrame({"sentiment": [score]}, index=[pd.Timestamp.now(tz="UTC")])
        df.index.name = "date"
        
        lock = FileLock(str(path) + ".lock")
        with lock:
            if path.exists():
                existing = pd.read_parquet(path)
                df = pd.concat([existing, df]).sort_index()
                # Keep only last 24h of sentiment history to avoid bloating
                limit = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)
                df = df[df.index >= limit]
            df.to_parquet(path)
