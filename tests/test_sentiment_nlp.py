from __future__ import annotations

from bot_cripto.core.config import Settings
from bot_cripto.data.sentiment_nlp import NLPSentimentScorer


def test_nlp_scorer_fallback_lexicon(tmp_path) -> None:
    settings = Settings(
        data_dir_raw=tmp_path / "raw",
        social_sentiment_nlp_enabled=False,
    )
    scorer = NLPSentimentScorer(settings)
    score = scorer.score_texts(
        [
            "BTC breakout rally strong up",
            "BTC crash bearish liquidation down",
        ]
    )
    assert score is not None
    assert -1.0 <= score <= 1.0


def test_nlp_scorer_returns_none_when_no_signal(tmp_path) -> None:
    settings = Settings(
        data_dir_raw=tmp_path / "raw",
        social_sentiment_nlp_enabled=False,
    )
    scorer = NLPSentimentScorer(settings)
    score = scorer.score_texts(["hello world", "just noise"])
    assert score is None
