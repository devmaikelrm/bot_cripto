"""NLP sentiment scorer with lazy transformer loading and safe fallback."""

from __future__ import annotations

from collections.abc import Iterable

from bot_cripto.core.config import Settings
from bot_cripto.core.logging import get_logger
from bot_cripto.data.sentiment_lexicon import score_text

logger = get_logger("data.sentiment_nlp")


class NLPSentimentScorer:
    """Scores text sentiment in [-1, 1] using transformers when available."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pipeline = None
        self._load_attempted = False

    def _ensure_pipeline(self) -> bool:
        if self._load_attempted:
            return self._pipeline is not None
        self._load_attempted = True
        if not self.settings.social_sentiment_nlp_enabled:
            return False
        try:
            from transformers import pipeline  # type: ignore

            self._pipeline = pipeline(
                "text-classification",
                model=self.settings.social_sentiment_nlp_model_id,
                truncation=True,
            )
            logger.info("nlp_sentiment_pipeline_loaded", model=self.settings.social_sentiment_nlp_model_id)
            return True
        except Exception as exc:
            logger.warning("nlp_sentiment_pipeline_unavailable", error=str(exc))
            return False

    @staticmethod
    def _label_to_score(label: str, confidence: float) -> float:
        lbl = label.upper()
        conf = float(min(1.0, max(0.0, confidence)))
        if "POS" in lbl:
            return conf
        if "NEG" in lbl:
            return -conf
        return 0.0

    def score_texts(self, texts: Iterable[str]) -> float | None:
        cleaned = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
        if not cleaned:
            return None

        max_texts = self.settings.social_sentiment_nlp_max_texts
        sample = cleaned[:max_texts]

        if self._ensure_pipeline():
            assert self._pipeline is not None
            try:
                outputs = self._pipeline(sample)
                scores: list[float] = []
                for item in outputs:
                    label = str(item.get("label", ""))
                    conf = float(item.get("score", 0.0))
                    scores.append(self._label_to_score(label, conf))
                if scores:
                    return float(sum(scores) / len(scores))
            except Exception as exc:
                logger.warning("nlp_sentiment_inference_failed", error=str(exc))

        # Fallback lexicon (still returns [-1, 1])
        scores = [s for s in (score_text(text) for text in sample) if s is not None]
        if not scores:
            return None
        return float(sum(scores) / len(scores))
