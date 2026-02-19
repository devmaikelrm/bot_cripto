"""Simple lexicon-based sentiment scoring helpers."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9_]+")

POSITIVE_WORDS = {
    "surge",
    "bull",
    "bullish",
    "breakout",
    "rally",
    "approval",
    "adoption",
    "buy",
    "up",
    "pump",
    "strong",
    "gain",
    "gains",
    "green",
}

NEGATIVE_WORDS = {
    "crash",
    "bear",
    "bearish",
    "dump",
    "selloff",
    "hack",
    "ban",
    "down",
    "liquidation",
    "weak",
    "loss",
    "losses",
    "red",
}


def score_text(text: str) -> float | None:
    """Return score in [-1, 1] or None when no known sentiment terms exist."""
    if not text:
        return None
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return None
    pos = sum(1 for token in tokens if token in POSITIVE_WORDS)
    neg = sum(1 for token in tokens if token in NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return None
    return float((pos - neg) / total)
