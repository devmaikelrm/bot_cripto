"""Feature layer orchestrator by market domain."""

from __future__ import annotations

import pandas as pd

from bot_cripto.core.market import market_domain
from bot_cripto.features.engineering import FeaturePipeline


class FeatureLayer:
    """Unified feature layer with domain-specific hooks."""

    def __init__(self) -> None:
        self._pipeline = FeaturePipeline()

    def build(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        domain = market_domain(symbol)
        out = self._pipeline.transform(df)
        # Domain marker can be useful for downstream model routing.
        out = out.copy()
        out["market_domain"] = domain
        return out
