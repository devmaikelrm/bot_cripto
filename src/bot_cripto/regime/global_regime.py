"""Global regime layer combining local ML regime and macro context."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from bot_cripto.data.quant_signals import QuantSignalFetcher
from bot_cripto.regime.ml_engine import MLRegimeEngine


@dataclass(frozen=True)
class GlobalRegime:
    local_regime: str
    macro_risk_off_score: float
    global_regime: str
    reason: str


class GlobalRegimeEngine:
    def __init__(self, ml_engine: MLRegimeEngine, quant_fetcher: QuantSignalFetcher) -> None:
        self.ml_engine = ml_engine
        self.quant_fetcher = quant_fetcher

    def evaluate(self, symbol: str, features_df: pd.DataFrame) -> GlobalRegime:
        local = self.ml_engine.predict(features_df)
        macro = self.quant_fetcher.fetch_macro_context(features_df["close"])
        risk_off = float(macro.get("macro_risk_off_score", 0.5))

        if risk_off >= 0.75:
            return GlobalRegime(
                local_regime=local,
                macro_risk_off_score=risk_off,
                global_regime="GLOBAL_RISK_OFF",
                reason="macro_risk_off_high",
            )
        if local == "CRISIS_HIGH_VOL":
            return GlobalRegime(
                local_regime=local,
                macro_risk_off_score=risk_off,
                global_regime="GLOBAL_CAUTIOUS",
                reason="local_crisis_mode",
            )
        return GlobalRegime(
            local_regime=local,
            macro_risk_off_score=risk_off,
            global_regime="GLOBAL_RISK_ON",
            reason="normal_conditions",
        )
