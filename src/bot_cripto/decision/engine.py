"""Decision Engine — Lógica de trading y generación de señales.

(Skill 06) Convierte predicciones (probabilidad, retorno) en acciones (BUY/SELL/HOLD).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from bot_cripto.core.config import get_settings
from bot_cripto.core.logging import get_logger
from bot_cripto.models.base import PredictionOutput

logger = get_logger("decision.engine")


class Action(StrEnum):
    """Acciones posibles del bot."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    NEUTRAL = "NEUTRAL"  # Sin señal clara


class TradeSignal(BaseModel):
    """Señal de trading generada por el engine."""

    action: Action
    confidence: float
    weight: float  # Sugerencia de tamaño (0.0 - 1.0)
    reason: str
    price_limit: float | None = None  # Precio límite sugerido (si aplica)


class DecisionEngine:
    """Motor de decisiones basado en reglas y umbrales adaptativos por régimen."""

    # Regime-specific multipliers for decision thresholds.
    # prob_mult > 1 means more conservative (harder to trigger BUY).
    # return_mult > 1 means requires higher expected return.
    # risk_mult < 1 means lower risk tolerance.
    _REGIME_ADJUSTMENTS: dict[str, dict[str, float]] = {
        "BULL_TREND":      {"prob_mult": 0.90, "return_mult": 0.80, "risk_mult": 1.10},
        "BEAR_TREND":      {"prob_mult": 1.15, "return_mult": 1.30, "risk_mult": 0.80},
        "RANGE_SIDEWAYS":  {"prob_mult": 1.10, "return_mult": 1.20, "risk_mult": 0.90},
        "CRISIS_HIGH_VOL": {"prob_mult": 1.30, "return_mult": 1.50, "risk_mult": 0.60},
        "UNKNOWN":         {"prob_mult": 1.00, "return_mult": 1.00, "risk_mult": 1.00},
    }

    def __init__(self) -> None:
        """Inicializa con configuración global."""
        self.settings = get_settings()

    def decide(
        self,
        prediction: PredictionOutput,
        current_price: float | None = None,
        regime: str = "UNKNOWN",
    ) -> TradeSignal:
        """Evalúa una predicción y emite una señal.

        Thresholds are adapted based on the current market regime.
        """

        adj = self._REGIME_ADJUSTMENTS.get(
            regime, self._REGIME_ADJUSTMENTS["UNKNOWN"]
        )

        # Umbrales adaptativos
        prob_thresh = self.settings.prob_min * adj["prob_mult"]
        min_return = self.settings.min_expected_return * adj["return_mult"]
        max_risk = self.settings.risk_max * adj["risk_mult"]

        # Extracción
        prob_up = prediction.prob_up
        exp_ret = prediction.expected_return
        risk = prediction.risk_score
        fees = self.settings.fees_decimal  # round-trip fees as decimal

        # 1. Filtro de Riesgo
        if risk > max_risk:
            return TradeSignal(
                action=Action.HOLD,
                confidence=0.0,
                weight=0.0,
                reason=f"Riesgo demasiado alto ({risk:.2f} > {max_risk})",
            )

        # 2. Expected Utility: EU = prob_up * upside + prob_down * downside - fees
        upside = prediction.p90   # optimistic scenario return
        downside = prediction.p10  # pessimistic scenario return
        eu = prob_up * upside + (1.0 - prob_up) * downside - fees

        # 3. BUY: positive EU, prob above threshold, return covers costs
        if eu > min_return and prob_up >= prob_thresh and exp_ret >= min_return:
            # Weight scaled by EU magnitude (capped at 1.0)
            weight = min(1.0, eu / (min_return * 3.0)) if min_return > 0 else 1.0
            reason = (
                f"BUY: EU={eu:.4f} > {min_return:.4f}, "
                f"Prob {prob_up:.1%} >= {prob_thresh:.1%}, "
                f"Ret {exp_ret:.2%}"
            )
            return TradeSignal(
                action=Action.BUY,
                confidence=prob_up,
                weight=weight,
                reason=reason,
                price_limit=current_price,
            )

        # 4. SELL: strongly negative EU
        prob_down = 1.0 - prob_up
        if eu < -min_return and (prob_down >= prob_thresh or exp_ret < -min_return):
            reason = (
                f"SELL: EU={eu:.4f} < -{min_return:.4f}, "
                f"Prob Down {prob_down:.1%}"
            )
            return TradeSignal(
                action=Action.SELL,
                confidence=prob_down,
                weight=1.0,
                reason=reason,
                price_limit=current_price,
            )

        # 5. HOLD
        return TradeSignal(
            action=Action.HOLD,
            confidence=0.0,
            weight=0.0,
            reason=f"EU={eu:.4f} insufficient (need > {min_return:.4f})",
        )
