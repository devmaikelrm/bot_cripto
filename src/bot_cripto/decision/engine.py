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
    """Motor de decisiones basado en reglas y umbrales."""

    def __init__(self) -> None:
        """Inicializa con configuración global."""
        self.settings = get_settings()

    def decide(
        self, prediction: PredictionOutput, current_price: float | None = None
    ) -> TradeSignal:
        """Evalúa una predicción y emite una señal."""

        # Umbrales
        prob_thresh = self.settings.prob_min
        min_return = self.settings.min_expected_return
        max_risk = self.settings.risk_max

        # Extracción
        prob_up = prediction.prob_up
        exp_ret = prediction.expected_return
        risk = prediction.risk_score

        # Lógica de decisión fundamental (Long Only para Spot)

        # 1. Filtro de Riesgo
        if risk > max_risk:
            return TradeSignal(
                action=Action.HOLD,
                confidence=0.0,
                weight=0.0,
                reason=f"Riesgo demasiado alto ({risk:.2f} > {max_risk})",
            )

        # 2. Señal de COMPRA (BUY)
        # Condiciones:
        # - Probabilidad de subida > umbral
        # - Retorno esperado > mínimo (cubrir fees + profit)
        # - (Opcional) p10 > stop loss implícito? (p10 es retorno pesimista)

        if prob_up >= prob_thresh and exp_ret >= min_return:
            # Sizing básico: escalar por confianza o riesgo
            # Kelly simple o fijo. Usemos una heurística simple:
            # Mayor prob -> mayor weight. Mayor riesgo -> menor weight.
            # Weight base 1.0, ajustado.
            weight = 1.0

            reason = (
                f"BUY SIGNAL: Prob {prob_up:.1%} >= {prob_thresh:.1%}, "
                f"Ret {exp_ret:.2%} >= {min_return:.1%}"
            )

            return TradeSignal(
                action=Action.BUY,
                confidence=prob_up,
                weight=weight,
                reason=reason,
                price_limit=current_price,
            )

        # 3. Señal de VENTA (SELL)
        # En spot, SELL significa salir de posición.
        # Condiciones para vender:
        # - Probabilidad de subida muy baja (o sea, prob bajada alta)
        # - Retorno esperado negativo fuerte

        # Umbral de venta: Si prob_up < (1 - prob_thresh) -> prob_down > prob_thresh
        prob_down = 1.0 - prob_up
        if prob_down >= prob_thresh or exp_ret < -min_return:
            reason = (
                f"SELL SIGNAL: Prob Down {prob_down:.1%} >= {prob_thresh:.1%} "
                f"OR Ret {exp_ret:.2%} < -{min_return:.1%}"
            )
            return TradeSignal(
                action=Action.SELL,
                confidence=prob_down,
                weight=1.0,
                reason=reason,
                price_limit=current_price,
            )

        # 4. NEUTRAL / HOLD
        return TradeSignal(
            action=Action.HOLD,
            confidence=0.0,
            weight=0.0,
            reason="Market noise (no signal)",
        )
