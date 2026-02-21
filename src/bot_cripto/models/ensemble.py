"""Model ensembling utilities."""

from __future__ import annotations

from dataclasses import dataclass

from bot_cripto.models.base import PredictionOutput


@dataclass(frozen=True)
class EnsembleWeights:
    trend: float = 0.34
    ret: float = 0.33
    risk: float = 0.33
    nbeats: float = 0.0


class WeightedEnsemble:
    """Combine multiple prediction outputs into a single contract output.

    When ``nbeats_pred`` is supplied to :meth:`combine`, all four weights
    are re-normalised so they sum to 1.0.  When it is *None* the original
    three-model behaviour is preserved (backward compatible).
    """

    def __init__(self, weights: EnsembleWeights | None = None) -> None:
        self.weights = weights or EnsembleWeights()

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def combine(
        self,
        trend_pred: PredictionOutput,
        return_pred: PredictionOutput,
        risk_pred: PredictionOutput,
        nbeats_pred: PredictionOutput | None = None,
    ) -> PredictionOutput:
        w = self.weights

        # Build (prediction, weight) pairs
        entries: list[tuple[PredictionOutput, float]] = [
            (trend_pred, w.trend),
            (return_pred, w.ret),
            (risk_pred, w.risk),
        ]
        if nbeats_pred is not None:
            entries.append((nbeats_pred, w.nbeats))

        # Normalise weights so they sum to 1.0
        total_w = sum(wt for _, wt in entries)
        if total_w <= 0:
            total_w = 1.0
        norm: list[tuple[PredictionOutput, float]] = [
            (pred, wt / total_w) for pred, wt in entries
        ]

        prob_up = self._clamp01(sum(p.prob_up * wt for p, wt in norm))
        expected_return = sum(p.expected_return * wt for p, wt in norm)

        # Percentiles: weighted average across models.
        # Taking min(P10) / max(P90) inflates the spread and causes risk_score
        # and stop-loss/take-profit levels to be systematically extreme.
        # Weighted average is a better approximation of the blended distribution.
        p10 = sum(p.p10 * wt for p, wt in norm)
        p50 = sum(p.p50 * wt for p, wt in norm)
        p90 = sum(p.p90 * wt for p, wt in norm)

        risk_score = self._clamp01(sum(p.risk_score * wt for p, wt in norm))

        return PredictionOutput(
            prob_up=float(prob_up),
            expected_return=float(expected_return),
            p10=float(p10),
            p50=float(p50),
            p90=float(p90),
            risk_score=float(risk_score),
        )
