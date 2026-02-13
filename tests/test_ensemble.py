from bot_cripto.models.base import PredictionOutput
from bot_cripto.models.ensemble import WeightedEnsemble


def test_weighted_ensemble_output_contract() -> None:
    ensemble = WeightedEnsemble()
    p1 = PredictionOutput(0.7, 0.01, -0.01, 0.01, 0.02, 0.2)
    p2 = PredictionOutput(0.6, 0.008, -0.012, 0.009, 0.018, 0.25)
    p3 = PredictionOutput(0.5, 0.007, -0.014, 0.008, 0.016, 0.30)

    merged = ensemble.combine(p1, p2, p3)

    assert 0.0 <= merged.prob_up <= 1.0
    assert 0.0 <= merged.risk_score <= 1.0
    assert merged.p10 <= merged.p50 <= merged.p90
