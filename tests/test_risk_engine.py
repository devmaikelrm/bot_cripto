from bot_cripto.models.base import PredictionOutput
from bot_cripto.risk.engine import RiskEngine, RiskLimits, RiskState


def test_risk_engine_blocks_high_vol_regime() -> None:
    engine = RiskEngine()
    pred = PredictionOutput(0.7, 0.01, -0.01, 0.01, 0.02, 0.2)
    state = RiskState()

    decision = engine.evaluate(prediction=pred, regime_str="CRISIS_HIGH_VOL", state=state)
    assert decision.allowed is False
    assert decision.position_size == 0.0


def test_risk_engine_position_size_positive() -> None:
    engine = RiskEngine(limits=RiskLimits(risk_per_trade=0.02, max_position_size=1.0))
    pred = PredictionOutput(0.8, 0.01, -0.01, 0.01, 0.02, 0.1)
    state = RiskState()

    decision = engine.evaluate(prediction=pred, regime_str="BULL_TREND", state=state)
    assert decision.allowed is True
    assert 0.0 <= decision.position_size <= 1.0
