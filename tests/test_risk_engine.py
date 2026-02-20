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


def test_risk_engine_blocks_on_cvar_breach_and_sets_circuit_breaker() -> None:
    limits = RiskLimits(
        cvar_enabled=True,
        cvar_alpha=0.10,
        cvar_min_samples=20,
        cvar_limit=-0.02,
        circuit_breaker_minutes=60,
    )
    engine = RiskEngine(limits=limits)
    pred = PredictionOutput(0.8, 0.01, -0.01, 0.01, 0.02, 0.1)
    state = RiskState()
    bad_returns = [-0.03] * 30 + [0.005] * 10

    decision = engine.evaluate(
        prediction=pred,
        regime_str="BULL_TREND",
        state=state,
        recent_returns=bad_returns,
    )
    assert decision.allowed is False
    assert "CVaR breach" in decision.reason
    assert state.circuit_breaker_until != ""


def test_risk_engine_blocks_when_circuit_breaker_active() -> None:
    from datetime import UTC, datetime, timedelta

    engine = RiskEngine()
    pred = PredictionOutput(0.8, 0.01, -0.01, 0.01, 0.02, 0.1)
    state = RiskState(
        circuit_breaker_until=(datetime.now(tz=UTC) + timedelta(minutes=30)).isoformat()
    )

    decision = engine.evaluate(
        prediction=pred,
        regime_str="BULL_TREND",
        state=state,
        recent_returns=[0.01] * 100,
    )
    assert decision.allowed is False
    assert "Circuit breaker active" in decision.reason
