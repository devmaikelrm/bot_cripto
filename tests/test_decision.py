"""Tests para Decision Engine."""

import pytest

from bot_cripto.core.config import Settings
from bot_cripto.decision.engine import Action, DecisionEngine
from bot_cripto.models.base import PredictionOutput


@pytest.fixture
def mock_settings(tmp_path):
    """Configuración controlada para tests."""
    return Settings(
        prob_min=0.6,
        min_expected_return=0.002,
        risk_max=0.8,
        data_dir_raw=tmp_path,
        data_dir_processed=tmp_path,
        models_dir=tmp_path,
        logs_dir=tmp_path,
    )


class TestDecisionEngine:
    """Tests de reglas de trading."""

    def test_buy_signal(self, mock_settings):
        """Genera BUY si condiciones se cumplen."""
        engine = DecisionEngine()
        engine.settings = mock_settings

        pred = PredictionOutput(
            prob_up=0.7,  # > 0.6
            expected_return=0.01,  # > 0.002
            risk_score=0.5,  # < 0.8
            p10=0.005,
            p50=0.01,
            p90=0.015,
        )

        signal = engine.decide(pred, current_price=100.0)

        assert signal.action == Action.BUY
        assert signal.confidence == 0.7
        assert "BUY" in signal.reason

    def test_hold_low_prob(self, mock_settings):
        """Genera HOLD si probabilidad es baja."""
        engine = DecisionEngine()
        engine.settings = mock_settings

        pred = PredictionOutput(
            prob_up=0.55, expected_return=0.01, risk_score=0.5, p10=0.0, p50=0.0, p90=0.0  # < 0.6
        )

        signal = engine.decide(pred)
        assert signal.action == Action.HOLD
        assert signal.action != Action.BUY

    def test_hold_low_return(self, mock_settings):
        """Genera HOLD si retorno es insuficiente (fees)."""
        engine = DecisionEngine()
        engine.settings = mock_settings

        pred = PredictionOutput(
            prob_up=0.9, expected_return=0.001, risk_score=0.5, p10=0.0, p50=0.0, p90=0.0  # < 0.002
        )

        signal = engine.decide(pred)
        assert signal.action == Action.HOLD

    def test_hold_high_risk(self, mock_settings):
        """Genera HOLD si riesgo es alto."""
        engine = DecisionEngine()
        engine.settings = mock_settings

        pred = PredictionOutput(
            prob_up=0.9, expected_return=0.05, risk_score=0.9, p10=0.0, p50=0.0, p90=0.0  # > 0.8
        )

        signal = engine.decide(pred)
        assert signal.action == Action.HOLD
        assert "Riesgo demasiado alto" in signal.reason

    def test_sell_signal(self, mock_settings):
        """Genera SELL si probabilidad de bajada es alta."""
        engine = DecisionEngine()
        engine.settings = mock_settings

        # Prob up muy bajo = Prob down alta, negative percentiles for EU
        pred = PredictionOutput(
            prob_up=0.2,  # Prob down 0.8 > 0.6
            expected_return=-0.01,
            risk_score=0.5,
            p10=-0.02,
            p50=-0.01,
            p90=-0.005,
        )

        signal = engine.decide(pred)
        assert signal.action == Action.SELL
        assert signal.confidence == 0.8  # 1 - 0.2
