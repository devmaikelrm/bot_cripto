from bot_cripto.core.config import Settings
from bot_cripto.decision.engine import Action, TradeSignal
from bot_cripto.execution.live import LiveExecutor
from bot_cripto.models.base import PredictionOutput


def _live_settings(tmp_path) -> Settings:
    settings = Settings(
        data_dir_raw=tmp_path / "raw",
        data_dir_processed=tmp_path / "processed",
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
        live_mode=True,
        live_confirm_token="I_UNDERSTAND_LIVE_TRADING",  # noqa: S106
        hard_stop_max_loss=0.03,
    )
    settings.ensure_dirs()
    return settings


def test_live_executor_blocks_without_prediction(tmp_path) -> None:
    settings = _live_settings(tmp_path)
    executor = LiveExecutor(settings=settings)
    signal = TradeSignal(action=Action.BUY, confidence=0.8, weight=1.0, reason="entry")
    out = executor.execute_signal(symbol="BTC/USDT", signal=signal, price=100.0, prediction=None)
    assert out["status"] == "blocked"


def test_live_executor_blocks_on_hard_stop_loss(tmp_path) -> None:
    settings = _live_settings(tmp_path)
    executor = LiveExecutor(settings=settings)
    signal = TradeSignal(action=Action.BUY, confidence=0.8, weight=1.0, reason="entry")
    pred = PredictionOutput(
        prob_up=0.7,
        expected_return=0.01,
        p10=-0.05,
        p50=0.01,
        p90=0.02,
        risk_score=0.2,
    )
    out = executor.execute_signal(symbol="BTC/USDT", signal=signal, price=100.0, prediction=pred)
    assert out["status"] == "blocked"
    assert "hard stop" in out["reason"]


def test_live_executor_returns_hard_stop_price(tmp_path) -> None:
    settings = _live_settings(tmp_path)
    executor = LiveExecutor(settings=settings)
    signal = TradeSignal(action=Action.BUY, confidence=0.8, weight=1.0, reason="entry")
    pred = PredictionOutput(
        prob_up=0.7,
        expected_return=0.01,
        p10=-0.01,
        p50=0.01,
        p90=0.02,
        risk_score=0.2,
    )
    out = executor.execute_signal(symbol="BTC/USDT", signal=signal, price=100.0, prediction=pred)
    assert out["status"] == "ready"
    assert isinstance(out["hard_stop_price"], float)
