from bot_cripto.decision.engine import Action, TradeSignal
from bot_cripto.execution.paper import PaperExecutor
from bot_cripto.models.base import PredictionOutput


def test_paper_executor_sets_and_triggers_stop_loss(tmp_path) -> None:
    from bot_cripto.core.config import Settings

    settings = Settings(
        data_dir_raw=tmp_path / "raw",
        data_dir_processed=tmp_path / "processed",
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
    )
    settings.ensure_dirs()
    executor = PaperExecutor(settings=settings)

    buy_signal = TradeSignal(action=Action.BUY, confidence=0.8, weight=1.0, reason="entry")
    pred = PredictionOutput(
        prob_up=0.8,
        expected_return=0.01,
        p10=-0.01,
        p50=0.01,
        p90=0.02,
        risk_score=0.2,
    )
    rec_open = executor.on_signal("BTC/USDT", buy_signal, price=100.0, prediction=pred)
    assert rec_open is not None

    hold_signal = TradeSignal(action=Action.HOLD, confidence=0.0, weight=0.0, reason="hold")
    rec_close = executor.on_signal("BTC/USDT", hold_signal, price=98.8)
    assert rec_close is not None
    assert rec_close.action == "SELL_SL"


def test_paper_executor_sets_and_triggers_take_profit(tmp_path) -> None:
    from bot_cripto.core.config import Settings

    settings = Settings(
        data_dir_raw=tmp_path / "raw2",
        data_dir_processed=tmp_path / "processed2",
        models_dir=tmp_path / "models2",
        logs_dir=tmp_path / "logs2",
    )
    settings.ensure_dirs()
    executor = PaperExecutor(settings=settings)

    buy_signal = TradeSignal(action=Action.BUY, confidence=0.8, weight=1.0, reason="entry")
    pred = PredictionOutput(
        prob_up=0.8,
        expected_return=0.01,
        p10=-0.01,
        p50=0.01,
        p90=0.02,
        risk_score=0.2,
    )
    rec_open = executor.on_signal("BTC/USDT", buy_signal, price=100.0, prediction=pred)
    assert rec_open is not None

    hold_signal = TradeSignal(action=Action.HOLD, confidence=0.0, weight=0.0, reason="hold")
    rec_close = executor.on_signal("BTC/USDT", hold_signal, price=102.5)
    assert rec_close is not None
    assert rec_close.action == "SELL_TP"
