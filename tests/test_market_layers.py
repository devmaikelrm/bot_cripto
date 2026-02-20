from __future__ import annotations

import pandas as pd

from bot_cripto.allocation.capital_allocator import CapitalAllocator
from bot_cripto.core.market import market_domain
from bot_cripto.core.config import Settings
from bot_cripto.decision.engine import Action, TradeSignal
from bot_cripto.execution.execution_router import ExecutionRouter


def test_market_domain_classifier() -> None:
    assert market_domain("BTC/USDT") == "crypto"
    assert market_domain("EUR/USD") == "forex"


def test_capital_allocator_outputs_notionals() -> None:
    returns = pd.DataFrame(
        {
            "BTC/USDT": [0.001, -0.002, 0.0015, 0.0002, -0.0004, 0.0011] * 10,
            "EUR/USD": [0.0002, -0.0001, 0.0003, 0.0001, -0.0002, 0.00015] * 10,
        }
    )
    alloc = CapitalAllocator().allocate(returns=returns, total_capital=10000.0)
    assert abs(sum(alloc.weights.values()) - 1.0) < 1e-9
    assert abs(sum(alloc.notionals.values()) - 10000.0) < 1e-6


def test_execution_router_paper_returns_domain_metadata(tmp_path) -> None:
    settings = Settings(
        data_dir_raw=tmp_path / "raw",
        data_dir_processed=tmp_path / "proc",
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
        watchtower_db_path=tmp_path / "logs" / "watchtower.db",
        paper_mode=True,
        live_mode=False,
    )
    settings.ensure_dirs()
    router = ExecutionRouter(settings=settings)
    signal = TradeSignal(action=Action.HOLD, confidence=0.0, weight=0.0, reason="test")
    out = router.execute(symbol="EUR/USD", signal=signal, price=1.08)
    assert out["execution_mode"] == "paper"
    assert out["market_domain"] == "forex"
