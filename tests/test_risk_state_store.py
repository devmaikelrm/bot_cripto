from bot_cripto.risk.engine import RiskState
from bot_cripto.risk.state_store import RiskStateStore


def test_risk_state_store_roundtrip(tmp_path) -> None:
    path = tmp_path / "risk_state.json"
    store = RiskStateStore(path)
    state = RiskState(
        equity=9900.0,
        day_start_equity=10_000.0,
        week_start_equity=10_100.0,
        day_id="2026-02-12",
        week_id="2026-07",
    )
    store.save(state)
    loaded = store.load(initial_equity=10_000.0)

    assert loaded.equity == 9900.0
    assert loaded.day_id == "2026-02-12"
