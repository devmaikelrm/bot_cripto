from datetime import UTC, datetime

from bot_cripto.monitoring.performance_store import PerformancePoint, PerformanceStore


def test_performance_store_append_and_read(tmp_path) -> None:
    path = tmp_path / "perf.json"
    store = PerformanceStore(path)

    store.append(PerformancePoint(ts=datetime.now(tz=UTC).isoformat(), metric=0.01))
    store.append(PerformancePoint(ts=datetime.now(tz=UTC).isoformat(), metric=-0.02))

    metrics = store.metrics()
    assert metrics == [0.01, -0.02]
