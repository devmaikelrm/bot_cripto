from bot_cripto.backtesting.model_benchmark import (
    BenchmarkResult,
    build_benchmark_summary,
    sort_benchmark_results,
)


def test_sort_benchmark_results_orders_ok_models_first() -> None:
    rows = [
        BenchmarkResult("m3", "error", 0.0, 0.0, 0.0, 0.0, "boom"),
        BenchmarkResult("m1", "ok", 0.5, 0.01, 0.7, 60.0),
        BenchmarkResult("m2", "ok", 0.5, 0.02, 0.6, 50.0),
    ]
    sorted_rows = sort_benchmark_results(rows)
    assert [r.model for r in sorted_rows] == ["m2", "m1", "m3"]


def test_sort_benchmark_results_handles_empty() -> None:
    assert sort_benchmark_results([]) == []


def test_build_benchmark_summary_contains_winner_and_tft_deltas() -> None:
    rows = [
        BenchmarkResult("nbeats", "ok", 0.54, 0.030, 1.10, 65.0),
        BenchmarkResult("tft", "ok", 0.52, 0.020, 0.90, 60.0),
    ]
    summary = build_benchmark_summary(rows)
    assert summary["winner"]["model"] == "nbeats"
    assert summary["deltas_vs_tft"] is not None
    assert abs(float(summary["deltas_vs_tft"]["net_return"]) - 0.01) < 1e-12
