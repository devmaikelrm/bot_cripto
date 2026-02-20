from __future__ import annotations

from bot_cripto.adaptive.retrain_orchestrator import execute_retrain_plan
from bot_cripto.core.config import Settings


def test_execute_retrain_plan_dry_run_includes_meta(tmp_path) -> None:
    settings = Settings(
        data_dir_raw=tmp_path / "raw",
        data_dir_processed=tmp_path / "processed",
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
    )
    settings.ensure_dirs()
    jobs = execute_retrain_plan(
        settings=settings,
        symbol="BTC/USDT",
        timeframe="5m",
        dry_run=True,
        include_meta=True,
    )
    assert [j.job for j in jobs] == ["trend", "return", "risk", "meta"]
    assert all(j.status == "planned" for j in jobs)


def test_execute_retrain_plan_dry_run_without_meta(tmp_path) -> None:
    settings = Settings(
        data_dir_raw=tmp_path / "raw2",
        data_dir_processed=tmp_path / "processed2",
        models_dir=tmp_path / "models2",
        logs_dir=tmp_path / "logs2",
    )
    settings.ensure_dirs()
    jobs = execute_retrain_plan(
        settings=settings,
        symbol="ETH/USDT",
        timeframe="1h",
        dry_run=True,
        include_meta=False,
    )
    assert [j.job for j in jobs] == ["trend", "return", "risk"]
