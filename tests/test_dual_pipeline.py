from __future__ import annotations

from bot_cripto.core.config import Settings
from bot_cripto.ops import dual_pipeline


def _settings(tmp_path) -> Settings:
    settings = Settings(
        data_dir_raw=tmp_path / "raw",
        data_dir_processed=tmp_path / "processed",
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
    )
    settings.ensure_dirs()
    return settings


def test_dual_pipeline_success(monkeypatch, tmp_path) -> None:
    settings = _settings(tmp_path)

    monkeypatch.setattr(
        dual_pipeline,
        "_fetch_symbol_data",
        lambda **kwargs: {"provider": "mock", "rows": 120, "path": "x.parquet"},
    )
    monkeypatch.setattr(
        dual_pipeline,
        "_build_features",
        lambda **kwargs: {"path": "f.parquet", "rows": 100, "cols": 42},
    )
    monkeypatch.setattr(dual_pipeline, "train_trend_run", lambda **kwargs: "models/trend/mock")
    monkeypatch.setattr(dual_pipeline, "train_return_run", lambda **kwargs: "models/return/mock")
    monkeypatch.setattr(dual_pipeline, "train_risk_run", lambda **kwargs: "models/risk/mock")
    monkeypatch.setattr(dual_pipeline, "train_meta_run", lambda **kwargs: "models/meta/mock")
    monkeypatch.setattr(
        dual_pipeline,
        "inference_run",
        lambda **kwargs: {"decision": "LONG", "confidence": 0.77, "reason": "ok"},
    )

    out = dual_pipeline.run_dual_pipeline(
        settings=settings,
        symbols=["BTC/USDT", "EUR/USD"],
        timeframe="5m",
        days=7,
    )
    assert out["status"] == "ok"
    assert out["summary"]["symbols_total"] == 2
    assert out["summary"]["symbols_ok"] == 2
    assert out["summary"]["symbols_failed"] == 0
    assert len(out["symbols"]) == 2


def test_dual_pipeline_partial_error(monkeypatch, tmp_path) -> None:
    settings = _settings(tmp_path)

    def _fetch(**kwargs):
        if kwargs["symbol"] == "EUR/USD":
            raise RuntimeError("forex unavailable")
        return {"provider": "mock", "rows": 120, "path": "x.parquet"}

    monkeypatch.setattr(dual_pipeline, "_fetch_symbol_data", _fetch)
    monkeypatch.setattr(
        dual_pipeline,
        "_build_features",
        lambda **kwargs: {"path": "f.parquet", "rows": 100, "cols": 42},
    )
    monkeypatch.setattr(dual_pipeline, "train_trend_run", lambda **kwargs: "models/trend/mock")
    monkeypatch.setattr(dual_pipeline, "train_return_run", lambda **kwargs: "models/return/mock")
    monkeypatch.setattr(dual_pipeline, "train_risk_run", lambda **kwargs: "models/risk/mock")
    monkeypatch.setattr(dual_pipeline, "train_meta_run", lambda **kwargs: "models/meta/mock")
    monkeypatch.setattr(
        dual_pipeline,
        "inference_run",
        lambda **kwargs: {"decision": "LONG", "confidence": 0.77, "reason": "ok"},
    )

    out = dual_pipeline.run_dual_pipeline(
        settings=settings,
        symbols=["BTC/USDT", "EUR/USD"],
        timeframe="5m",
        days=7,
    )
    assert out["status"] == "partial_error"
    assert out["summary"]["symbols_ok"] == 1
    assert out["summary"]["symbols_failed"] == 1
    failed = [s for s in out["symbols"] if s["status"] == "failed"]
    assert failed and "forex unavailable" in failed[0]["error"]


def test_dual_pipeline_meta_failure_is_non_blocking(monkeypatch, tmp_path) -> None:
    settings = _settings(tmp_path)

    monkeypatch.setattr(
        dual_pipeline,
        "_fetch_symbol_data",
        lambda **kwargs: {"provider": "mock", "rows": 120, "path": "x.parquet"},
    )
    monkeypatch.setattr(
        dual_pipeline,
        "_build_features",
        lambda **kwargs: {"path": "f.parquet", "rows": 100, "cols": 42},
    )
    monkeypatch.setattr(dual_pipeline, "train_trend_run", lambda **kwargs: "models/trend/mock")
    monkeypatch.setattr(dual_pipeline, "train_return_run", lambda **kwargs: "models/return/mock")
    monkeypatch.setattr(dual_pipeline, "train_risk_run", lambda **kwargs: "models/risk/mock")

    def _meta_fail(**kwargs):
        raise RuntimeError("Insufficient meta samples; need >= 100")

    monkeypatch.setattr(dual_pipeline, "train_meta_run", _meta_fail)
    monkeypatch.setattr(
        dual_pipeline,
        "inference_run",
        lambda **kwargs: {"decision": "NO_TRADE", "confidence": 0.0, "reason": "ok"},
    )

    out = dual_pipeline.run_dual_pipeline(
        settings=settings,
        symbols=["EUR/USD"],
        timeframe="5m",
        days=7,
        include_meta=True,
    )
    assert out["status"] == "ok"
    assert out["summary"]["symbols_ok"] == 1
    step = out["symbols"][0]["steps"]["train_meta"]
    assert step["status"] == "skipped"
    assert "Insufficient meta samples" in step["reason"]
