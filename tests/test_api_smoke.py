from __future__ import annotations

from bot_cripto.core.config import Settings
from bot_cripto.ops import api_smoke as smoke


def test_api_smoke_writes_report(monkeypatch, tmp_path) -> None:
    settings = Settings(
        data_dir_raw=tmp_path / "raw",
        data_dir_processed=tmp_path / "processed",
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
    )
    settings.ensure_dirs()

    monkeypatch.setattr(
        smoke,
        "_check_market_provider",
        lambda provider, symbol, timeframe: smoke.APICheckResult(
            name=provider, status="ok", latency_ms=1.0, detail="ok", payload={}
        ),
    )
    monkeypatch.setattr(
        smoke,
        "_check_sentiment_source",
        lambda fetcher, source, symbol, needs_key: smoke.APICheckResult(
            name=source, status="ok", latency_ms=1.0, detail="ok", payload={}
        ),
    )
    monkeypatch.setattr(
        smoke,
        "_check_fng",
        lambda fetcher: smoke.APICheckResult(
            name="fear_greed", status="ok", latency_ms=1.0, detail="ok", payload={}
        ),
    )
    monkeypatch.setattr(
        smoke,
        "_check_global_context",
        lambda fetcher: smoke.APICheckResult(
            name="global_context", status="ok", latency_ms=1.0, detail="ok", payload={}
        ),
    )
    monkeypatch.setattr(
        smoke,
        "_check_http",
        lambda name, url, headers=None: smoke.APICheckResult(
            name=name, status="ok", latency_ms=1.0, detail="ok", payload={}
        ),
    )

    out = smoke.run_api_smoke(settings=settings, symbol="BTC/USDT", timeframe="5m")
    assert out["status"] == "ok"
    assert out["summary"]["error"] == 0
    assert out["summary"]["ok"] >= 10
