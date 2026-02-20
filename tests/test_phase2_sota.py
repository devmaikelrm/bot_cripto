from __future__ import annotations

import pandas as pd

from bot_cripto.backtesting.phase2_sota import run_phase2_sota
from bot_cripto.core.config import Settings
from bot_cripto.models.base import ModelMetadata, PredictionOutput


class _DummyModel:
    def train(self, df: pd.DataFrame, target_col: str = "close") -> ModelMetadata:  # noqa: ARG002
        return ModelMetadata.create(model_type="dummy", version="0.1.0", metrics={})

    def predict(self, df: pd.DataFrame) -> PredictionOutput:
        c0 = float(df["close"].iloc[-2]) if len(df) >= 2 else float(df["close"].iloc[-1])
        c1 = float(df["close"].iloc[-1])
        exp = (c1 - c0) / c0 if c0 != 0 else 0.0
        prob = 0.6 if exp >= 0 else 0.4
        return PredictionOutput(
            prob_up=prob,
            expected_return=exp,
            p10=exp - 0.01,
            p50=exp,
            p90=exp + 0.01,
            risk_score=0.2,
        )

    def save(self, path) -> None:  # noqa: ANN001
        path.mkdir(parents=True, exist_ok=True)
        (path / "dummy.txt").write_text("ok", encoding="utf-8")


def test_run_phase2_sota_ok_and_skip(monkeypatch, tmp_path) -> None:
    def _resolver(name: str):
        if name == "dummy":
            return (lambda: _DummyModel()), ""
        return None, "unsupported"

    monkeypatch.setattr("bot_cripto.backtesting.phase2_sota.resolve_model_factory", _resolver)

    n = 320
    close = [100.0 + (i * 0.05) for i in range(n)]
    df = pd.DataFrame(
        {
            "close": close,
            "open": close,
            "high": [x + 0.5 for x in close],
            "low": [x - 0.5 for x in close],
            "volume": [1000.0] * n,
        }
    )

    settings = Settings(
        data_dir_raw=tmp_path / "raw",
        data_dir_processed=tmp_path / "processed",
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
    )
    settings.ensure_dirs()

    rows, summary = run_phase2_sota(
        settings=settings,
        symbol="BTC/USDT",
        timeframe="5m",
        df=df,
        model_names=["dummy", "unknown"],
        train_frac=0.7,
    )
    assert len(rows) == 2
    assert rows[0].status == "ok"
    assert rows[1].status == "skipped"
    assert summary["complete_success"] is False
    assert summary["winner"] is not None
