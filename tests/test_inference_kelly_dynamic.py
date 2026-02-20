from __future__ import annotations

import json

from bot_cripto.core.config import Settings
from bot_cripto.jobs.inference import _dynamic_kelly_fraction


def _settings(tmp_path) -> Settings:
    settings = Settings(
        data_dir_raw=tmp_path / "raw",
        data_dir_processed=tmp_path / "processed",
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
        risk_kelly_dynamic_enabled=True,
        risk_kelly_fraction=0.2,
        risk_kelly_fraction_min=0.1,
        risk_kelly_fraction_max=0.4,
        risk_kelly_val_loss_good=0.003,
        risk_kelly_val_loss_bad=0.03,
    )
    settings.ensure_dirs()
    return settings


def test_dynamic_kelly_fraction_uses_val_loss_band(tmp_path) -> None:
    settings = _settings(tmp_path)
    model_path = tmp_path / "trend_model"
    model_path.mkdir(parents=True, exist_ok=True)
    (model_path / "metadata.json").write_text(
        json.dumps({"metrics": {"val_loss": 0.003}}),
        encoding="utf-8",
    )
    frac_good = _dynamic_kelly_fraction(settings, model_path)
    assert abs(frac_good - 0.4) < 1e-9

    (model_path / "metadata.json").write_text(
        json.dumps({"metrics": {"val_loss": 0.03}}),
        encoding="utf-8",
    )
    frac_bad = _dynamic_kelly_fraction(settings, model_path)
    assert abs(frac_bad - 0.1) < 1e-9

