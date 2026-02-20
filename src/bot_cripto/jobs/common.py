from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from bot_cripto.core.config import Settings
from bot_cripto.models.base import ModelMetadata


def now_utc_compact() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")


def safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_")


def load_feature_dataset(settings: Settings, symbol: str, timeframe: str | None = None) -> pd.DataFrame:
    tf = timeframe or settings.timeframe
    path = (
        settings.data_dir_processed / f"{safe_symbol(symbol)}_{tf}_features.parquet"
    )
    if not path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {path}")
    return pd.read_parquet(path)


def load_feature_dataset_for_training(
    settings: Settings,
    symbol: str,
    timeframe: str | None = None,
    prefer_triple_barrier: bool = False,
) -> pd.DataFrame:
    """Load training dataset, optionally preferring triple-barrier labeled file."""
    tf = timeframe or settings.timeframe
    base = settings.data_dir_processed / f"{safe_symbol(symbol)}_{tf}_features.parquet"
    tb = settings.data_dir_processed / f"{safe_symbol(symbol)}_{tf}_features_tb.parquet"
    path = tb if (prefer_triple_barrier and tb.exists()) else base
    if not path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {path}")
    return pd.read_parquet(path)


def build_version_dir(
    settings: Settings,
    model_name: str,
    symbol: str,
    metadata: ModelMetadata,
    timeframe: str | None = None,
) -> Path:
    tf = timeframe or settings.timeframe
    model_root = settings.models_dir / model_name / safe_symbol(symbol) / tf
    version = f"{now_utc_compact()}_{metadata.git_commit}"
    out = model_root / version
    out.mkdir(parents=True, exist_ok=True)
    return out


def latest_model_dir(
    settings: Settings, model_name: str, symbol: str, timeframe: str | None = None
) -> Path:
    tf = timeframe or settings.timeframe
    model_root = settings.models_dir / model_name / safe_symbol(symbol) / tf
    if not model_root.exists():
        raise FileNotFoundError(f"Model root not found: {model_root}")
    candidates = sorted([p for p in model_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not candidates:
        raise FileNotFoundError(f"No model versions found in: {model_root}")
    return candidates[-1]


def model_version_dirs(
    settings: Settings, model_name: str, symbol: str, timeframe: str | None = None
) -> list[Path]:
    """Return sorted version directories for a model/symbol/timeframe."""
    tf = timeframe or settings.timeframe
    model_root = settings.models_dir / model_name / safe_symbol(symbol) / tf
    if not model_root.exists():
        return []
    return sorted([p for p in model_root.iterdir() if p.is_dir()], key=lambda p: p.name)


def write_signal_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_model_metadata(path: Path, metadata: ModelMetadata) -> Path:
    payload = {
        "model_type": metadata.model_type,
        "version": metadata.version,
        "git_commit": metadata.git_commit,
        "trained_at": metadata.trained_at,
        "metrics": metadata.metrics,
    }
    out_path = path / "metadata.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
