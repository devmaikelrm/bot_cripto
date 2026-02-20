from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from bot_cripto.core.config import Settings, get_settings
from bot_cripto.core.market import market_domain
from bot_cripto.data.crypto_feeds import CryptoFeedLayer
from bot_cripto.data.forex_feeds import ForexFeedLayer
from bot_cripto.features.layer import FeatureLayer
from bot_cripto.jobs.inference import run as inference_run
from bot_cripto.jobs.train_meta import run as train_meta_run
from bot_cripto.jobs.train_return import run as train_return_run
from bot_cripto.jobs.train_risk import run as train_risk_run
from bot_cripto.jobs.train_trend import run as train_trend_run
from bot_cripto.jobs.common import safe_symbol


def _now_utc() -> str:
    return datetime.now(tz=UTC).isoformat()


def _compact_utc() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")


def _fetch_symbol_data(
    *,
    settings: Settings,
    symbol: str,
    timeframe: str,
    days: int,
) -> dict[str, Any]:
    domain = market_domain(symbol)
    if domain == "forex":
        result = ForexFeedLayer(settings).fetch_history(symbol=symbol, timeframe=timeframe, days=days)
    else:
        result = CryptoFeedLayer(settings).fetch_history(symbol=symbol, timeframe=timeframe, days=days)
    return {
        "provider": result.provider,
        "rows": int(result.rows),
        "path": result.path,
    }


def _build_features(
    *,
    settings: Settings,
    symbol: str,
    timeframe: str,
) -> dict[str, Any]:
    raw_path = settings.data_dir_raw / f"{safe_symbol(symbol)}_{timeframe}.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")
    df = pd.read_parquet(raw_path)
    features = FeatureLayer().build(symbol=symbol, df=df)
    out_path = settings.data_dir_processed / f"{safe_symbol(symbol)}_{timeframe}_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, compression="snappy")
    return {
        "path": str(out_path),
        "rows": int(len(features)),
        "cols": int(features.shape[1]),
    }


def run_dual_pipeline(
    *,
    settings: Settings | None = None,
    symbols: list[str] | None = None,
    timeframe: str | None = None,
    days: int = 30,
    include_meta: bool = True,
    skip_fetch: bool = False,
    skip_train: bool = False,
    skip_inference: bool = False,
) -> dict[str, Any]:
    cfg = settings or get_settings()
    cfg.ensure_dirs()
    target_symbols = symbols or cfg.symbols_list
    tf = timeframe or cfg.timeframe

    report: dict[str, Any] = {
        "ts_start": _now_utc(),
        "timeframe": tf,
        "days": int(days),
        "include_meta": bool(include_meta),
        "skip_fetch": bool(skip_fetch),
        "skip_train": bool(skip_train),
        "skip_inference": bool(skip_inference),
        "symbols": [],
    }

    ok_count = 0
    fail_count = 0

    for symbol in target_symbols:
        item: dict[str, Any] = {
            "symbol": symbol,
            "market_domain": market_domain(symbol),
            "status": "ok",
            "steps": {},
        }
        try:
            if not skip_fetch:
                item["steps"]["fetch"] = _fetch_symbol_data(
                    settings=cfg,
                    symbol=symbol,
                    timeframe=tf,
                    days=days,
                )

            item["steps"]["features"] = _build_features(
                settings=cfg,
                symbol=symbol,
                timeframe=tf,
            )

            if not skip_train:
                item["steps"]["train_trend"] = {"model_dir": train_trend_run(symbol=symbol, timeframe=tf)}
                item["steps"]["train_return"] = {"model_dir": train_return_run(symbol=symbol, timeframe=tf)}
                item["steps"]["train_risk"] = {"model_dir": train_risk_run(symbol=symbol, timeframe=tf)}
                if include_meta:
                    try:
                        item["steps"]["train_meta"] = {"model_dir": train_meta_run(symbol=symbol, timeframe=tf)}
                    except Exception as exc:
                        # Meta is an enhancement layer; keep pipeline alive if not enough samples.
                        item["steps"]["train_meta"] = {
                            "status": "skipped",
                            "reason": str(exc),
                        }

            if not skip_inference:
                payload = inference_run(symbol=symbol, timeframe=tf)
                item["steps"]["inference"] = {
                    "decision": payload.get("decision", "NO_TRADE"),
                    "confidence": float(payload.get("confidence", 0.0)),
                    "reason": payload.get("reason", ""),
                }

            ok_count += 1
        except Exception as exc:
            item["status"] = "failed"
            item["error"] = str(exc)
            fail_count += 1
        report["symbols"].append(item)

    report["status"] = "ok" if fail_count == 0 else "partial_error"
    report["summary"] = {
        "symbols_total": len(target_symbols),
        "symbols_ok": ok_count,
        "symbols_failed": fail_count,
    }
    report["ts_end"] = _now_utc()

    out_path = cfg.logs_dir / f"dual_pipeline_{_compact_utc()}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["report_path"] = str(out_path)
    return report
