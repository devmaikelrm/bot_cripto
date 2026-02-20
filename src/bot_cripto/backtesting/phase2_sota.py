"""Phase 2 SOTA training + OOS benchmark orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import json
import numpy as np
import pandas as pd

from bot_cripto.backtesting.model_benchmark import resolve_model_factory
from bot_cripto.core.config import Settings
from bot_cripto.jobs.common import build_version_dir, write_model_metadata


@dataclass(frozen=True)
class Phase2ModelResult:
    model: str
    status: str
    train_rows: int
    test_rows: int
    mse: float
    mae: float
    accuracy: float
    total_net_return: float
    sharpe: float
    stability_score: float
    artifact_dir: str
    error: str = ""


def _sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns, dtype=float)
    std = float(np.std(arr, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.mean(arr)) / std


def _stability(net_returns: list[float]) -> float:
    if len(net_returns) < 5:
        return 0.0
    arr = np.array(net_returns, dtype=float)
    std = float(np.std(arr))
    if std <= 0.0:
        return 100.0
    # Simple normalized proxy in [0,100].
    score = 100.0 * max(0.0, 1.0 - min(1.0, std / 0.01))
    return round(score, 2)


def _render_markdown_table(rows: list[Phase2ModelResult], summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 2 SOTA OOS Report",
        "",
        f"- Generated at: {summary['generated_at']}",
        f"- Symbol: {summary['symbol']}",
        f"- Timeframe: {summary['timeframe']}",
        f"- Complete success (no skipped/error): {summary['complete_success']}",
        "",
        "| Model | Status | MSE | MAE | Accuracy | Net Return | Sharpe | Stability |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.model} | {r.status} | {r.mse:.8f} | {r.mae:.8f} | {r.accuracy:.4f} | "
            f"{r.total_net_return:.6f} | {r.sharpe:.4f} | {r.stability_score:.2f} |"
        )

    winner = summary.get("winner")
    if winner:
        lines.extend(
            [
                "",
                "## Winner",
                f"- Model: `{winner['model']}`",
                f"- Net return: `{winner['total_net_return']:.6f}`",
                f"- Sharpe: `{winner['sharpe']:.4f}`",
                f"- MSE: `{winner['mse']:.8f}`",
                f"- MAE: `{winner['mae']:.8f}`",
            ]
        )

    deltas = summary.get("winner_deltas_vs_tft")
    if deltas:
        lines.extend(
            [
                "",
                "## Winner vs TFT",
                f"- Delta MSE: `{deltas['mse']:.8f}`",
                f"- Delta MAE: `{deltas['mae']:.8f}`",
                f"- Delta net return: `{deltas['total_net_return']:.6f}`",
                f"- Delta sharpe: `{deltas['sharpe']:.4f}`",
            ]
        )
    return "\n".join(lines) + "\n"


def run_phase2_sota(
    *,
    settings: Settings,
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    model_names: list[str],
    train_frac: float,
) -> tuple[list[Phase2ModelResult], dict[str, Any]]:
    if len(df) < 250:
        raise ValueError("Dataset too small for phase2 SOTA run")

    split_idx = int(len(df) * train_frac)
    split_idx = max(200, min(split_idx, len(df) - 20))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    roundtrip_cost = (2 * settings.fees_bps + settings.spread_bps + settings.slippage_bps) / 10_000
    rows: list[Phase2ModelResult] = []

    for model_name in model_names:
        factory, reason = resolve_model_factory(model_name)
        if factory is None:
            rows.append(
                Phase2ModelResult(
                    model=model_name,
                    status="skipped",
                    train_rows=len(train_df),
                    test_rows=len(test_df),
                    mse=0.0,
                    mae=0.0,
                    accuracy=0.0,
                    total_net_return=0.0,
                    sharpe=0.0,
                    stability_score=0.0,
                    artifact_dir="",
                    error=reason,
                )
            )
            continue

        artifact_dir = ""
        try:
            model = factory()
            metadata = model.train(train_df, target_col="close")
            model_root_name = f"sota_{model_name.strip().lower()}"
            out_dir = build_version_dir(
                settings=settings,
                model_name=model_root_name,
                symbol=symbol,
                metadata=metadata,
                timeframe=timeframe,
            )
            model.save(out_dir)
            write_model_metadata(out_dir, metadata)
            artifact_dir = str(out_dir)

            sq_errors: list[float] = []
            abs_errors: list[float] = []
            net_returns: list[float] = []
            hits = 0
            total = 0

            for i in range(1, len(test_df)):
                window = pd.concat([train_df, test_df.iloc[:i]])
                pred = model.predict(window)

                c_prev = float(test_df["close"].iloc[i - 1])
                c_now = float(test_df["close"].iloc[i])
                if c_prev == 0:
                    continue

                predicted_close = c_prev * (1.0 + float(pred.expected_return))
                err = c_now - predicted_close
                sq_errors.append(err * err)
                abs_errors.append(abs(err))

                realized = (c_now - c_prev) / c_prev
                expected_sign = 1 if pred.expected_return >= 0 else -1
                realized_sign = 1 if realized >= 0 else -1
                net = expected_sign * realized - roundtrip_cost
                net_returns.append(net)
                hits += int(expected_sign == realized_sign)
                total += 1

            mse = float(np.mean(np.array(sq_errors, dtype=float))) if sq_errors else 0.0
            mae = float(np.mean(np.array(abs_errors, dtype=float))) if abs_errors else 0.0
            accuracy = (hits / total) if total > 0 else 0.0
            total_net = float(np.sum(np.array(net_returns, dtype=float))) if net_returns else 0.0
            sharpe = _sharpe(net_returns)
            stability = _stability(net_returns)

            rows.append(
                Phase2ModelResult(
                    model=model_name,
                    status="ok",
                    train_rows=len(train_df),
                    test_rows=len(test_df),
                    mse=mse,
                    mae=mae,
                    accuracy=accuracy,
                    total_net_return=total_net,
                    sharpe=sharpe,
                    stability_score=stability,
                    artifact_dir=artifact_dir,
                )
            )
        except Exception as exc:
            rows.append(
                Phase2ModelResult(
                    model=model_name,
                    status="error",
                    train_rows=len(train_df),
                    test_rows=len(test_df),
                    mse=0.0,
                    mae=0.0,
                    accuracy=0.0,
                    total_net_return=0.0,
                    sharpe=0.0,
                    stability_score=0.0,
                    artifact_dir=artifact_dir,
                    error=str(exc),
                )
            )

    ok_rows = [r for r in rows if r.status == "ok"]
    ok_rows_sorted = sorted(ok_rows, key=lambda r: (r.total_net_return, r.sharpe), reverse=True)
    winner = ok_rows_sorted[0] if ok_rows_sorted else None
    tft = next((r for r in ok_rows if r.model.strip().lower() == "tft"), None)
    complete_success = all(r.status == "ok" for r in rows) if rows else False

    summary: dict[str, Any] = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "symbol": symbol,
        "timeframe": timeframe,
        "rows_total": len(df),
        "rows_train": len(train_df),
        "rows_test": len(test_df),
        "models_requested": model_names,
        "models_ok": len(ok_rows),
        "models_total": len(rows),
        "complete_success": complete_success,
        "winner": (
            {
                "model": winner.model,
                "mse": winner.mse,
                "mae": winner.mae,
                "accuracy": winner.accuracy,
                "total_net_return": winner.total_net_return,
                "sharpe": winner.sharpe,
                "stability_score": winner.stability_score,
            }
            if winner
            else None
        ),
        "winner_deltas_vs_tft": None,
    }
    if winner and tft:
        summary["winner_deltas_vs_tft"] = {
            "mse": float(winner.mse - tft.mse),
            "mae": float(winner.mae - tft.mae),
            "accuracy": float(winner.accuracy - tft.accuracy),
            "total_net_return": float(winner.total_net_return - tft.total_net_return),
            "sharpe": float(winner.sharpe - tft.sharpe),
        }

    return rows, summary


def write_phase2_artifacts(
    *,
    settings: Settings,
    symbol: str,
    timeframe: str,
    rows: list[Phase2ModelResult],
    summary: dict[str, Any],
) -> dict[str, str]:
    safe_symbol = symbol.replace("/", "_")
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    out_json = settings.logs_dir / f"phase2_sota_{safe_symbol}_{timeframe}_{ts}.json"
    out_md = settings.logs_dir / f"phase2_sota_{safe_symbol}_{timeframe}_{ts}.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "summary": summary,
        "results": [r.__dict__ for r in rows],
        "output_markdown": str(out_md),
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out_md.write_text(_render_markdown_table(rows, summary), encoding="utf-8")
    return {"json": str(out_json), "markdown": str(out_md)}
