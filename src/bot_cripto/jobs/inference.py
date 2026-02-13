from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from bot_cripto.core.config import get_settings
from bot_cripto.core.logging import get_logger
from bot_cripto.decision.engine import Action, DecisionEngine
from bot_cripto.jobs.common import latest_model_dir, load_feature_dataset, write_signal_json
from bot_cripto.models.base import PredictionOutput
from bot_cripto.models.baseline import BaselineModel
from bot_cripto.models.ensemble import WeightedEnsemble
from bot_cripto.monitoring.performance_store import PerformancePoint, PerformanceStore
from bot_cripto.monitoring.watchtower_store import WatchtowerStore
from bot_cripto.notifications.telegram import TelegramNotifier
from bot_cripto.ops.operator_flags import default_flags_store
from bot_cripto.regime.engine import MarketRegime, RegimeEngine
from bot_cripto.risk.engine import RiskEngine, RiskLimits
from bot_cripto.risk.state_store import RiskStateStore

logger = get_logger("jobs.inference")


def _load_model_and_path(
    model_name: str, symbol: str, timeframe: str | None = None
) -> tuple[BaselineModel, Path] | None:
    settings = get_settings()
    try:
        path = latest_model_dir(settings, model_name, symbol, timeframe=timeframe)
        model = BaselineModel()
        model.load(path)
        logger.info("model_loaded", model=model_name, path=str(path))
        return model, path
    except FileNotFoundError:
        logger.warning("model_missing", model=model_name, symbol=symbol)
        return None


def _resolve_model(
    primary: tuple[BaselineModel, Path] | None,
    fallback: tuple[BaselineModel, Path],
) -> tuple[BaselineModel, Path]:
    return primary if primary is not None else fallback


def _to_contract_decision(action: Action, blocked: bool) -> str:
    if blocked:
        return "NO_TRADE"
    if action == Action.BUY:
        return "LONG"
    if action == Action.SELL:
        return "SHORT"
    return "NO_TRADE"


def run(symbol: str | None = None, timeframe: str | None = None) -> dict[str, Any]:
    started = perf_counter()
    settings = get_settings()
    target = symbol or settings.symbols_list[0]
    tf = timeframe or settings.timeframe
    df = load_feature_dataset(settings, target, timeframe=tf)

    trend = _load_model_and_path("trend", target, timeframe=tf)
    ret = _load_model_and_path("return", target, timeframe=tf)
    risk = _load_model_and_path("risk", target, timeframe=tf)
    fallback = _load_model_and_path("baseline", target, timeframe=tf)

    if fallback is None and (trend is None or ret is None or risk is None):
        raise FileNotFoundError("No trend/return/risk models or baseline model found.")

    if fallback is None:
        assert trend is not None
        assert ret is not None
        assert risk is not None
        trend_model, trend_path = trend
        ret_model, ret_path = ret
        risk_model, risk_path = risk
    else:
        trend_model, trend_path = _resolve_model(trend, fallback)
        ret_model, ret_path = _resolve_model(ret, fallback)
        risk_model, risk_path = _resolve_model(risk, fallback)

    pred_trend = trend_model.predict(df)
    pred_return = ret_model.predict(df)
    pred_risk = risk_model.predict(df)

    ensemble = WeightedEnsemble()
    merged: PredictionOutput = ensemble.combine(pred_trend, pred_return, pred_risk)

    regime_engine = RegimeEngine(
        adx_trend_min=settings.regime_adx_trend_min,
        atr_high_vol_pct=settings.regime_atr_high_vol_pct,
    )
    regime = regime_engine.detect(df)

    risk_engine = RiskEngine(
        limits=RiskLimits(
            risk_per_trade=settings.risk_per_trade,
            max_daily_drawdown=settings.max_daily_drawdown,
            max_weekly_drawdown=settings.max_weekly_drawdown,
            max_position_size=settings.max_position_size,
            risk_score_block_threshold=settings.risk_score_block_threshold,
            position_size_multiplier=settings.risk_position_size_multiplier,
        )
    )
    safe_symbol = target.replace("/", "_")
    risk_state_store = RiskStateStore(settings.logs_dir / f"risk_state_{safe_symbol}.json")
    state = risk_state_store.load(initial_equity=settings.initial_equity)
    risk_decision = risk_engine.evaluate(merged, regime.regime, state)
    risk_state_store.save(state)

    engine = DecisionEngine()
    price = float(df["close"].iloc[-1])
    signal = engine.decide(merged, current_price=price)

    blocked_by_regime = regime.regime in {MarketRegime.RANGE, MarketRegime.HIGH_VOL}
    flags = default_flags_store(settings).load()
    operator_paused = flags.is_paused()
    # Pause blocks new entries, but still allows exits (SELL) to reduce risk.
    blocked_by_operator = operator_paused and signal.action == Action.BUY
    blocked = blocked_by_regime or (not risk_decision.allowed) or blocked_by_operator

    decision = _to_contract_decision(signal.action, blocked=blocked)
    confidence = signal.confidence if not blocked else 0.0
    if not blocked:
        reason = signal.reason
    else:
        parts: list[str] = []
        if blocked_by_operator:
            parts.append("operator_pause")
        parts.append(regime.reason)
        parts.append(risk_decision.reason)
        reason = "Blocked: " + "; ".join([p for p in parts if p])

    model_version = ",".join(
        [
            trend_path.name,
            ret_path.name,
            risk_path.name,
        ]
    )

    payload: dict[str, Any] = {
        "ts": datetime.now(tz=UTC).isoformat(),
        "symbol": target,
        "timeframe": tf,
        "horizon_steps": settings.pred_horizon_steps,
        **merged.to_dict(),
        "decision": decision,
        "confidence": float(confidence),
        "reason": reason,
        "regime": regime.regime.value,
        "regime_reason": regime.reason,
        "position_size": float(risk_decision.position_size),
        "risk_allowed": bool(risk_decision.allowed),
        "equity": float(state.equity),
        "version": {
            "git_commit": trend_path.name.split("_")[-1] if "_" in trend_path.name else "unknown",
            "model_version": model_version,
        },
    }

    out_path = settings.logs_dir / "signal.json"
    write_signal_json(out_path, payload)
    latency_ms = (perf_counter() - started) * 1000
    watchtower = WatchtowerStore(settings.watchtower_db_path)
    watchtower.log_decision(payload, latency_ms=latency_ms)
    watchtower.log_equity(payload["ts"], float(state.equity), source="inference")
    perf_store = PerformanceStore(settings.logs_dir / f"performance_history_{safe_symbol}.json")
    perf_store.append(
        PerformancePoint(
            ts=payload["ts"],
            metric=float(payload["expected_return"] * payload["position_size"]),
        )
    )
    notifier = TelegramNotifier(settings=settings)
    if payload["decision"] == "NO_TRADE":
        notifier.send(
            f"[inference] {target} {payload['decision']} regime={payload['regime']} "
            f"risk_allowed={payload['risk_allowed']}"
        )
    else:
        notifier.send(
            f"[signal] {target} {payload['decision']} conf={payload['confidence']:.2f} "
            f"ret={payload['expected_return']:.4f} size={payload['position_size']:.3f}"
        )
    logger.info("inference_done", path=str(out_path), decision=payload["decision"])
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--timeframe", type=str, default=None)
    args = parser.parse_args()
    print(run(symbol=args.symbol, timeframe=args.timeframe))


if __name__ == "__main__":
    main()
