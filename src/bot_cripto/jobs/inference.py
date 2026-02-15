from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from bot_cripto.core.config import get_settings
from bot_cripto.core.logging import get_logger
from bot_cripto.data.quant_signals import QuantSignalFetcher
from bot_cripto.decision.engine import Action, DecisionEngine
from bot_cripto.jobs.common import latest_model_dir, load_feature_dataset, write_signal_json
from bot_cripto.models.base import BasePredictor, PredictionOutput
from bot_cripto.models.baseline import BaselineModel
from bot_cripto.models.ensemble import EnsembleWeights, WeightedEnsemble
from bot_cripto.models.meta import MetaModel
from bot_cripto.models.tft import TFTPredictor
from bot_cripto.monitoring.watchtower_store import WatchtowerStore
from bot_cripto.notifications.telegram import TelegramNotifier
from bot_cripto.ops.operator_flags import default_flags_store
from bot_cripto.regime.ml_engine import MLRegimeEngine
from bot_cripto.risk.engine import RiskEngine, RiskLimits
from bot_cripto.risk.state_store import RiskStateStore

logger = get_logger("jobs.inference")


def _load_model_and_path(
    model_name: str, symbol: str, timeframe: str | None = None
) -> tuple[BasePredictor, Path] | None:
    settings = get_settings()
    try:
        path = latest_model_dir(settings, model_name, symbol, timeframe=timeframe)
        if model_name == "nbeats" and (path / "model.pt").exists():
            from bot_cripto.models.nbeats import NBeatsPredictor
            model = NBeatsPredictor()
            logger.info("model_type_detected", type="nbeats", path=str(path))
        elif (path / "model.pt").exists():
            model = TFTPredictor()
            logger.info("model_type_detected", type="tft", path=str(path))
        else:
            model = BaselineModel()
            logger.info("model_type_detected", type="baseline", path=str(path))

        model.load(path)
        logger.info("model_loaded", model=model_name, path=str(path))
        return model, path
    except FileNotFoundError:
        logger.warning("model_missing", model=model_name, symbol=symbol)
        return None


def _resolve_model(
    primary: tuple[BasePredictor, Path] | None,
    fallback: tuple[BasePredictor, Path] | None,
) -> tuple[BasePredictor, Path]:
    result = primary if primary is not None else fallback
    if result is None:
        raise FileNotFoundError(
            "No model available: both primary and fallback are missing."
        )
    return result


def _to_contract_decision(action: Action, blocked: bool) -> str:
    if blocked:
        return "NO_TRADE"
    if action == Action.BUY:
        return "LONG"
    if action == Action.SELL:
        return "SHORT"
    return "NO_TRADE"


def _fetch_quant_signals_safe(
    settings: object, target: str
) -> dict[str, float]:
    """Fetch quant signals with fallback to cached/neutral values."""
    try:
        fetcher = QuantSignalFetcher(settings)
        funding = fetcher.fetch_funding_rate(target)
        fng = fetcher.fetch_fear_and_greed()
        fetcher.save_signals(target, funding, fng)
        return {"funding_rate": funding, "fear_greed": fng}
    except Exception as exc:
        logger.warning("quant_signals_fetch_failed_using_defaults", error=str(exc))
        return {"funding_rate": 0.0, "fear_greed": 0.5}


def run(symbol: str | None = None, timeframe: str | None = None) -> dict[str, Any]:
    started = perf_counter()
    settings = get_settings()
    target = symbol or settings.symbols_list[0]
    tf = timeframe or settings.timeframe
    df = load_feature_dataset(settings, target, timeframe=tf)

    # 1. Quant Signals (with fallback — never blocks inference)
    q_data = _fetch_quant_signals_safe(settings, target)

    # 2. Load models
    trend = _load_model_and_path("trend", target, timeframe=tf)
    ret = _load_model_and_path("return", target, timeframe=tf)
    risk = _load_model_and_path("risk", target, timeframe=tf)
    nbeats = _load_model_and_path("nbeats", target, timeframe=tf)
    fallback = _load_model_and_path("baseline", target, timeframe=tf)

    if fallback is None and (trend is None or ret is None or risk is None):
        raise FileNotFoundError("No models found: need trend+return+risk or baseline.")

    trend_model, trend_path = _resolve_model(trend, fallback)
    ret_model, ret_path = _resolve_model(ret, fallback)
    risk_model, risk_path = _resolve_model(risk, fallback)

    # 3. Predict & Ensemble
    pred_trend = trend_model.predict(df)
    pred_return = ret_model.predict(df)
    pred_risk = risk_model.predict(df)

    pred_nbeats: PredictionOutput | None = None
    if nbeats is not None:
        nbeats_model, _ = nbeats
        pred_nbeats = nbeats_model.predict(df)

    weights = EnsembleWeights(
        trend=0.30, ret=0.25, risk=0.25, nbeats=0.20
    ) if pred_nbeats is not None else EnsembleWeights()
    ensemble = WeightedEnsemble(weights=weights)
    merged: PredictionOutput = ensemble.combine(
        pred_trend, pred_return, pred_risk, nbeats_pred=pred_nbeats
    )

    # 4. ML Regime Detection (never train during inference)
    regime_engine = MLRegimeEngine()
    regime_path = settings.models_dir / "regime" / target.replace("/", "_")
    if (regime_path / "regime_model.joblib").exists():
        regime_engine.load(regime_path)
        regime_str = regime_engine.predict(df)
    else:
        logger.warning("regime_model_missing_using_unknown", path=str(regime_path))
        regime_str = "UNKNOWN"

    # 5. Meta-model Filter
    meta_model = MetaModel()
    meta_path = settings.models_dir / "meta" / target.replace("/", "_")
    if (meta_path / "meta_model.joblib").exists():
        meta_model.load(meta_path)
    
    meta_blocked = meta_model.should_filter(merged.to_dict(), regime_str, q_data)

    # 6. Risk & Decision
    risk_engine = RiskEngine(
        limits=RiskLimits(
            risk_per_trade=settings.risk_per_trade,
            max_daily_drawdown=settings.max_daily_drawdown,
            position_size_multiplier=settings.risk_position_size_multiplier,
        )
    )
    safe_symbol = target.replace("/", "_")
    risk_state_store = RiskStateStore(settings.logs_dir / f"risk_state_{safe_symbol}.json")
    state = risk_state_store.load(initial_equity=settings.initial_equity)
    risk_decision = risk_engine.evaluate(merged, regime_str, state)
    risk_state_store.save(state)

    engine = DecisionEngine()
    price = float(df["close"].iloc[-1])
    signal = engine.decide(merged, current_price=price)

    # Multi-layered blocking logic
    flags = default_flags_store(settings).load()
    blocked_by_operator = flags.is_paused() and signal.action == Action.BUY
    
    # Meta-model or Risk engine can block the trade
    blocked = (not risk_decision.allowed) or meta_blocked or blocked_by_operator

    decision = _to_contract_decision(signal.action, blocked=blocked)
    confidence = getattr(merged, "confidence", signal.confidence) if not blocked else 0.0
    
    reason = signal.reason if not blocked else f"Blocked: risk={risk_decision.reason}, meta={meta_blocked}"

    payload: dict[str, Any] = {
        "ts": datetime.now(tz=UTC).isoformat(),
        "symbol": target,
        "timeframe": tf,
        "horizon_steps": settings.pred_horizon_steps,
        **merged.to_dict(),
        "decision": decision,
        "confidence": float(confidence),
        "reason": reason,
        "regime": regime_str,
        "position_size": float(risk_decision.position_size),
        "risk_allowed": bool(risk_decision.allowed),
        "equity": float(state.equity),
        "quant_signals": q_data,
        "version": {"model_version": f"{trend_path.name},{ret_path.name}"}
    }

    out_path = settings.logs_dir / "signal.json"
    write_signal_json(out_path, payload)
    
    # Notifications and logging
    watchtower = WatchtowerStore(settings.watchtower_db_path)
    watchtower.log_decision(payload, latency_ms=(perf_counter() - started) * 1000)
    
    notifier = TelegramNotifier(settings=settings)
    notifier.send(
        f"[{'signal' if not blocked else 'inference'}] {target} {decision} "
        f"regime={regime_str} conf={confidence:.2f} size={payload['position_size']:.3f}"
    )
    
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--timeframe", type=str, default=None)
    args = parser.parse_args()
    print(run(symbol=args.symbol, timeframe=args.timeframe))


if __name__ == "__main__":
    main()
