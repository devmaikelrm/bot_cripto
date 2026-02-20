from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from bot_cripto.core.config import Settings, get_settings
from bot_cripto.core.logging import get_logger
from bot_cripto.core.market import market_domain
from bot_cripto.data.quant_signals import QuantSignalFetcher
from bot_cripto.decision.engine import Action, DecisionEngine
from bot_cripto.jobs.common import latest_model_dir, load_feature_dataset, write_signal_json
from bot_cripto.models.base import BasePredictor, PredictionOutput
from bot_cripto.models.baseline import BaselineModel
from bot_cripto.models.ensemble import EnsembleWeights, WeightedEnsemble
from bot_cripto.models.meta import MetaModel
from bot_cripto.monitoring.performance_store import PerformanceStore
from bot_cripto.models.tft import TFTPredictor
from bot_cripto.monitoring.watchtower_store import WatchtowerStore
from bot_cripto.notifications.telegram import TelegramNotifier
from bot_cripto.ops.operator_flags import default_flags_store
from bot_cripto.regime.ml_engine import MLRegimeEngine
from bot_cripto.risk.engine import RiskEngine, RiskLimits
from bot_cripto.risk.state_store import RiskStateStore

logger = get_logger("jobs.inference")


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


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


def _dynamic_kelly_fraction(settings: Settings, trend_model_path: Path) -> float:
    """Scale Kelly fraction from recent val_loss into configured [min, max] band."""
    base = float(settings.risk_kelly_fraction)
    if not settings.risk_kelly_dynamic_enabled:
        return base
    min_k = float(settings.risk_kelly_fraction_min)
    max_k = float(settings.risk_kelly_fraction_max)
    if max_k < min_k:
        min_k, max_k = max_k, min_k
    val_loss: float | None = None
    meta_path = trend_model_path / "metadata.json"
    if meta_path.exists():
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            val_raw = payload.get("metrics", {}).get("val_loss")
            if val_raw is not None:
                val_loss = float(val_raw)
        except Exception:
            val_loss = None
    if val_loss is None:
        return float(min(max(base, min_k), max_k))

    good = float(settings.risk_kelly_val_loss_good)
    bad = float(settings.risk_kelly_val_loss_bad)
    if bad <= good:
        bad = good * 10.0
    if val_loss <= good:
        return max_k
    if val_loss >= bad:
        return min_k
    # Linear interpolation: better loss => bigger Kelly fraction.
    ratio = (bad - val_loss) / (bad - good)
    return float(min_k + (max_k - min_k) * ratio)


def _default_quant_signals() -> dict[str, float]:
    return {
        "funding_rate": 0.0,
        "fear_greed": 0.5,
        "open_interest": 0.0,
        "long_short_ratio": 1.0,
        "orderbook_imbalance": 0.0,
        "social_sentiment": 0.5,
        "social_sentiment_raw": 0.5,
        "social_sentiment_anomaly": 0.0,
        "social_sentiment_zscore": 0.0,
        "social_sentiment_velocity": 0.0,
        "social_sentiment_acceleration": 0.0,
        "social_sentiment_x": 0.5,
        "social_sentiment_news": 0.5,
        "social_sentiment_telegram": 0.5,
        "social_sentiment_retail": 0.5,
        "social_sentiment_institutional": 0.5,
        "social_sentiment_contrarian": 0.0,
        "social_sentiment_regime_code": 0.0,
        "social_sentiment_reliability_x": 1.0,
        "social_sentiment_reliability_news": 1.0,
        "social_sentiment_reliability_telegram": 1.0,
        "sp500_ret_1d": 0.0,
        "dxy_ret_1d": 0.0,
        "corr_btc_sp500": 0.0,
        "corr_btc_dxy": 0.0,
        "macro_risk_off_score": 0.5,
    }


def _fetch_quant_signals_safe(
    settings: Settings, target: str, df: Any
) -> dict[str, float]:
    """Fetch quant signals with fallback to cached/neutral values."""
    try:
        fetcher = QuantSignalFetcher(settings)
        domain = market_domain(target)

        # For forex symbols, avoid crypto derivatives/orderbook endpoints.
        if domain == "forex":
            funding = 0.0
            oi = 0.0
            lsr = 1.0
            obi = 0.0
        else:
            funding = fetcher.fetch_funding_rate(target)
            oi = fetcher.fetch_open_interest(target)
            lsr = fetcher.fetch_long_short_ratio(target)
            obi = fetcher.fetch_orderbook_imbalance(target)

        fng = fetcher.fetch_fear_and_greed()
        social_bundle = fetcher.fetch_social_sentiment_bundle(target)
        regime_map = {"PANIC": -1.0, "APATHY": 0.0, "NEUTRAL": 0.25, "EUPHORIA": 1.0}
        regime_code = float(regime_map.get(str(social_bundle.get("social_sentiment_regime", "APATHY")), 0.0))
        social = float(social_bundle["social_sentiment"])
        macro = fetcher.fetch_macro_context(df["close"])
        fetcher.save_signals(
            target,
            funding,
            fng,
            open_interest=oi,
            long_short_ratio=lsr,
            orderbook_imbalance=obi,
            social_sentiment=social,
            social_sentiment_raw=float(social_bundle["social_sentiment_raw"]),
            social_sentiment_anomaly=float(social_bundle["social_sentiment_anomaly"]),
            social_sentiment_zscore=float(social_bundle["social_sentiment_zscore"]),
            social_sentiment_velocity=float(social_bundle["social_sentiment_velocity"]),
            social_sentiment_acceleration=float(social_bundle.get("social_sentiment_acceleration", 0.0)),
            social_sentiment_x=float(social_bundle["social_sentiment_x"]),
            social_sentiment_news=float(social_bundle["social_sentiment_news"]),
            social_sentiment_telegram=float(social_bundle["social_sentiment_telegram"]),
            social_sentiment_retail=float(social_bundle.get("social_sentiment_retail", 0.5)),
            social_sentiment_institutional=float(social_bundle.get("social_sentiment_institutional", 0.5)),
            social_sentiment_contrarian=float(social_bundle.get("social_sentiment_contrarian", 0.0)),
            social_sentiment_regime=str(social_bundle.get("social_sentiment_regime", "APATHY")),
            social_sentiment_reliability_x=float(social_bundle["social_sentiment_reliability_x"]),
            social_sentiment_reliability_news=float(social_bundle["social_sentiment_reliability_news"]),
            social_sentiment_reliability_telegram=float(
                social_bundle["social_sentiment_reliability_telegram"]
            ),
            sp500_ret_1d=macro["sp500_ret_1d"],
            dxy_ret_1d=macro["dxy_ret_1d"],
            corr_btc_sp500=macro["corr_btc_sp500"],
            corr_btc_dxy=macro["corr_btc_dxy"],
            macro_risk_off_score=macro["macro_risk_off_score"],
        )
        return {
            "funding_rate": funding,
            "fear_greed": fng,
            "open_interest": oi,
            "long_short_ratio": lsr,
            "orderbook_imbalance": obi,
            "social_sentiment": social,
            "social_sentiment_raw": float(social_bundle["social_sentiment_raw"]),
            "social_sentiment_anomaly": float(social_bundle["social_sentiment_anomaly"]),
            "social_sentiment_zscore": float(social_bundle["social_sentiment_zscore"]),
            "social_sentiment_velocity": float(social_bundle["social_sentiment_velocity"]),
            "social_sentiment_acceleration": float(social_bundle.get("social_sentiment_acceleration", 0.0)),
            "social_sentiment_x": float(social_bundle["social_sentiment_x"]),
            "social_sentiment_news": float(social_bundle["social_sentiment_news"]),
            "social_sentiment_telegram": float(social_bundle["social_sentiment_telegram"]),
            "social_sentiment_retail": float(social_bundle.get("social_sentiment_retail", 0.5)),
            "social_sentiment_institutional": float(social_bundle.get("social_sentiment_institutional", 0.5)),
            "social_sentiment_contrarian": float(social_bundle.get("social_sentiment_contrarian", 0.0)),
            "social_sentiment_regime_code": regime_code,
            "social_sentiment_reliability_x": float(social_bundle["social_sentiment_reliability_x"]),
            "social_sentiment_reliability_news": float(social_bundle["social_sentiment_reliability_news"]),
            "social_sentiment_reliability_telegram": float(
                social_bundle["social_sentiment_reliability_telegram"]
            ),
            "sp500_ret_1d": macro["sp500_ret_1d"],
            "dxy_ret_1d": macro["dxy_ret_1d"],
            "corr_btc_sp500": macro["corr_btc_sp500"],
            "corr_btc_dxy": macro["corr_btc_dxy"],
            "macro_risk_off_score": macro["macro_risk_off_score"],
        }
    except Exception as exc:
        logger.warning("quant_signals_fetch_failed_using_defaults", error=str(exc))
        return _default_quant_signals()


def _detect_volatility_mode(settings: Settings, df: Any) -> tuple[str, float]:
    """Return mode + realised volatility (std returns) for recent window."""
    try:
        close = df["close"].astype(float)
        ret = close.pct_change().dropna()
        if len(ret) < 20:
            return "NORMAL", 0.0
        recent = ret.tail(96)
        vol = float(recent.std())
        mode = "CRISIS_HIGH_VOL" if vol >= settings.regime_atr_high_vol_pct else "NORMAL"
        return mode, vol
    except Exception:
        return "NORMAL", 0.0


def _is_macro_event_window(settings: Settings, now_utc: datetime) -> bool:
    if not settings.macro_event_crisis_enabled:
        return False
    if now_utc.weekday() not in settings.macro_event_weekdays:
        return False
    now_t = now_utc.time().replace(second=0, microsecond=0)
    for start, end in settings.macro_event_windows:
        if start <= end:
            if start <= now_t <= end:
                return True
        else:
            if now_t >= start or now_t <= end:
                return True
    return False


def _apply_context_adjustments(
    pred: PredictionOutput,
    q_data: dict[str, float],
    settings: Settings,
) -> tuple[PredictionOutput, dict[str, float]]:
    """Apply lightweight context adjustment from sentiment/orderbook/macro."""
    social = _clamp(float(q_data.get("social_sentiment", 0.5)), 0.0, 1.0)
    social_anomaly = _clamp(float(q_data.get("social_sentiment_anomaly", 0.0)), 0.0, 1.0)
    social_accel = _clamp(float(q_data.get("social_sentiment_acceleration", 0.0)), -1.0, 1.0)
    social_contrarian = _clamp(float(q_data.get("social_sentiment_contrarian", 0.0)), -1.0, 1.0)
    social_retail = _clamp(float(q_data.get("social_sentiment_retail", 0.5)), 0.0, 1.0)
    social_inst = _clamp(float(q_data.get("social_sentiment_institutional", 0.5)), 0.0, 1.0)
    social_regime = _clamp(float(q_data.get("social_sentiment_regime_code", 0.0)), -1.0, 1.0)
    orderbook = _clamp(float(q_data.get("orderbook_imbalance", 0.0)), -1.0, 1.0)
    macro_risk_off = _clamp(float(q_data.get("macro_risk_off_score", 0.5)), 0.0, 1.0)
    sp500_ret = float(q_data.get("sp500_ret_1d", 0.0))
    dxy_ret = float(q_data.get("dxy_ret_1d", 0.0))
    corr_spx = float(q_data.get("corr_btc_sp500", 0.0))
    corr_dxy = float(q_data.get("corr_btc_dxy", 0.0))

    social_component = (social - 0.5) * 2.0
    macro_component = (0.5 - macro_risk_off) * 2.0

    corr_component = 0.0
    if corr_spx > 0.2:
        corr_component += _clamp(sp500_ret * 20.0, -1.0, 1.0)
    if corr_dxy < -0.2:
        corr_component += _clamp((-dxy_ret) * 20.0, -1.0, 1.0)
    corr_component = _clamp(corr_component / 2.0, -1.0, 1.0)

    # If retail euphoria conflicts with bearish orderbook flow, downweight sentiment.
    social_orderflow_div = social_component - orderbook
    social_flow_penalty = -0.2 * abs(social_orderflow_div) if abs(social_orderflow_div) > 0.8 else 0.0

    context_score = _clamp(
        (0.35 * social_component)
        + (0.35 * orderbook)
        + (0.20 * macro_component)
        + (0.10 * corr_component)
        - (0.15 * social_anomaly),
        -1.0,
        1.0,
    )
    context_score = _clamp(
        context_score
        + (0.10 * social_accel)
        + (0.15 * social_inst - 0.10 * social_retail)
        + (0.15 * social_contrarian)
        + (0.08 * social_regime)
        + social_flow_penalty,
        -1.0,
        1.0,
    )

    delta_prob = context_score * settings.context_prob_adjust_max
    adj_prob = _clamp(pred.prob_up + delta_prob, 0.0, 1.0)
    adj_return = float(pred.expected_return * (1.0 + 0.25 * context_score))
    adj_risk = _clamp(
        pred.risk_score * (1.0 - 0.20 * context_score) * (1.0 + (0.25 * social_anomaly)),
        0.0,
        1.0,
    )

    adjusted = PredictionOutput(
        prob_up=adj_prob,
        expected_return=adj_return,
        p10=pred.p10,
        p50=pred.p50,
        p90=pred.p90,
        risk_score=adj_risk,
    )
    debug = {
        "context_score": context_score,
        "context_prob_delta": delta_prob,
        "social_component": social_component,
        "orderbook_component": orderbook,
        "macro_component": macro_component,
        "corr_component": corr_component,
        "social_accel_component": 0.10 * social_accel,
        "social_contrarian_component": 0.15 * social_contrarian,
        "social_inst_component": 0.15 * social_inst,
        "social_retail_component": -0.10 * social_retail,
        "social_regime_component": 0.08 * social_regime,
        "social_orderflow_penalty": social_flow_penalty,
        "anomaly_component": -0.15 * social_anomaly,
    }
    return adjusted, debug


def _calibrate_prediction(pred: PredictionOutput, model_path: Path) -> PredictionOutput:
    """Apply probability calibration if a calibrator exists for this model."""
    cal_path = model_path / "calibrator.joblib"
    if not cal_path.exists():
        return pred
    try:
        from bot_cripto.models.calibration import ProbabilityCalibrator
        calibrator = ProbabilityCalibrator()
        calibrator.load(cal_path)
        raw = np.array([pred.prob_up])
        calibrated = float(calibrator.predict(raw)[0])
        logger.info("prediction_calibrated", raw=pred.prob_up, calibrated=calibrated)
        return PredictionOutput(
            prob_up=calibrated,
            expected_return=pred.expected_return,
            p10=pred.p10,
            p50=pred.p50,
            p90=pred.p90,
            risk_score=pred.risk_score,
        )
    except Exception as exc:
        logger.warning("calibration_failed_using_raw", error=str(exc))
        return pred


def run(symbol: str | None = None, timeframe: str | None = None) -> dict[str, Any]:
    started = perf_counter()
    settings = get_settings()
    target = symbol or settings.symbols_list[0]
    tf = timeframe or settings.timeframe
    df = load_feature_dataset(settings, target, timeframe=tf)

    # 1. Quant Signals (with fallback — never blocks inference)
    q_data = _fetch_quant_signals_safe(settings, target, df)

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

    # 3. Predict, Calibrate & Ensemble
    pred_trend = _calibrate_prediction(trend_model.predict(df), trend_path)
    pred_return = _calibrate_prediction(ret_model.predict(df), ret_path)
    pred_risk = _calibrate_prediction(risk_model.predict(df), risk_path)

    pred_nbeats: PredictionOutput | None = None
    if nbeats is not None:
        nbeats_model, nbeats_path = nbeats
        pred_nbeats = _calibrate_prediction(nbeats_model.predict(df), nbeats_path)

    weights = EnsembleWeights(
        trend=0.30, ret=0.25, risk=0.25, nbeats=0.20
    ) if pred_nbeats is not None else EnsembleWeights()
    ensemble = WeightedEnsemble(weights=weights)
    merged: PredictionOutput = ensemble.combine(
        pred_trend, pred_return, pred_risk, nbeats_pred=pred_nbeats
    )

    # 3.1 Context adjustments (sentiment/orderbook/macro)
    merged_raw = merged
    merged, context_debug = _apply_context_adjustments(merged_raw, q_data, settings)

    # 4. ML Regime Detection (never train during inference)
    regime_engine = MLRegimeEngine()
    regime_path = settings.models_dir / "regime" / target.replace("/", "_")
    if (regime_path / "regime_model.joblib").exists():
        regime_engine.load(regime_path)
        regime_str = regime_engine.predict(df)
    else:
        regime_str = "RANGE_SIDEWAYS"
        logger.warning(
            "regime_model_missing_using_safe_fallback",
            path=str(regime_path),
            fallback=regime_str,
        )

    # 4.1 Volatility crisis mode override
    vol_mode, realised_vol = _detect_volatility_mode(settings, df)
    macro_event_mode = _is_macro_event_window(settings, datetime.now(tz=UTC))
    effective_regime = (
        "CRISIS_HIGH_VOL"
        if (vol_mode == "CRISIS_HIGH_VOL" or macro_event_mode)
        else regime_str
    )

    # 5. Meta-model Filter
    meta_model = MetaModel(min_prob_success=settings.meta_model_min_prob_success)
    meta_path: Path | None = None
    if settings.meta_model_enabled:
        try:
            meta_path = latest_model_dir(settings, "meta", target, timeframe=tf)
            if (meta_path / "meta_model.joblib").exists():
                meta_model.load(meta_path)
                logger.info("meta_model_loaded", path=str(meta_path))
        except FileNotFoundError:
            logger.info("meta_model_missing_skip_filter", symbol=target, timeframe=tf)
    else:
        logger.info("meta_model_disabled")

    meta_prob_success = meta_model.predict_success_prob(
        merged.to_dict(), effective_regime, q_data
    )
    meta_blocked = (
        settings.meta_model_enabled
        and meta_model.should_filter(merged.to_dict(), effective_regime, q_data)
    )

    # 6. Risk & Decision
    risk_engine = RiskEngine(
        limits=RiskLimits(
            risk_per_trade=settings.risk_per_trade,
            max_daily_drawdown=settings.max_daily_drawdown,
            max_weekly_drawdown=settings.max_weekly_drawdown,
            max_position_size=settings.max_position_size,
            risk_score_block_threshold=settings.risk_score_block_threshold,
            position_size_multiplier=settings.risk_position_size_multiplier,
            cooldown_minutes=settings.risk_cooldown_minutes,
            enable_kelly=settings.risk_enable_kelly,
            kelly_fraction=_dynamic_kelly_fraction(settings, trend_path),
            cvar_enabled=settings.risk_cvar_enabled,
            cvar_alpha=settings.risk_cvar_alpha,
            cvar_min_samples=settings.risk_cvar_min_samples,
            cvar_limit=settings.risk_cvar_limit,
            circuit_breaker_minutes=settings.risk_circuit_breaker_minutes,
            long_only=settings.risk_long_only,
            bear_trend_multiplier=settings.risk_bear_trend_multiplier,
            high_score_size_factor=settings.risk_high_score_size_factor,
        )
    )
    safe_symbol = target.replace("/", "_")
    risk_state_store = RiskStateStore(settings.logs_dir / f"risk_state_{safe_symbol}.json")
    state = risk_state_store.load(initial_equity=settings.initial_equity)
    perf_history = PerformanceStore(settings.logs_dir / "performance_history.json").metrics()
    risk_decision = risk_engine.evaluate(
        merged, effective_regime, state, recent_returns=perf_history
    )
    risk_state_store.save(state)

    engine = DecisionEngine()
    price = float(df["close"].iloc[-1])
    signal = engine.decide(merged, current_price=price, regime=effective_regime)

    # Multi-layered blocking logic
    flags = default_flags_store(settings).load()
    blocked_by_operator = flags.is_paused() and signal.action == Action.BUY
    
    # Macro + orderbook gate
    macro_risk_off = float(q_data.get("macro_risk_off_score", 0.5))
    orderbook_imbalance = float(q_data.get("orderbook_imbalance", 0.0))
    macro_blocked = signal.action == Action.BUY and macro_risk_off >= settings.macro_block_threshold
    orderbook_blocked = signal.action == Action.BUY and orderbook_imbalance <= settings.orderbook_sell_wall_threshold

    # Meta-model, risk, macro/orderbook, operator can block the trade
    blocked = (
        (not risk_decision.allowed)
        or meta_blocked
        or blocked_by_operator
        or macro_blocked
        or orderbook_blocked
    )

    decision = _to_contract_decision(signal.action, blocked=blocked)
    confidence = getattr(merged, "confidence", signal.confidence) if not blocked else 0.0

    reason = signal.reason if not blocked else (
        f"Blocked: risk={risk_decision.reason}, meta={meta_blocked}, "
        f"macro={macro_blocked}, orderbook={orderbook_blocked}, operator={blocked_by_operator}"
    )

    # Record last trade timestamp for cooldown
    if not blocked and decision != "NO_TRADE":
        state.last_trade_ts = datetime.now(tz=UTC).isoformat()
        risk_state_store.save(state)

    payload: dict[str, Any] = {
        "ts": datetime.now(tz=UTC).isoformat(),
        "symbol": target,
        "timeframe": tf,
        "horizon_steps": settings.pred_horizon_steps,
        **merged.to_dict(),
        "prob_up_model_raw": float(merged_raw.prob_up),
        "expected_return_model_raw": float(merged_raw.expected_return),
        "risk_score_model_raw": float(merged_raw.risk_score),
        "decision": decision,
        "confidence": float(confidence),
        "reason": reason,
        "regime": effective_regime,
        "regime_base": regime_str,
        "volatility_mode": vol_mode,
        "macro_event_mode": bool(macro_event_mode),
        "realised_volatility": realised_vol,
        "position_size": float(risk_decision.position_size),
        "kelly_fraction_effective": float(risk_engine.limits.kelly_fraction),
        "risk_allowed": bool(risk_decision.allowed),
        "equity": float(state.equity),
        "circuit_breaker_until": state.circuit_breaker_until,
        "quant_signals": q_data,
        "context_adjustment": context_debug,
        "meta_prob_success": float(meta_prob_success),
        "meta_blocked": bool(meta_blocked),
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
        f"regime={effective_regime} conf={confidence:.2f} size={payload['position_size']:.3f}"
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
