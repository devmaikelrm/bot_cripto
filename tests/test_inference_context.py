from __future__ import annotations

from datetime import UTC, datetime

from bot_cripto.core.config import Settings
from bot_cripto.jobs.inference import _apply_context_adjustments, _is_macro_event_window
from bot_cripto.models.base import PredictionOutput


def _base_prediction() -> PredictionOutput:
    return PredictionOutput(
        prob_up=0.60,
        expected_return=0.010,
        p10=-0.004,
        p50=0.008,
        p90=0.015,
        risk_score=0.30,
    )


def test_context_adjustments_bullish_increase_prob_and_return() -> None:
    settings = Settings(context_prob_adjust_max=0.05)
    q_data = {
        "social_sentiment": 0.90,
        "orderbook_imbalance": 0.50,
        "macro_risk_off_score": 0.20,
        "sp500_ret_1d": 0.01,
        "dxy_ret_1d": -0.005,
        "corr_btc_sp500": 0.4,
        "corr_btc_dxy": -0.3,
    }

    adjusted, debug = _apply_context_adjustments(_base_prediction(), q_data, settings)
    assert adjusted.prob_up > 0.60
    assert adjusted.expected_return > 0.010
    assert adjusted.risk_score < 0.30
    assert debug["context_score"] > 0.0


def test_context_adjustments_bearish_decrease_prob_and_return() -> None:
    settings = Settings(context_prob_adjust_max=0.05)
    q_data = {
        "social_sentiment": 0.10,
        "orderbook_imbalance": -0.70,
        "macro_risk_off_score": 0.90,
        "sp500_ret_1d": -0.01,
        "dxy_ret_1d": 0.01,
        "corr_btc_sp500": 0.5,
        "corr_btc_dxy": -0.4,
    }

    adjusted, debug = _apply_context_adjustments(_base_prediction(), q_data, settings)
    assert adjusted.prob_up < 0.60
    assert adjusted.expected_return < 0.010
    assert adjusted.risk_score > 0.30
    assert debug["context_score"] < 0.0


def test_context_adjustments_clamp_prob_range() -> None:
    settings = Settings(context_prob_adjust_max=0.30)
    pred = PredictionOutput(
        prob_up=0.99,
        expected_return=0.02,
        p10=-0.01,
        p50=0.015,
        p90=0.03,
        risk_score=0.2,
    )
    q_data = {
        "social_sentiment": 1.0,
        "orderbook_imbalance": 1.0,
        "macro_risk_off_score": 0.0,
        "sp500_ret_1d": 0.03,
        "dxy_ret_1d": -0.03,
        "corr_btc_sp500": 0.8,
        "corr_btc_dxy": -0.8,
    }
    adjusted, _ = _apply_context_adjustments(pred, q_data, settings)
    assert 0.0 <= adjusted.prob_up <= 1.0


def test_macro_event_window_active() -> None:
    settings = Settings(
        macro_event_crisis_enabled=True,
        macro_event_crisis_windows_utc="13:20-14:10",
        macro_event_crisis_weekdays="0,1,2,3,4",
    )
    # Wednesday 13:30 UTC
    now = datetime(2026, 2, 18, 13, 30, tzinfo=UTC)
    assert _is_macro_event_window(settings, now) is True


def test_macro_event_window_inactive_outside_time() -> None:
    settings = Settings(
        macro_event_crisis_enabled=True,
        macro_event_crisis_windows_utc="13:20-14:10",
        macro_event_crisis_weekdays="0,1,2,3,4",
    )
    # Wednesday 15:00 UTC
    now = datetime(2026, 2, 18, 15, 0, tzinfo=UTC)
    assert _is_macro_event_window(settings, now) is False
