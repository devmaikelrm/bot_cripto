"""Compass Phase 1 KPI consolidation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from bot_cripto.backtesting.purged_cv import CPCVReport
from bot_cripto.backtesting.walk_forward import BacktestReport
from bot_cripto.models.base import PredictionOutput


@dataclass(frozen=True)
class Phase1KPIReport:
    symbol: str
    timeframe: str
    generated_at: str
    walk_forward_sharpe_oos: float
    in_sample_sharpe: float
    wf_efficiency: float
    cpcv_sharpe_mean: float
    cpcv_sharpe_p5: float
    cpcv_combinations_total: int
    pass_wf_efficiency: bool
    pass_cpcv_sharpe_mean: bool
    pass_cpcv_sharpe_p5: bool
    phase1_ready: bool


def _sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns, dtype=float)
    std = float(np.std(arr, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.mean(arr)) / std


def estimate_in_sample_sharpe(
    df: pd.DataFrame,
    model: object,
    train_size: int,
    target_col: str = "close",
    roundtrip_cost: float = 0.0,
) -> float:
    """Estimate IS Sharpe using one-step simulation inside training window."""
    if train_size < 3 or len(df) < train_size:
        return 0.0

    train_df = df.iloc[:train_size]
    if len(train_df) < 3:
        return 0.0

    # Model is pre-trained on the same IS window for direct IS/OOS ratio baseline.
    if hasattr(model, "train"):
        model.train(train_df, target_col=target_col)

    net_returns: list[float] = []
    for i in range(1, len(train_df)):
        window = train_df.iloc[:i]
        pred = model.predict(window)
        if not isinstance(pred, PredictionOutput):
            continue
        c0 = float(train_df[target_col].iloc[i - 1])
        c1 = float(train_df[target_col].iloc[i])
        realized = (c1 - c0) / c0 if c0 != 0 else 0.0
        expected_sign = 1 if pred.expected_return >= 0 else -1
        net_returns.append(expected_sign * realized - roundtrip_cost)
    return _sharpe(net_returns)


def build_phase1_kpi_report(
    *,
    symbol: str,
    timeframe: str,
    walk_forward_report: BacktestReport,
    cpcv_report: CPCVReport,
    in_sample_sharpe: float,
) -> Phase1KPIReport:
    oos_sharpe = float(walk_forward_report.sharpe)
    denom = float(in_sample_sharpe)
    wf_eff = 0.0 if abs(denom) < 1e-12 else (oos_sharpe / denom)

    pass_wf = 0.5 <= wf_eff <= 0.85
    pass_cpcv_mean = float(cpcv_report.sharpe_mean) > 0.5
    pass_cpcv_p5 = float(cpcv_report.sharpe_p5) > 0.0
    phase1_ready = pass_wf and pass_cpcv_mean and pass_cpcv_p5

    return Phase1KPIReport(
        symbol=symbol,
        timeframe=timeframe,
        generated_at=datetime.now(tz=UTC).isoformat(),
        walk_forward_sharpe_oos=oos_sharpe,
        in_sample_sharpe=denom,
        wf_efficiency=wf_eff,
        cpcv_sharpe_mean=float(cpcv_report.sharpe_mean),
        cpcv_sharpe_p5=float(cpcv_report.sharpe_p5),
        cpcv_combinations_total=int(cpcv_report.combinations_total),
        pass_wf_efficiency=pass_wf,
        pass_cpcv_sharpe_mean=pass_cpcv_mean,
        pass_cpcv_sharpe_p5=pass_cpcv_p5,
        phase1_ready=phase1_ready,
    )
