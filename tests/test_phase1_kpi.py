from __future__ import annotations

from bot_cripto.backtesting.phase1_kpi import build_phase1_kpi_report
from bot_cripto.backtesting.purged_cv import CPCVReport
from bot_cripto.backtesting.walk_forward import BacktestReport


def test_build_phase1_kpi_report_marks_ready_when_thresholds_pass() -> None:
    wf = BacktestReport(
        folds=4,
        accuracy=0.56,
        avg_return=0.001,
        total_return=0.12,
        avg_net_return=0.0008,
        total_net_return=0.09,
        sharpe=0.60,
    )
    cpcv = CPCVReport(
        n_groups=6,
        n_test_groups=2,
        combinations_total=15,
        purge_size=5,
        embargo_size=5,
        accuracy_mean=0.55,
        total_return_mean=0.03,
        total_net_return_mean=0.02,
        total_net_return_p5=0.001,
        sharpe_mean=0.72,
        sharpe_p5=0.12,
    )
    report = build_phase1_kpi_report(
        symbol="BTC/USDT",
        timeframe="5m",
        walk_forward_report=wf,
        cpcv_report=cpcv,
        in_sample_sharpe=0.9,
    )
    assert 0.5 <= report.wf_efficiency <= 0.85
    assert report.pass_cpcv_sharpe_mean is True
    assert report.pass_cpcv_sharpe_p5 is True
    assert report.phase1_ready is True


def test_build_phase1_kpi_report_not_ready_when_kpis_fail() -> None:
    wf = BacktestReport(
        folds=4,
        accuracy=0.50,
        avg_return=0.0,
        total_return=0.0,
        avg_net_return=-0.0002,
        total_net_return=-0.02,
        sharpe=0.20,
    )
    cpcv = CPCVReport(
        n_groups=6,
        n_test_groups=2,
        combinations_total=15,
        purge_size=5,
        embargo_size=5,
        accuracy_mean=0.50,
        total_return_mean=0.0,
        total_net_return_mean=-0.01,
        total_net_return_p5=-0.05,
        sharpe_mean=0.30,
        sharpe_p5=-0.10,
    )
    report = build_phase1_kpi_report(
        symbol="BTC/USDT",
        timeframe="1h",
        walk_forward_report=wf,
        cpcv_report=cpcv,
        in_sample_sharpe=1.0,
    )
    assert report.pass_wf_efficiency is False
    assert report.pass_cpcv_sharpe_mean is False
    assert report.pass_cpcv_sharpe_p5 is False
    assert report.phase1_ready is False
