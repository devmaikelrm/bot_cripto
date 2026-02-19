"""Advanced Performance Evaluator for Quantitative Trading."""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any

@dataclass
class PerformanceMetrics:
    total_pnl: float
    roi: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_gain: float
    avg_loss: float
    total_trades: int
    benchmark_comparison: float # vs Buy & Hold

class QuantEvaluator:
    """Calcula métricas profesionales de trading ajustadas por riesgo."""

    def __init__(self, initial_equity: float = 100.0, risk_free_rate: float = 0.02):
        self.initial_equity = initial_equity
        self.rf_rate = risk_free_rate

    def calculate_metrics(self, trades_df: pd.DataFrame, price_history: pd.Series) -> PerformanceMetrics:
        """
        Calcula el set completo de métricas a partir de un historial de trades.
        trades_df columns: ['ts', 'pnl', 'action', 'price']
        """
        if trades_df.empty:
            return None

        # 1. Profitabilidad Básica
        total_pnl = trades_df['pnl'].sum()
        roi = total_pnl / self.initial_equity
        
        # 2. Curva de Equity y Returns
        # Simulamos la curva temporal para ratios de riesgo
        equity_curve = self.initial_equity + trades_df['pnl'].cumsum()
        returns = equity_curve.pct_change().dropna()

        # 3. Sharpe & Sortino (Anualizados para Cripto 24/7)
        # 288 velas de 5m al día * 365 días = 105,120 periodos
        # Para 1h: 24 * 365 = 8760
        ann_factor = 8760 # Asumimos 1h por defecto, ajustar según timeframe
        
        std = returns.std()
        sharpe = (returns.mean() * ann_factor - self.rf_rate) / (std * np.sqrt(ann_factor)) if std > 0 else 0
        
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        sortino = (returns.mean() * ann_factor - self.rf_rate) / (downside_std * np.sqrt(ann_factor)) if downside_std > 0 else 0

        # 4. Max Drawdown
        rolling_max = equity_curve.cummax()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        max_dd = drawdowns.min()

        # 5. Eficiencia
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] < 0]
        win_rate = len(wins) / len(trades_df)
        
        gross_profit = wins['pnl'].sum()
        gross_loss = abs(losses['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # 6. Benchmark Comparison (Buy & Hold)
        start_price = price_history.iloc[0]
        end_price = price_history.iloc[-1]
        bh_return = (end_price - start_price) / start_price

        return PerformanceMetrics(
            total_pnl=float(total_pnl),
            roi=float(roi),
            annualized_return=float(returns.mean() * ann_factor),
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            max_drawdown=float(max_dd),
            win_rate=float(win_rate),
            profit_factor=float(profit_factor),
            avg_gain=float(wins['pnl'].mean() if not wins.empty else 0),
            avg_loss=float(losses['pnl'].mean() if not losses.empty else 0),
            total_trades=len(trades_df),
            benchmark_comparison=float(roi - bh_return)
        )

    def generate_markdown_report(self, metrics: PerformanceMetrics, regime_metrics: dict[str, PerformanceMetrics]) -> str:
        report = f"""
# 📈 Reporte de Evaluación Quant - Bot Cripto

## 📊 Métricas Globales
| Métrica | Valor |
| :--- | :--- |
| **PnL Total** | ${metrics.total_pnl:.2f} |
| **ROI** | {metrics.roi:.2%} |
| **Sharpe Ratio** | {metrics.sharpe_ratio:.2f} |
| **Sortino Ratio** | {metrics.sortino_ratio:.2f} |
| **Max Drawdown** | {metrics.max_drawdown:.2%} |
| **Win Rate** | {metrics.win_rate:.2%} |
| **Profit Factor** | {metrics.profit_factor:.2f} |
| **Vs Buy & Hold** | {metrics.benchmark_comparison:+.2%} |

## 🔍 Análisis por Régimen de Mercado
| Régimen | Win Rate | PnL | Sharpe |
| :--- | :--- | :--- | :--- |
"""
        for regime, m in regime_metrics.items():
            report += f"| {regime} | {m.win_rate:.1%} | ${m.total_pnl:.2f} | {m.sharpe_ratio:.2f} |
"
            
        return report
