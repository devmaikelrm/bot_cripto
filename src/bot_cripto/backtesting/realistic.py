"""Realistic backtesting with dynamic costs, partial fills, and latency."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from bot_cripto.core.logging import get_logger

logger = get_logger("backtesting.realistic")


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Trade:
    """Detailed record of a single round-trip trade."""

    entry_idx: int
    exit_idx: int
    side: str
    entry_price: float
    exit_price: float
    volume_at_entry: float
    volume_at_exit: float
    intended_qty: float
    filled_qty: float
    fill_ratio: float
    maker_fee: float
    taker_fee: float
    total_fee: float
    slippage_entry: float
    slippage_exit: float
    latency_bars: int
    gross_pnl: float
    net_pnl: float
    net_return: float
    duration_bars: int


@dataclass(frozen=True)
class RealisticBacktestReport:
    """Summary of a realistic backtest run."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_gross_pnl: float
    total_net_pnl: float
    total_fees: float
    total_slippage_cost: float
    avg_fill_ratio: float
    sharpe: float
    max_drawdown: float
    profit_factor: float
    avg_trade_duration: float
    net_return_pct: float
    trades: list[Trade] = field(default_factory=list)


class CostModel:
    """Realistic transaction cost model.

    Parameters
    ----------
    maker_fee_bps : float
        Maker fee in basis points (e.g. 2.0 for Binance VIP0 maker).
    taker_fee_bps : float
        Taker fee in basis points (e.g. 4.0 for Binance VIP0 taker).
    base_slippage_bps : float
        Minimum slippage even with infinite liquidity.
    volume_impact_factor : float
        Controls how much slippage increases when trade size is large
        relative to bar volume.  slippage = base + factor * sqrt(qty / volume).
    latency_bars : int
        Simulated execution delay in bars (0 = fill at signal bar,
        1 = fill at next bar open, etc.).
    max_fill_ratio : float
        Maximum fraction of bar volume that can be filled (0.0-1.0).
        Prevents unrealistic fills on low-volume bars.
    """

    def __init__(
        self,
        maker_fee_bps: float = 2.0,
        taker_fee_bps: float = 4.0,
        base_slippage_bps: float = 1.0,
        volume_impact_factor: float = 0.1,
        latency_bars: int = 1,
        max_fill_ratio: float = 0.10,
    ) -> None:
        self.maker_fee_bps = maker_fee_bps
        self.taker_fee_bps = taker_fee_bps
        self.base_slippage_bps = base_slippage_bps
        self.volume_impact_factor = volume_impact_factor
        self.latency_bars = latency_bars
        self.max_fill_ratio = max_fill_ratio

    def dynamic_slippage_bps(self, qty: float, bar_volume: float) -> float:
        """Slippage increases with trade size relative to bar volume."""
        if bar_volume <= 0:
            return self.base_slippage_bps * 100.0  # extreme penalty for no liquidity
        ratio = qty / bar_volume
        return self.base_slippage_bps + self.volume_impact_factor * math.sqrt(ratio) * 10_000

    def compute_fill(self, intended_qty: float, bar_volume: float) -> float:
        """How much of the intended quantity can actually be filled."""
        if bar_volume <= 0:
            return 0.0
        max_fill = bar_volume * self.max_fill_ratio
        return min(intended_qty, max_fill)

    def entry_cost(
        self, price: float, qty: float, bar_volume: float, is_maker: bool = False
    ) -> tuple[float, float, float]:
        """Return (fill_price, fee, slippage_cost) for entry.

        Entry buys at a worse (higher) price due to slippage.
        """
        slip_bps = self.dynamic_slippage_bps(qty, bar_volume)
        slip_frac = slip_bps / 10_000
        fill_price = price * (1.0 + slip_frac)
        fee_bps = self.maker_fee_bps if is_maker else self.taker_fee_bps
        fee = fill_price * qty * fee_bps / 10_000
        slippage_cost = price * qty * slip_frac
        return fill_price, fee, slippage_cost

    def exit_cost(
        self, price: float, qty: float, bar_volume: float, is_maker: bool = False
    ) -> tuple[float, float, float]:
        """Return (fill_price, fee, slippage_cost) for exit.

        Exit sells at a worse (lower) price due to slippage.
        """
        slip_bps = self.dynamic_slippage_bps(qty, bar_volume)
        slip_frac = slip_bps / 10_000
        fill_price = price * (1.0 - slip_frac)
        fee_bps = self.maker_fee_bps if is_maker else self.taker_fee_bps
        fee = fill_price * qty * fee_bps / 10_000
        slippage_cost = price * qty * slip_frac
        return fill_price, fee, slippage_cost


class RealisticBacktester:
    """Backtester with dynamic slippage, partial fills, and latency.

    Runs a signal-based backtest over a DataFrame with OHLCV + signal columns.
    Expects columns: open, high, low, close, volume, plus a signal column
    with values in {1, 0, -1} (long / flat / short).

    Parameters
    ----------
    cost_model : CostModel
        Transaction cost configuration.
    initial_equity : float
        Starting capital.
    position_size_frac : float
        Fraction of equity to risk per trade (0.0-1.0).
    signal_col : str
        Name of the column containing trade signals.
    """

    def __init__(
        self,
        cost_model: CostModel | None = None,
        initial_equity: float = 10_000.0,
        position_size_frac: float = 0.02,
        signal_col: str = "signal",
    ) -> None:
        self.cost = cost_model or CostModel()
        self.initial_equity = initial_equity
        self.position_size_frac = position_size_frac
        self.signal_col = signal_col

    def run(self, df: pd.DataFrame) -> RealisticBacktestReport:
        """Execute backtest on DataFrame with OHLCV + signal column.

        The signal column should contain:
          1  = go long / stay long
          0  = flat / close position
         -1  = go short / stay short
        """
        required = {"open", "high", "low", "close", "volume", self.signal_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        closes = df["close"].values
        opens = df["open"].values
        volumes = df["volume"].values
        signals = df[self.signal_col].values
        n = len(df)

        trades: list[Trade] = []
        equity = self.initial_equity

        # Position state
        in_position = False
        position_side = Side.LONG
        entry_idx = 0
        entry_fill_price = 0.0
        entry_volume = 0.0
        entry_intended_qty = 0.0
        filled_qty = 0.0
        entry_fee = 0.0
        entry_slippage = 0.0

        latency = self.cost.latency_bars

        for i in range(1, n):
            sig = int(signals[i])

            if in_position:
                # Check if signal flips or goes flat -> close position
                should_close = False
                if position_side == Side.LONG and sig <= 0:
                    should_close = True
                elif position_side == Side.SHORT and sig >= 0:
                    should_close = True

                if should_close:
                    # Exit at the bar affected by latency
                    exit_bar = min(i + latency, n - 1)
                    exit_ref_price = opens[exit_bar] if exit_bar > i else closes[i]
                    exit_volume = float(volumes[exit_bar] if exit_bar < n else volumes[i])

                    exit_fill_price, exit_fee, exit_slippage = self.cost.exit_cost(
                        exit_ref_price, filled_qty, exit_volume
                    )

                    if position_side == Side.LONG:
                        gross_pnl = (exit_fill_price - entry_fill_price) * filled_qty
                    else:
                        gross_pnl = (entry_fill_price - exit_fill_price) * filled_qty

                    total_fee = entry_fee + exit_fee
                    net_pnl = gross_pnl - total_fee
                    notional = entry_fill_price * filled_qty
                    net_return = net_pnl / notional if notional > 0 else 0.0

                    trades.append(
                        Trade(
                            entry_idx=entry_idx,
                            exit_idx=exit_bar,
                            side=position_side.value,
                            entry_price=entry_fill_price,
                            exit_price=exit_fill_price,
                            volume_at_entry=entry_volume,
                            volume_at_exit=exit_volume,
                            intended_qty=entry_intended_qty,
                            filled_qty=filled_qty,
                            fill_ratio=filled_qty / entry_intended_qty if entry_intended_qty > 0 else 0.0,
                            maker_fee=0.0,
                            taker_fee=total_fee,
                            total_fee=total_fee,
                            slippage_entry=entry_slippage,
                            slippage_exit=exit_slippage,
                            latency_bars=latency,
                            gross_pnl=gross_pnl,
                            net_pnl=net_pnl,
                            net_return=net_return,
                            duration_bars=exit_bar - entry_idx,
                        )
                    )
                    equity += net_pnl
                    in_position = False

            if not in_position and sig != 0:
                # Open new position
                entry_bar = min(i + latency, n - 1)
                entry_ref_price = opens[entry_bar] if entry_bar > i else closes[i]
                entry_volume = float(volumes[entry_bar] if entry_bar < n else volumes[i])

                # Position sizing
                notional_target = equity * self.position_size_frac
                entry_intended_qty = notional_target / entry_ref_price if entry_ref_price > 0 else 0.0

                # Partial fill
                filled_qty = self.cost.compute_fill(entry_intended_qty, entry_volume)
                if filled_qty <= 0:
                    continue  # skip — no liquidity

                entry_fill_price, entry_fee, entry_slippage = self.cost.entry_cost(
                    entry_ref_price, filled_qty, entry_volume
                )
                entry_idx = entry_bar
                position_side = Side.LONG if sig > 0 else Side.SHORT
                in_position = True

        # Force close any open position at the end
        if in_position:
            exit_bar = n - 1
            exit_ref_price = closes[exit_bar]
            exit_volume = float(volumes[exit_bar])
            exit_fill_price, exit_fee, exit_slippage = self.cost.exit_cost(
                exit_ref_price, filled_qty, exit_volume
            )
            if position_side == Side.LONG:
                gross_pnl = (exit_fill_price - entry_fill_price) * filled_qty
            else:
                gross_pnl = (entry_fill_price - exit_fill_price) * filled_qty
            total_fee = entry_fee + exit_fee
            net_pnl = gross_pnl - total_fee
            notional = entry_fill_price * filled_qty
            net_return = net_pnl / notional if notional > 0 else 0.0
            trades.append(
                Trade(
                    entry_idx=entry_idx,
                    exit_idx=exit_bar,
                    side=position_side.value,
                    entry_price=entry_fill_price,
                    exit_price=exit_fill_price,
                    volume_at_entry=entry_volume,
                    volume_at_exit=exit_volume,
                    intended_qty=entry_intended_qty,
                    filled_qty=filled_qty,
                    fill_ratio=filled_qty / entry_intended_qty if entry_intended_qty > 0 else 0.0,
                    maker_fee=0.0,
                    taker_fee=total_fee,
                    total_fee=total_fee,
                    slippage_entry=entry_slippage,
                    slippage_exit=exit_slippage,
                    latency_bars=latency,
                    gross_pnl=gross_pnl,
                    net_pnl=net_pnl,
                    net_return=net_return,
                    duration_bars=exit_bar - entry_idx,
                )
            )
            equity += net_pnl

        return self._build_report(trades, self.initial_equity)

    @staticmethod
    def _build_report(trades: list[Trade], initial_equity: float = 10_000.0) -> RealisticBacktestReport:
        if not trades:
            return RealisticBacktestReport(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_gross_pnl=0.0,
                total_net_pnl=0.0,
                total_fees=0.0,
                total_slippage_cost=0.0,
                avg_fill_ratio=0.0,
                sharpe=0.0,
                max_drawdown=0.0,
                profit_factor=0.0,
                avg_trade_duration=0.0,
                net_return_pct=0.0,
            )

        net_returns = [t.net_return for t in trades]
        net_pnls = [t.net_pnl for t in trades]
        winners = [t for t in trades if t.net_pnl > 0]
        losers = [t for t in trades if t.net_pnl <= 0]

        total_fees = sum(t.total_fee for t in trades)
        total_slippage = sum(t.slippage_entry + t.slippage_exit for t in trades)
        total_gross = sum(t.gross_pnl for t in trades)
        total_net = sum(t.net_pnl for t in trades)

        # Sharpe — annualised assuming 365 calendar days (BTC trades 24/7)
        arr = np.array(net_returns)
        if len(arr) < 5:
            sharpe = 0.0
        else:
            std = float(np.std(arr, ddof=1))
            per_trade_sharpe = float(np.mean(arr)) / std if std > 0 else 0.0
            bar_span = max(1, trades[-1].exit_idx - trades[0].entry_idx)
            trades_per_year = 365.0 * len(trades) / bar_span
            sharpe = per_trade_sharpe * math.sqrt(trades_per_year)

        # Max drawdown — percentage of equity (not absolute dollars)
        equity_curve = initial_equity + np.cumsum(net_pnls)
        peak = np.maximum.accumulate(equity_curve)
        dd_pct = (equity_curve - peak) / np.where(peak > 0, peak, 1.0)
        max_dd = float(np.min(dd_pct)) if len(dd_pct) > 0 else 0.0

        # Profit factor
        gross_profit = sum(t.net_pnl for t in winners)
        gross_loss = abs(sum(t.net_pnl for t in losers))
        if gross_loss > 0:
            pf = gross_profit / gross_loss
        else:
            pf = float("inf") if gross_profit > 0 else 0.0

        # Fill ratio
        avg_fill = float(np.mean([t.fill_ratio for t in trades]))

        # Duration
        avg_dur = float(np.mean([t.duration_bars for t in trades]))

        # Net return % — expressed relative to starting equity, not first trade notional
        net_ret_pct = total_net / initial_equity * 100 if initial_equity > 0 else 0.0

        report = RealisticBacktestReport(
            total_trades=len(trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=len(winners) / len(trades),
            total_gross_pnl=total_gross,
            total_net_pnl=total_net,
            total_fees=total_fees,
            total_slippage_cost=total_slippage,
            avg_fill_ratio=avg_fill,
            sharpe=sharpe,
            max_drawdown=max_dd,
            profit_factor=pf,
            avg_trade_duration=avg_dur,
            net_return_pct=net_ret_pct,
            trades=trades,
        )
        logger.info(
            "realistic_backtest_completed",
            total_trades=report.total_trades,
            win_rate=report.win_rate,
            total_net_pnl=report.total_net_pnl,
            total_fees=report.total_fees,
            total_slippage=report.total_slippage_cost,
            avg_fill_ratio=report.avg_fill_ratio,
            sharpe=report.sharpe,
            max_drawdown=report.max_drawdown,
        )
        return report
