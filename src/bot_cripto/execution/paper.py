"""Paper trading executor."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from bot_cripto.core.config import Settings, get_settings
from bot_cripto.core.logging import get_logger
from bot_cripto.decision.engine import Action, TradeSignal
from bot_cripto.models.base import PredictionOutput
from bot_cripto.monitoring.performance_store import PerformancePoint, PerformanceStore
from bot_cripto.monitoring.watchtower_store import WatchtowerStore
from bot_cripto.risk.state_store import RiskStateStore

logger = get_logger("execution.paper")


@dataclass
class Position:
    symbol: str
    entry_price: float
    qty: float
    opened_at: str
    stop_loss: float | None = None
    take_profit: float | None = None


@dataclass
class TradeRecord:
    ts: str
    symbol: str
    action: str
    price: float
    qty: float
    fee: float
    pnl: float


class PaperExecutor:
    """Executes paper trades and tracks realized PnL."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.state_path = self.settings.logs_dir / "paper_state.json"
        self.positions: dict[str, Position] = {}
        self.trades: list[TradeRecord] = []
        self.realized_pnl = 0.0
        self.risk_state_store = RiskStateStore(self.settings.logs_dir / "paper_risk_state.json")
        self.performance_store = PerformanceStore(
            self.settings.logs_dir / "performance_history.json"
        )
        self.watchtower = WatchtowerStore(self.settings.watchtower_db_path)
        self.risk_state = self.risk_state_store.load(initial_equity=self.settings.initial_equity)
        self._load_state()

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        try:
            self.realized_pnl = float(raw.get("realized_pnl", 0.0))

            positions_raw = raw.get("positions", {}) or {}
            positions: dict[str, Position] = {}
            if isinstance(positions_raw, dict):
                for sym, p in positions_raw.items():
                    if not isinstance(p, dict):
                        continue
                    positions[str(sym)] = Position(
                        symbol=str(p.get("symbol", sym)),
                        entry_price=float(p.get("entry_price", 0.0)),
                        qty=float(p.get("qty", 0.0)),
                        opened_at=str(p.get("opened_at", "")),
                        stop_loss=float(p["stop_loss"]) if p.get("stop_loss") is not None else None,
                        take_profit=float(p["take_profit"]) if p.get("take_profit") is not None else None,
                    )
            self.positions = positions

            trades_raw = raw.get("trades", []) or []
            trades: list[TradeRecord] = []
            if isinstance(trades_raw, list):
                for t in trades_raw[-5000:]:
                    if not isinstance(t, dict):
                        continue
                    trades.append(
                        TradeRecord(
                            ts=str(t.get("ts", "")),
                            symbol=str(t.get("symbol", "")),
                            action=str(t.get("action", "")),
                            price=float(t.get("price", 0.0)),
                            qty=float(t.get("qty", 0.0)),
                            fee=float(t.get("fee", 0.0)),
                            pnl=float(t.get("pnl", 0.0)),
                        )
                    )
            self.trades = trades

            # Align equity tracking with persisted realized PnL.
            self._persist_equity()
            logger.info(
                "paper_state_loaded",
                positions=len(self.positions),
                trades=len(self.trades),
                realized_pnl=float(self.realized_pnl),
            )
        except Exception:
            return

    def _persist_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": datetime.now(tz=UTC).isoformat(),
            "realized_pnl": float(self.realized_pnl),
            "positions": {
                sym: {
                    "symbol": p.symbol,
                    "entry_price": float(p.entry_price),
                    "qty": float(p.qty),
                    "opened_at": p.opened_at,
                    "stop_loss": float(p.stop_loss) if p.stop_loss is not None else None,
                    "take_profit": float(p.take_profit) if p.take_profit is not None else None,
                }
                for sym, p in self.positions.items()
            },
            "trades": [
                {
                    "ts": t.ts,
                    "symbol": t.symbol,
                    "action": t.action,
                    "price": float(t.price),
                    "qty": float(t.qty),
                    "fee": float(t.fee),
                    "pnl": float(t.pnl),
                }
                for t in self.trades[-5000:]
            ],
        }
        tmp = self.state_path.with_name(self.state_path.name + f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def _fee(self, notional: float) -> float:
        return notional * self.settings.fees_decimal

    def _fill_price(self, side: Literal["BUY", "SELL"], reference_price: float) -> float:
        cost_bps = self.settings.execution_cost_bps
        if side == "BUY":
            return reference_price * (1 + cost_bps / 10_000)
        return reference_price * (1 - cost_bps / 10_000)

    def _persist_equity(self) -> None:
        self.risk_state.equity = self.settings.initial_equity + self.realized_pnl
        self.risk_state_store.save(self.risk_state)
        self.watchtower.log_equity(
            ts=datetime.now(tz=UTC).isoformat(),
            equity=float(self.risk_state.equity),
            source="paper",
        )

    def _close_position(
        self,
        symbol: str,
        price: float,
        action: str,
        ts: str,
    ) -> TradeRecord | None:
        current = self.positions.get(symbol)
        if current is None:
            return None

        qty_to_sell = current.qty
        fill_price = self._fill_price("SELL", price)
        notional = fill_price * qty_to_sell
        fee = self._fee(notional)
        gross = (fill_price - current.entry_price) * qty_to_sell
        pnl = gross - fee
        self.realized_pnl += pnl
        rec = TradeRecord(
            ts=ts,
            symbol=symbol,
            action=action,
            price=fill_price,
            qty=qty_to_sell,
            fee=fee,
            pnl=pnl,
        )
        self.trades.append(rec)
        del self.positions[symbol]
        self._persist_equity()
        self._persist_state()
        trade_return = pnl / self.settings.initial_equity
        self.performance_store.append(PerformancePoint(ts=ts, metric=float(trade_return)))
        logger.info("paper_position_closed", symbol=symbol, action=action, pnl=pnl, fee=fee)
        return rec

    def _levels_from_prediction(
        self,
        entry_price: float,
        prediction: PredictionOutput | None,
    ) -> tuple[float | None, float | None]:
        if prediction is None:
            return None, None

        stop_loss = entry_price * (1 + prediction.p10 - self.settings.stop_loss_buffer)
        take_profit = entry_price * (1 + prediction.p90 + self.settings.take_profit_buffer)

        if stop_loss >= entry_price:
            stop_loss = entry_price * 0.995
        if take_profit <= entry_price:
            take_profit = entry_price * 1.005

        return stop_loss, take_profit

    def on_signal(
        self,
        symbol: str,
        signal: TradeSignal,
        price: float,
        qty: float = 1.0,
        prediction: PredictionOutput | None = None,
    ) -> TradeRecord | None:
        ts = datetime.now(tz=UTC).isoformat()
        current = self.positions.get(symbol)
        if current is not None:
            if current.stop_loss is not None and price <= current.stop_loss:
                return self._close_position(symbol, price, "SELL_SL", ts)
            if current.take_profit is not None and price >= current.take_profit:
                return self._close_position(symbol, price, "SELL_TP", ts)

        if signal.action == Action.HOLD:
            logger.info("paper_hold", symbol=symbol, reason=signal.reason)
            return None

        if signal.action == Action.BUY:
            if current is not None:
                logger.info("paper_buy_ignored_open_position", symbol=symbol)
                return None
            fill_price = self._fill_price("BUY", price)
            notional = fill_price * qty
            fee = self._fee(notional)
            stop_loss, take_profit = self._levels_from_prediction(fill_price, prediction)
            self.positions[symbol] = Position(
                symbol=symbol,
                entry_price=fill_price,
                qty=qty,
                opened_at=ts,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            rec = TradeRecord(
                ts=ts,
                symbol=symbol,
                action="BUY",
                price=fill_price,
                qty=qty,
                fee=fee,
                pnl=-fee,
            )
            self.realized_pnl -= fee
            self._persist_equity()
            self.trades.append(rec)
            self._persist_state()
            logger.info("paper_buy_executed", symbol=symbol, price=fill_price, qty=qty, fee=fee)
            return rec

        if signal.action == Action.SELL:
            if current is None:
                logger.info("paper_sell_ignored_no_position", symbol=symbol)
                return None
            return self._close_position(symbol, price, "SELL", ts)

        logger.warning("paper_unknown_action", action=str(signal.action))
        return None

    def report(self) -> dict[str, float | int]:
        return {
            "open_positions": len(self.positions),
            "trades": len(self.trades),
            "realized_pnl": float(self.realized_pnl),
            "equity": float(self.settings.initial_equity + self.realized_pnl),
        }

    def save_report(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "ts": t.ts,
                "symbol": t.symbol,
                "action": t.action,
                "price": t.price,
                "qty": t.qty,
                "fee": t.fee,
                "pnl": t.pnl,
            }
            for t in self.trades
        ]
        pd.DataFrame(data).to_csv(output_path, index=False)
        logger.info("paper_report_saved", path=str(output_path), rows=len(data))
        return output_path
