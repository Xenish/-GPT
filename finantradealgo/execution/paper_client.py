from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from finantradealgo.core.portfolio import Portfolio, Position
from .client_base import ExecutionClientBase


class PaperExecutionClient(ExecutionClientBase):
    """
    Minimal execution adapter that mimics live trading using an in-memory
    portfolio. It accepts signals from the live engine and records fills
    plus equity curve snapshots to CSV for quick inspection.
    """

    def __init__(
        self,
        *,
        initial_cash: float = 10_000.0,
        fee_pct: float = 0.0004,
        slippage_pct: float = 0.0005,
        output_dir: str = "outputs/live_paper",
        state_path: Optional[str] = None,
        symbol: str = "UNKNOWN",
    ):
        self.portfolio = Portfolio(
            initial_cash=initial_cash,
            cash=initial_cash,
            equity=initial_cash,
        )
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.closed_trades: List[Dict[str, Any]] = []
        self.symbol = symbol
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._last_price: Optional[float] = None
        self._last_timestamp: Optional[pd.Timestamp] = None
        self._active_trade: Optional[Dict[str, Any]] = None
        self.state_path = (
            Path(state_path) if state_path else self.output_dir / "paper_state.json"
        )

    # -----------------
    # Portfolio helpers
    # -----------------
    def has_position(self) -> bool:
        return self.portfolio.position is not None

    def get_position(self) -> Optional[Position]:
        return self.portfolio.position

    def mark_to_market(self, price: float, timestamp) -> None:
        self._last_price = price
        ts = pd.to_datetime(timestamp)
        self._last_timestamp = ts
        self.portfolio.record(ts, price)

    def get_portfolio(self) -> Dict[str, float]:
        if self._last_price is not None:
            self.portfolio.update_equity(self._last_price)
        return {
            "cash": self.portfolio.cash,
            "equity": self.portfolio.equity,
            "position": self.portfolio.position,
        }

    def get_trade_log(self) -> List[Dict]:
        return list(self.closed_trades)

    def get_realized_pnl(self) -> float:
        pnl_values = [t.get("pnl", 0.0) for t in self.closed_trades if t.get("pnl") is not None]
        return float(sum(pnl_values)) if pnl_values else 0.0

    def get_unrealized_pnl(self) -> float:
        if self.portfolio.position is None or self._last_price is None:
            return 0.0
        position = self.portfolio.position
        if position.side == "LONG":
            return (self._last_price - position.entry_price) * position.qty
        if position.side == "SHORT":
            return (position.entry_price - self._last_price) * position.qty
        return 0.0

    def get_open_positions(self) -> List[Dict[str, Any]]:
        if self.portfolio.position is None:
            return []
        pos = self.portfolio.position
        open_trade = self._active_trade or {}
        entry_time = open_trade.get("timestamp_entry")
        return [
            {
                "id": open_trade.get("trade_id", f"{self.symbol}_{pos.side.lower()}"),
                "symbol": self.symbol,
                "side": pos.side,
                "qty": pos.qty,
                "entry_price": pos.entry_price,
                "current_price": self._last_price,
                "pnl": self.get_unrealized_pnl(),
                "entry_time": entry_time.isoformat() if isinstance(entry_time, pd.Timestamp) else None,
            }
        ]

    def close_position_market(self, pos: Dict[str, Any]) -> Optional[Dict]:
        price = self._last_price or pos.get("current_price") or pos.get("entry_price")
        ts = self._last_timestamp or pd.Timestamp.utcnow()
        if price is None:
            return None
        return self._close_position(price=price, timestamp=ts, reason="MANUAL_FLATTEN")

    # --------------
    # Order routing
    # --------------
    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        *,
        price: Optional[float] = None,
        reduce_only: bool = False,
        client_order_id: Optional[str] = None,
        **_: Any,
    ) -> Optional[Dict]:
        side = side.upper()
        order_type = order_type.upper()
        if order_type != "MARKET":
            raise ValueError("Paper client only supports MARKET orders.")

        trade_price = price if price is not None else self._last_price
        if trade_price is None:
            raise ValueError("Price required for paper fills.")
        timestamp = self._last_timestamp or pd.Timestamp.utcnow()

        if reduce_only:
            trade = self._close_position(
                price=trade_price,
                timestamp=timestamp,
                reason="REDUCE_ONLY",
            )
            if trade is not None:
                trade["client_order_id"] = client_order_id
            return trade

        current = self.portfolio.position
        if current is not None:
            if current.side == "LONG" and side == "SELL":
                return self._close_position(
                    price=trade_price,
                    timestamp=timestamp,
                    reason="OPPOSITE_SIGNAL",
                )
            if current.side == "SHORT" and side == "BUY":
                return self._close_position(
                    price=trade_price,
                    timestamp=timestamp,
                    reason="OPPOSITE_SIGNAL",
                )

        if qty is None or qty <= 0:
            return None
        mapped_side = "LONG" if side == "BUY" else "SHORT"
        trade = self._open_position(
            side=mapped_side,
            price=trade_price,
            size=qty,
            timestamp=timestamp,
        )
        if trade is not None:
            trade["client_order_id"] = client_order_id
        return trade

    def cancel_order(self, symbol: str, order_id: str | int) -> None:
        return None

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        return []

    def _open_position(self, *, side: str, price: float, size: float, timestamp) -> Optional[Dict]:
        if self.portfolio.position is not None:
            return None

        if side == "LONG":
            entry_price = price * (1 + self.slippage_pct)
        else:
            entry_price = price * (1 - self.slippage_pct)

        qty = size
        notional = entry_price * qty
        fee = notional * self.fee_pct
        total_cost = notional + fee
        if total_cost > self.portfolio.cash:
            affordable = self.portfolio.cash / (1 + self.fee_pct)
            qty = affordable / entry_price
            if qty <= 0:
                return None
            notional = entry_price * qty
            fee = notional * self.fee_pct
            total_cost = notional + fee
            if total_cost > self.portfolio.cash:
                return None

        self.portfolio.cash -= total_cost
        self.portfolio.position = Position(side=side, qty=qty, entry_price=entry_price)
        self.portfolio.update_equity(entry_price)
        self._last_price = entry_price
        ts_entry = pd.to_datetime(timestamp)
        trade_id = f"{self.symbol}_{side.lower()}_{int(ts_entry.timestamp())}"
        self._active_trade = {
            "trade_id": trade_id,
            "timestamp_entry": ts_entry,
            "side": side,
            "qty": qty,
            "entry_price": entry_price,
        }

        return dict(self._active_trade)

    def _close_position(self, *, price: float, timestamp, reason: str) -> Optional[Dict]:
        if self.portfolio.position is None:
            return None

        position = self.portfolio.position
        qty = position.qty
        if position.side == "LONG":
            exit_price = price * (1 - self.slippage_pct)
        else:
            exit_price = price * (1 + self.slippage_pct)

        notional = exit_price * qty
        commission_close = notional * self.fee_pct
        self.portfolio.cash += notional - commission_close

        if position.side == "LONG":
            gross_pnl = (exit_price - position.entry_price) * qty
        else:
            gross_pnl = (position.entry_price - exit_price) * qty

        commission_open = position.entry_price * qty * self.fee_pct
        net_pnl = gross_pnl - commission_open - commission_close

        trade = {
            "side": position.side,
            "qty": qty,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "timestamp_entry": self._active_trade.get("timestamp_entry") if self._active_trade else None,
            "timestamp_exit": pd.to_datetime(timestamp),
            "pnl": net_pnl,
            "reason": reason,
        }
        self.closed_trades.append(trade)
        self._active_trade = None

        self.portfolio.position = None
        self.portfolio.update_equity(exit_price)
        self._last_price = exit_price
        return trade

    # --------------------
    # Persistence helpers
    # --------------------
    def export_logs(self, *, timeframe: str) -> Dict[str, Path]:
        equity_path = self.output_dir / f"live_paper_equity_{timeframe}.csv"
        trades_path = self.output_dir / f"live_paper_trades_{timeframe}.csv"

        equity_series = pd.Series(
            data=self.portfolio.equity_curve,
            index=self.portfolio.timestamps,
            name="equity",
        )
        equity_series.to_csv(equity_path, header=True)

        trades_df = pd.DataFrame(self.closed_trades)
        trades_df.to_csv(trades_path, index=False)

        return {"equity": equity_path, "trades": trades_path}

    def to_state_dict(self) -> Dict[str, Any]:
        open_positions = self.get_open_positions()
        return {
            "cash": self.portfolio.cash,
            "equity": self.portfolio.equity,
            "realized_pnl": self.get_realized_pnl(),
            "unrealized_pnl": self.get_unrealized_pnl(),
            "open_positions": open_positions,
            "open_trade": self._serialize_trade(self._active_trade),
            "closed_trades": [self._serialize_trade(t) for t in self.closed_trades],
            "last_bar_timestamp": self._last_timestamp.isoformat()
            if self._last_timestamp
            else None,
        }

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        self.portfolio.cash = state.get("cash", self.portfolio.cash)
        self.portfolio.equity = state.get("equity", self.portfolio.equity)

        open_trade = state.get("open_trade")
        self._active_trade = self._deserialize_trade(open_trade)

        closed_trades = state.get("closed_trades", [])
        self.closed_trades = [
            self._deserialize_trade(trade) for trade in closed_trades if trade is not None
        ]

        last_ts = state.get("last_bar_timestamp")
        self._last_timestamp = pd.to_datetime(last_ts) if last_ts else None

    def save_state(self, path: Optional[str | Path] = None) -> Path:
        target = Path(path) if path else self.state_path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_state_dict()
        with target.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        return target

    def load_state(self, path: Optional[str | Path] = None) -> None:
        target = Path(path) if path else self.state_path
        if not target.exists():
            return
        with target.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        self.from_state_dict(payload)

    @staticmethod
    def _serialize_trade(trade: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if trade is None:
            return None
        serialized = dict(trade)
        for key in ("timestamp_entry", "timestamp_exit"):
            if key in serialized and serialized[key] is not None:
                serialized[key] = pd.to_datetime(serialized[key]).isoformat()
        return serialized

    @staticmethod
    def _deserialize_trade(trade: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if trade is None:
            return None
        deserialized = dict(trade)
        for key in ("timestamp_entry", "timestamp_exit"):
            if key in deserialized and deserialized[key] is not None:
                deserialized[key] = pd.to_datetime(deserialized[key])
        return deserialized
