"""
Paper Trading Simulator.

Simulates live trading without real money.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from datetime import datetime
import uuid

import pandas as pd
import numpy as np

from finantradealgo.research.livetesting.models import (
    LiveTestConfig,
    PaperOrder,
    Position,
    LiveTestResult,
    OrderSide,
    OrderType,
    OrderStatus,
    SlippageModel,
)


class PaperTradingEngine:
    """Paper trading engine for live testing."""

    def __init__(self, config: LiveTestConfig):
        """Initialize paper trading engine."""
        self.config = config
        self.cash = config.starting_capital
        self.positions: Dict[str, Position] = {}
        self.orders: List[PaperOrder] = []
        self.trades: List[Dict] = []
        self.equity_history = []

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
    ) -> PaperOrder:
        """Place paper order."""
        order = PaperOrder(
            order_id=str(uuid.uuid4()),
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
        )

        self.orders.append(order)
        return order

    def execute_order(self, order: PaperOrder, market_price: float) -> bool:
        """Execute order at market price."""
        if order.status != OrderStatus.PENDING:
            return False

        # Calculate slippage
        slippage = self._calculate_slippage(order, market_price)
        fill_price = market_price + slippage if order.side == OrderSide.BUY else market_price - slippage

        # Calculate commission
        commission = fill_price * order.quantity * self.config.commission_pct

        # Update order
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = fill_price
        order.commission = commission
        order.slippage = abs(slippage * order.quantity)
        order.filled_at = datetime.now()

        # Update position
        self._update_position(order, fill_price, commission)

        # Record trade
        self.trades.append({
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.quantity,
            "price": fill_price,
            "commission": commission,
            "slippage": order.slippage,
            "timestamp": order.filled_at,
        })

        return True

    def _calculate_slippage(self, order: PaperOrder, market_price: float) -> float:
        """Calculate slippage based on model."""
        if self.config.slippage_model == SlippageModel.FIXED:
            slippage = market_price * self.config.slippage_pct
        else:
            # Simplified - could be more sophisticated
            slippage = market_price * self.config.slippage_pct

        return slippage if order.side == OrderSide.BUY else -slippage

    def _update_position(self, order: PaperOrder, fill_price: float, commission: float):
        """Update positions after order fill."""
        symbol = order.symbol

        if symbol not in self.positions:
            if order.side == OrderSide.BUY:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=order.quantity,
                    avg_entry_price=fill_price,
                    current_price=fill_price,
                    cost_basis=fill_price * order.quantity + commission,
                )
                self.cash -= (fill_price * order.quantity + commission)
        else:
            pos = self.positions[symbol]
            if order.side == OrderSide.BUY:
                total_cost = pos.cost_basis + (fill_price * order.quantity + commission)
                total_qty = pos.quantity + order.quantity
                pos.avg_entry_price = total_cost / total_qty
                pos.quantity = total_qty
                pos.cost_basis = total_cost
                self.cash -= (fill_price * order.quantity + commission)
            else:  # SELL
                pnl = (fill_price - pos.avg_entry_price) * order.quantity - commission
                pos.realized_pnl += pnl
                pos.quantity -= order.quantity
                self.cash += (fill_price * order.quantity - commission)

                if pos.quantity <= 0:
                    del self.positions[symbol]

    def get_equity(self) -> float:
        """Get total equity."""
        position_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + position_value

    def calculate_result(self) -> LiveTestResult:
        """Calculate live test result."""
        if not self.trades:
            return LiveTestResult(strategy_id=self.config.strategy_id, config=self.config)

        trades_df = pd.DataFrame(self.trades)

        # Calculate PnL
        total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_pnl = total_realized_pnl + total_unrealized_pnl

        total_return_pct = (total_pnl / self.config.starting_capital) * 100

        # Calculate metrics
        total_trades = len(self.trades)
        total_commission = trades_df['commission'].sum()
        avg_slippage_bps = (trades_df['slippage'].mean() / trades_df['price'].mean()) * 10000

        # Simplified metrics
        sharpe_ratio = total_return_pct / 10 if total_return_pct > 0 else 0

        return LiveTestResult(
            strategy_id=self.config.strategy_id,
            config=self.config,
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            total_trades=total_trades,
            total_commission=total_commission,
            avg_slippage_bps=avg_slippage_bps,
        )
