"""
Live Testing Models.

Data structures for paper trading and live testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Any

import pandas as pd


class OrderSide(str, Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    """Order execution status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class SlippageModel(str, Enum):
    """Slippage model type."""
    FIXED = "fixed"  # Fixed percentage
    VOLUME_BASED = "volume_based"  # Based on order size vs volume
    VOLATILITY_BASED = "volatility_based"  # Based on market volatility


@dataclass
class LiveTestConfig:
    """Live testing configuration."""

    strategy_id: str
    starting_capital: float = 10000.0

    # Execution costs
    commission_pct: float = 0.001  # 0.1% commission
    slippage_model: SlippageModel = SlippageModel.FIXED
    slippage_pct: float = 0.0005  # 0.05% slippage

    # Position sizing
    max_position_size_pct: float = 0.2  # Max 20% per position
    max_leverage: float = 1.0  # No leverage by default

    # Risk limits
    max_daily_loss_pct: float = 5.0  # Stop if lose 5% in a day
    max_drawdown_pct: float = 20.0  # Stop if DD exceeds 20%

    # Testing duration
    test_start_time: Optional[datetime] = None
    test_end_time: Optional[datetime] = None


@dataclass
class PaperOrder:
    """Paper trading order."""

    order_id: str
    strategy_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float

    # Pricing
    limit_price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None

    # Execution
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0

    # Costs
    commission: float = 0.0
    slippage: float = 0.0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "limit_price": self.limit_price,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "commission": self.commission,
            "slippage": self.slippage,
            "created_at": self.created_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
        }


@dataclass
class Position:
    """Current position."""

    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float

    # PnL
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    # Metrics
    cost_basis: float = 0.0
    market_value: float = 0.0

    def update_price(self, current_price: float):
        """Update position with current price."""
        self.current_price = current_price
        self.market_value = self.quantity * current_price
        self.unrealized_pnl = (current_price - self.avg_entry_price) * self.quantity


@dataclass
class LiveTestResult:
    """Live testing session result."""

    strategy_id: str
    config: LiveTestConfig

    # Performance
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0

    # Trading activity
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # Execution quality
    avg_slippage_bps: float = 0.0  # Basis points
    total_commission: float = 0.0

    # Risk metrics
    sharpe_ratio_live: float = 0.0
    max_dd_live: float = 0.0
    current_drawdown_pct: float = 0.0

    # Comparison to backtest
    expected_sharpe: Optional[float] = None
    expected_return: Optional[float] = None
    sharpe_deviation_pct: float = 0.0

    # Timestamps
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_hours: float = 0.0

    # Equity curve
    equity_curve: Optional[pd.Series] = None

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary."""
        return {
            "strategy_id": self.strategy_id,
            "total_pnl": round(self.total_pnl, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 3),
            "avg_slippage_bps": round(self.avg_slippage_bps, 2),
            "total_commission": round(self.total_commission, 2),
            "duration_hours": round(self.duration_hours, 1),
        }


@dataclass
class ProductionReadiness:
    """Production readiness assessment."""

    is_ready: bool
    overall_score: float  # 0-100

    # Checks
    passed_checks: List[str] = field(default_factory=list)
    failed_checks: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Requirements
    min_test_duration_hours: float = 24.0
    min_trades: int = 50
    min_sharpe: float = 0.5
    max_drawdown_threshold: float = 20.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_ready": self.is_ready,
            "overall_score": round(self.overall_score, 1),
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "warnings": self.warnings,
        }
