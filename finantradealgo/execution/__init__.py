from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Protocol


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(str, Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class ExecutionClient(Protocol):
    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        **kwargs: Any,
    ) -> Any:
        ...


@dataclass
class SimulatedFill:
    price: float
    qty: float
    timestamp: datetime
    liquidity_taken: float | None = None
    slippage: float | None = None


@dataclass
class ExecutionContext:
    symbol: str
    side: OrderSide
    order_type: OrderType
    order_qty: float
    limit_price: float | None
    timestamp: datetime
    mid_price: float | None = None
    best_bid: float | None = None
    best_ask: float | None = None
    spread: float | None = None
    available_volume: dict[str, float] | None = None
    volatility: float | None = None
    liquidity_regime: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class ExecutionSimulationConfig:
    enable_slippage: bool = True
    enable_latency: bool = True
    enable_partial_fills: bool = True
    max_partial_fill_delay_ms: int = 5000
    default_liquidity_regime: str = "normal"
    metadata: dict[str, Any] | None = None


__all__ = [
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "SimulatedFill",
    "ExecutionContext",
    "ExecutionSimulationConfig",
    "ExecutionClient",
]
