from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from finantradealgo.execution import OrderSide, OrderType


class TimeInForce(Enum):
    GTC = auto()  # Good-Til-Cancel
    IOC = auto()  # Immediate-Or-Cancel
    FOK = auto()  # Fill-Or-Kill
    DAY = auto()  # Day order (session-based)


class AlgoOrderKind(Enum):
    TWAP = auto()
    VWAP = auto()
    IMPLEMENTATION_SHORTFALL = auto()
    ADAPTIVE = auto()


@dataclass
class OrderSpec:
    """
    Base specification for an order intent.

    This is independent from the exact exchange API and execution client.
    Algo engines and smart routing will transform these into concrete child orders.
    """

    internal_symbol: str
    side: OrderSide
    qty: float
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.GTC

    # Optional price fields
    limit_price: float | None = None
    stop_price: float | None = None

    # Optional client-provided ID for tracking
    client_order_id: str | None = None

    # Free-form metadata (strategy, bucket, tags, etc.)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IcebergOrderSpec:
    base: OrderSpec
    display_qty: float
    max_slices: int | None = None
    min_slice_qty: float | None = None
    refresh_delay_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StopLimitOrderSpec:
    base: OrderSpec
    stop_price: float
    limit_price: float
    metadata: dict[str, Any] = field(default_factory=dict)


class TrailingMode(Enum):
    ABSOLUTE = auto()  # trailing offset is in price units
    PERCENT = auto()  # trailing offset is in percentage


@dataclass
class TrailingStopOrderSpec:
    base: OrderSpec
    trailing_offset: float
    mode: TrailingMode = TrailingMode.PERCENT
    activation_price: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OCOOrderLeg:
    """One leg of an OCO pair; typically a limit or stop-limit order."""

    spec: OrderSpec
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OCOOrderSpec:
    """
    One-Cancels-Other order:
    execution of one leg should cancel the other leg.
    """

    internal_symbol: str
    leg_a: OCOOrderLeg
    leg_b: OCOOrderLeg
    client_order_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderSlice:
    """
    A single executable slice of an order.

    This is close to what an execution client will see:
    - exchange_symbol is resolved later from routing/symbol mapping.
    - internal_symbol + side + qty + limit_price are enough for planning.
    """

    internal_symbol: str
    side: OrderSide
    qty: float

    # Optional price control for this slice
    limit_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.GTC

    # Planned execution timestamp (wall clock / epoch seconds)
    planned_ts: float | None = None

    # Optional “preferred venue” hint; routing may override
    preferred_exchange: Any | None = None  # e.g. ExchangeId

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AlgoOrderSpec:
    """
    High-level algorithmic order specification (TWAP/VWAP/etc.).
    """

    base: OrderSpec
    kind: AlgoOrderKind
    start_ts: float | None = None
    end_ts: float | None = None
    target_slices: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
