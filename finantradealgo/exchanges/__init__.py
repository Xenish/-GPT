from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Protocol, Mapping


class ExchangeId(Enum):
    BINANCE = auto()
    BYBIT = auto()
    OKX = auto()
    KRAKEN = auto()


class MarketType(Enum):
    SPOT = auto()
    PERP = auto()
    FUTURES = auto()


@dataclass
class SymbolMapping:
    internal_symbol: str  # e.g. "BTCUSDT_PERP"
    exchange_symbol: str  # e.g. "BTCUSDT", "BTC-USDT-SWAP"
    exchange: ExchangeId
    market_type: MarketType
    base_asset: str
    quote_asset: str
    metadata: dict[str, Any] | None = None


@dataclass
class NormalizedTicker:
    exchange: ExchangeId
    symbol: str  # internal symbol
    bid: float
    ask: float
    last_price: float
    timestamp: float  # unix seconds
    volume_24h: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class NormalizedOrderBook:
    exchange: ExchangeId
    symbol: str
    bids: list[tuple[float, float]]  # (price, qty)
    asks: list[tuple[float, float]]
    timestamp: float
    metadata: dict[str, Any] | None = None


@dataclass
class NormalizedTrade:
    exchange: ExchangeId
    symbol: str
    price: float
    qty: float
    side: str  # "buy"/"sell" (taker side)
    timestamp: float
    metadata: dict[str, Any] | None = None


class ExchangeStatus(Enum):
    HEALTHY = auto()
    DEGRADED = auto()
    DOWN = auto()


@dataclass
class ExchangeHealth:
    exchange: ExchangeId
    status: ExchangeStatus
    uptime_ratio: float | None = None
    avg_latency_ms: float | None = None
    data_delay_ms: float | None = None
    last_heartbeat_ts: float | None = None
    data_quality_score: float | None = None  # 0-1
    metadata: dict[str, Any] | None = None


class ExchangeAdapter(Protocol):
    """
    Unified interface for exchange data adapters.
    This is for read-only data (order book, ticker, trades, OHLCV).
    """

    def get_ticker(self, symbol: SymbolMapping) -> NormalizedTicker:
        ...

    def get_order_book(self, symbol: SymbolMapping, depth: int = 50) -> NormalizedOrderBook:
        ...

    def get_recent_trades(self, symbol: SymbolMapping, limit: int = 100) -> list[NormalizedTrade]:
        ...

    def get_ohlcv(
        self,
        symbol: SymbolMapping,
        timeframe: str,
        start_ts: float | None = None,
        end_ts: float | None = None,
        limit: int | None = None,
    ) -> Any:
        """
        Return OHLCV as a pandas.DataFrame-like object:
        index: timestamp, columns: open, high, low, close, volume.
        (Actual type will depend on rest of the project; we keep it generic here.)
        """
        ...

    def get_health(self) -> ExchangeHealth:
        ...
