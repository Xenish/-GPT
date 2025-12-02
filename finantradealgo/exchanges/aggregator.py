from __future__ import annotations

"""
Aggregate normalized data across multiple exchange adapters.

Assumptions:
- Prices for a given internal symbol are already normalized into the same quote currency.
- Contract spec differences (inverse/linear, contract size) should be handled inside adapters.
- Fee normalization (taker fees, funding, withdrawal) can be attached later via metadata.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

from finantradealgo.exchanges import (
    ExchangeAdapter,
    ExchangeHealth,
    ExchangeId,
    NormalizedOrderBook,
    NormalizedTicker,
    SymbolMapping,
)
from finantradealgo.exchanges.adapters import (
    BinanceAdapter,
    BybitAdapter,
    OKXAdapter,
    KrakenAdapter,
)


@dataclass
class SymbolRegistry:
    mappings: dict[str, list[SymbolMapping]] = field(default_factory=dict)

    def register_mapping(self, mapping: SymbolMapping) -> None:
        self.mappings.setdefault(mapping.internal_symbol, []).append(mapping)

    def get_mappings(self, internal_symbol: str) -> list[SymbolMapping]:
        return self.mappings.get(internal_symbol, [])

    def get_for_exchange(self, internal_symbol: str, exchange: ExchangeId) -> SymbolMapping | None:
        return next(
            (m for m in self.mappings.get(internal_symbol, []) if m.exchange == exchange),
            None,
        )


@dataclass
class AggregatedQuote:
    symbol: str  # internal symbol
    best_bid: float | None
    best_ask: float | None
    mid_price: float | None
    spreads: dict[ExchangeId, float]
    tickers: dict[ExchangeId, NormalizedTicker]
    timestamp: float
    metadata: dict[str, Any] | None = None


@dataclass
class ArbitrageOpportunity:
    symbol: str
    buy_exchange: ExchangeId
    sell_exchange: ExchangeId
    buy_price: float
    sell_price: float
    edge_bps: float
    timestamp: float
    metadata: dict[str, Any] | None = None


class MultiExchangeAggregator:
    def __init__(
        self,
        adapters: Mapping[ExchangeId, ExchangeAdapter],
        symbol_registry: SymbolRegistry,
    ) -> None:
        self.adapters = dict(adapters)
        self.symbol_registry = symbol_registry

    def get_aggregated_quote(self, internal_symbol: str) -> AggregatedQuote:
        mappings = self.symbol_registry.get_mappings(internal_symbol)
        tickers: dict[ExchangeId, NormalizedTicker] = {}
        spreads: dict[ExchangeId, float] = {}

        for mapping in mappings:
            adapter = self.adapters.get(mapping.exchange)
            if adapter is None:
                continue
            ticker = adapter.get_ticker(mapping)
            tickers[mapping.exchange] = ticker
            if ticker.bid is not None and ticker.ask is not None:
                spreads[mapping.exchange] = ticker.ask - ticker.bid

        best_bid = max((t.bid for t in tickers.values() if t.bid is not None), default=None)
        best_ask = min((t.ask for t in tickers.values() if t.ask is not None), default=None)
        mid_price = (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else None
        timestamp = max((t.timestamp for t in tickers.values()), default=0.0)

        return AggregatedQuote(
            symbol=internal_symbol,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            spreads=spreads,
            tickers=tickers,
            timestamp=timestamp,
            metadata={
                "note": "Prices assumed normalized per internal symbol; adapters must align contract specs and fees."
            },
        )

    def detect_arbitrage(self, internal_symbol: str, min_edge_bps: float = 1.0) -> List[ArbitrageOpportunity]:
        quote = self.get_aggregated_quote(internal_symbol)
        opportunities: List[ArbitrageOpportunity] = []
        exchanges = list(quote.tickers.keys())

        for buy_ex in exchanges:
            buy_ticker = quote.tickers[buy_ex]
            if buy_ticker.ask is None:
                continue
            for sell_ex in exchanges:
                if sell_ex == buy_ex:
                    continue
                sell_ticker = quote.tickers[sell_ex]
                if sell_ticker.bid is None:
                    continue
                edge = sell_ticker.bid - buy_ticker.ask
                if edge <= 0:
                    continue
                edge_bps = (edge / buy_ticker.ask) * 10000
                if edge_bps >= min_edge_bps:
                    opportunities.append(
                        ArbitrageOpportunity(
                            symbol=internal_symbol,
                            buy_exchange=buy_ex,
                            sell_exchange=sell_ex,
                            buy_price=buy_ticker.ask,
                            sell_price=sell_ticker.bid,
                            edge_bps=edge_bps,
                            timestamp=max(buy_ticker.timestamp, sell_ticker.timestamp),
                            metadata={"note": "Fees and slippage not considered; normalize in adapters/metadata."},
                        )
                    )

        return opportunities

    def get_exchange_health(self) -> dict[ExchangeId, ExchangeHealth]:
        return {ex_id: adapter.get_health() for ex_id, adapter in self.adapters.items()}

    def best_price(self, internal_symbol: str) -> tuple[float | None, float | None]:
        quote = self.get_aggregated_quote(internal_symbol)
        return quote.best_bid, quote.best_ask

