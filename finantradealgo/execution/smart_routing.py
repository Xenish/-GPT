from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

from finantradealgo.execution.order_types import OrderSpec, OrderSlice
from finantradealgo.execution import OrderSide
from finantradealgo.exchanges import (
    ExchangeId,
    SymbolMapping,
    MultiExchangeAggregator,
    ExchangeHealthMonitor,
)
from finantradealgo.exchanges.aggregator import SymbolRegistry


@dataclass
class SmartRoutingConfig:
    """
    Configuration for smart order routing across multiple exchanges.

    - max_venues:
        Max number of exchanges to use for a single order.
    - min_venue_share:
        Minimum fraction of total qty allocated to any chosen venue.
    - min_health_score:
        Exchanges below this score are ignored.
    - consider_spread:
        If True, routing prefers tighter spreads.
    - consider_depth:
        If True, routing considers depth / available liquidity (future hook).
    """

    max_venues: int = 2
    min_venue_share: float = 0.1
    min_health_score: float = 0.5
    consider_spread: bool = True
    consider_depth: bool = True
    metadata: dict[str, Any] | None = None


@dataclass
class VenueAllocation:
    exchange: ExchangeId
    symbol_mapping: SymbolMapping
    qty: float
    limit_price: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SmartRoutePlan:
    order: OrderSpec
    allocations: list[VenueAllocation]
    health_scores: dict[ExchangeId, float]
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def total_allocated_qty(self) -> float:
        return sum(a.qty for a in self.allocations)


class SmartOrderRouter:
    """
    Smart order routing engine.

    Responsibilities:
    - Given an OrderSpec, plan how to allocate qty across exchanges.
    - Use MultiExchangeAggregator for best bid/ask.
    - Use ExchangeHealthMonitor for filtering unhealthy venues.
    - Produce a SmartRoutePlan that algo orders / execution engine can consume.
    """

    def __init__(
        self,
        aggregator: MultiExchangeAggregator,
        health_monitor: ExchangeHealthMonitor,
        symbol_registry: SymbolRegistry,
        config: SmartRoutingConfig | None = None,
    ) -> None:
        self.aggregator = aggregator
        self.health_monitor = health_monitor
        self.symbol_registry = symbol_registry
        self.config = config or SmartRoutingConfig()

    def _get_health_scores(self) -> dict[ExchangeId, float]:
        """
        Fetch or compute health scores from the ExchangeHealthMonitor.

        For now, call a `compute_scores()` method if present, otherwise
        derive trivial scores from latest health snapshots (e.g. 1.0 for HEALTHY,
        0.5 for DEGRADED, 0.0 for DOWN).
        """
        if hasattr(self.health_monitor, "compute_scores"):
            scores = self.health_monitor.compute_scores()
            if scores:
                return scores

        # Fallback: derive from latest health snapshots if available.
        from finantradealgo.exchanges import ExchangeStatus

        scores: dict[ExchangeId, float] = {}
        history = getattr(self.health_monitor, "history", {})
        for ex_id, hist in history.items():
            latest = hist.latest() if hasattr(hist, "latest") else None
            if latest is None:
                scores[ex_id] = 0.0
                continue
            status = latest.status
            if status == ExchangeStatus.HEALTHY:
                scores[ex_id] = 1.0
            elif status == ExchangeStatus.DEGRADED:
                scores[ex_id] = 0.5
            else:
                scores[ex_id] = 0.0
        return scores

    def _candidate_mappings(self, order: OrderSpec) -> list[SymbolMapping]:
        """
        Return all SymbolMapping entries for this internal symbol.

        The SymbolRegistry maps internal_symbol -> [SymbolMapping per exchange].
        """
        return self.symbol_registry.get_mappings(order.internal_symbol)

    def _filter_by_health(
        self,
        mappings: list[SymbolMapping],
        scores: dict[ExchangeId, float],
    ) -> list[SymbolMapping]:
        """
        Filter mappings to those whose exchange has health score >= config.min_health_score.
        """
        return [
            m for m in mappings if scores.get(m.exchange, 0.0) >= self.config.min_health_score
        ]

    def plan_route(self, order: OrderSpec) -> SmartRoutePlan:
        """
        Build a SmartRoutePlan for the given order.

        Algorithm (simplified but realistic):

        1) Get health scores and symbol mappings for order.internal_symbol.
        2) Filter out exchanges below min_health_score.
        3) Fetch an aggregated quote for the symbol via MultiExchangeAggregator.
        4) For eligible exchanges:
            - For BUY:
                prefer lower ask.
            - For SELL:
                prefer higher bid.
        5) Select up to config.max_venues venues with best prices.
        6) Allocate quantity across chosen venues:
            - Basic strategy: proportional to “price attractiveness”,
              constrained by min_venue_share.
        7) Optionally set per-venue limit_price close to that venue's
           best bid/ask (with small safety margin).
        8) Return SmartRoutePlan with VenueAllocations & health scores.
        """
        health_scores = self._get_health_scores()
        mappings = self._candidate_mappings(order)
        healthy_mappings = self._filter_by_health(mappings, health_scores)

        aggregated = self.aggregator.get_aggregated_quote(order.internal_symbol)

        attractiveness: list[tuple[SymbolMapping, float, float]] = []
        for mapping in healthy_mappings:
            ticker = aggregated.tickers.get(mapping.exchange)
            if ticker is None:
                continue
            price = ticker.ask if order.side == OrderSide.BUY else ticker.bid
            if price is None:
                continue
            # Lower ask is better for BUY, higher bid is better for SELL.
            base_score = (1.0 / price) if order.side == OrderSide.BUY else price
            score = base_score * max(health_scores.get(mapping.exchange, 0.0), 0.0)

            if self.config.consider_spread:
                spread = aggregated.spreads.get(mapping.exchange)
                if spread is not None and price > 0:
                    score = score / (1.0 + max(spread / price, 0.0))

            # consider_depth hook: placeholder for future depth-aware scoring.
            attractiveness.append((mapping, price, score))

        # Sort by attractiveness descending and take top venues.
        sorted_candidates = sorted(attractiveness, key=lambda x: x[2], reverse=True)
        selected = sorted_candidates[: self.config.max_venues]

        # Filter out zero scores.
        selected = [entry for entry in selected if entry[2] > 0]
        weights = [entry[2] for entry in selected]
        total_weight = sum(weights)

        allocations: list[VenueAllocation] = []
        if total_weight > 0 and selected:
            # Enforce min_venue_share by dropping venues below the threshold after normalization.
            normalized = [
                (mapping, price, weight / total_weight) for (mapping, price, weight) in selected
            ]
            normalized = [
                (mapping, price, share) for (mapping, price, share) in normalized if share >= self.config.min_venue_share
            ]

            if not normalized and selected:
                # Fallback: keep the top venue if min_venue_share filtered out everything.
                mapping, price, weight = selected[0]
                normalized = [(mapping, price, 1.0)]

            remaining_qty = order.qty
            for idx, (mapping, price, share) in enumerate(normalized):
                qty = order.qty * share if idx < len(normalized) - 1 else remaining_qty
                if idx < len(normalized) - 1:
                    remaining_qty -= qty

                limit_price: float | None = None
                if order.side == OrderSide.BUY:
                    limit_price = price * 1.001 if price is not None else None
                else:
                    limit_price = price * 0.999 if price is not None else None

                allocations.append(
                    VenueAllocation(
                        exchange=mapping.exchange,
                        symbol_mapping=mapping,
                        qty=qty,
                        limit_price=limit_price,
                        metadata={"attractiveness": share},
                    )
                )

        reason = (
            "health-filtered best-venue selection based on bid/ask and spread weighting"
        )
        return SmartRoutePlan(
            order=order,
            allocations=allocations,
            health_scores=health_scores,
            reason=reason,
            metadata={"aggregated_quote_ts": aggregated.timestamp},
        )

    def to_slices(self, plan: SmartRoutePlan) -> list[OrderSlice]:
        """
        Convert a SmartRoutePlan into OrderSlice objects.

        - One slice per VenueAllocation by default.
        - `preferred_exchange` is set on the slice, but routing/execution
          may override in edge cases.
        """
        slices: list[OrderSlice] = []
        for alloc in plan.allocations:
            slices.append(
                OrderSlice(
                    internal_symbol=plan.order.internal_symbol,
                    side=plan.order.side,
                    qty=alloc.qty,
                    limit_price=alloc.limit_price,
                    preferred_exchange=alloc.exchange,
                    metadata={"route_reason": plan.reason, **alloc.metadata},
                )
            )
        return slices
