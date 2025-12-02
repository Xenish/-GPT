from __future__ import annotations

"""
Exchange routing layer for live trading.

Assumptions:
- Prices for an internal symbol are already normalized to the same quote asset by adapters.
- Contract/lot/fee differences should be handled in adapters or passed via metadata, not here.
- This module is pure routing logic: it does not place orders or mutate exchange state.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from finantradealgo.exchanges import ExchangeId, SymbolMapping

try:
    from finantradealgo.exchanges import ExchangeHealthMonitor
except ImportError:
    from finantradealgo.exchanges.health_monitor import ExchangeHealthMonitor

try:
    from finantradealgo.exchanges import MultiExchangeAggregator
except ImportError:
    from finantradealgo.exchanges.aggregator import MultiExchangeAggregator

try:
    from finantradealgo.exchanges import SymbolRegistry
except ImportError:  # SymbolRegistry is currently defined in aggregator.py only.
    from finantradealgo.exchanges.aggregator import SymbolRegistry


@dataclass
class RoutingConfig:
    """
    Configuration for exchange routing decisions.

    - default_exchange_per_symbol:
        Internal symbol -> preferred primary exchange.
    - backup_exchanges_per_symbol:
        Internal symbol -> ordered list of backup exchanges.
    - min_health_score:
        Threshold below which an exchange is considered unhealthy.
    - prefer_best_price:
        If True, route to the best-priced healthy exchange instead of always
        sticking to primary.
    """

    default_exchange_per_symbol: dict[str, ExchangeId]
    backup_exchanges_per_symbol: dict[str, list[ExchangeId]]
    min_health_score: float = 0.5
    prefer_best_price: bool = True
    metadata: dict[str, Any] | None = None


@dataclass
class RoutingDecision:
    internal_symbol: str
    chosen_exchange: ExchangeId
    symbol_mapping: SymbolMapping
    reason: str
    health_scores: dict[ExchangeId, float]
    metadata: dict[str, Any] | None = None


class ExchangeRoutingEngine:
    """
    Exchange routing engine for live trading.

    Responsibilities:
    - Given an internal symbol, decide which exchange to route to.
    - Use:
        - RoutingConfig preferences (primary + backups).
        - ExchangeHealthMonitor scores.
        - MultiExchangeAggregator (optional) for best price info.
    """

    def __init__(
        self,
        aggregator: MultiExchangeAggregator,
        health_monitor: ExchangeHealthMonitor,
        symbol_registry: SymbolRegistry,
        config: RoutingConfig,
    ) -> None:
        self.aggregator = aggregator
        self.health_monitor = health_monitor
        self.symbol_registry = symbol_registry
        self.config = config

    def _get_health_scores(self) -> dict[ExchangeId, float]:
        """
        Fetch or compute health scores from the ExchangeHealthMonitor.

        For now, call `compute_scores()` if available, or derive a trivial score
        from the latest health snapshots.
        """

        if hasattr(self.health_monitor, "compute_scores"):
            return self.health_monitor.compute_scores()

        scores: dict[ExchangeId, float] = {}
        history: Mapping[ExchangeId, Any] | None = getattr(self.health_monitor, "history", None)
        if history:
            for ex_id, hist in history.items():
                latest = getattr(hist, "latest", lambda: None)()
                if latest is None:
                    scores[ex_id] = 0.0
                    continue
                status = getattr(latest, "status", None)
                status_name = getattr(status, "name", str(status))
                if status_name == "DOWN":
                    scores[ex_id] = 0.0
                elif status_name == "DEGRADED":
                    scores[ex_id] = 0.4
                else:
                    scores[ex_id] = 0.8
        return scores

    def _get_candidates(self, internal_symbol: str) -> list[ExchangeId]:
        """
        Returns [primary, *backups] for the given internal_symbol based on config.
        If not found in config, infer from symbol_registry mappings.
        """

        candidates: list[ExchangeId] = []
        primary = self.config.default_exchange_per_symbol.get(internal_symbol)
        if primary is not None:
            candidates.append(primary)

        for backup in self.config.backup_exchanges_per_symbol.get(internal_symbol, []):
            if backup not in candidates:
                candidates.append(backup)

        if not candidates:
            for mapping in self.symbol_registry.get_mappings(internal_symbol):
                if mapping.exchange not in candidates:
                    candidates.append(mapping.exchange)

        return candidates

    def _choose_best_priced(
        self,
        internal_symbol: str,
        healthy_candidates: Sequence[ExchangeId],
        health_scores: Mapping[ExchangeId, float],
    ) -> tuple[ExchangeId | None, str]:
        """
        Pick the healthiest best-priced exchange among candidates.

        Assumption: direction-neutral. We prioritize the tightest spread; if spread
        data is missing, fall back to the most recent ticker among healthy venues.
        """

        quote = self.aggregator.get_aggregated_quote(internal_symbol)
        ranking: list[tuple[float, float, float, ExchangeId]] = []

        for ex in healthy_candidates:
            ticker = quote.tickers.get(ex)
            spread = quote.spreads.get(ex)
            if spread is not None:
                price_score = spread
            elif ticker and ticker.bid is not None and ticker.ask is not None:
                price_score = abs((ticker.bid + ticker.ask) / 2 - (quote.mid_price or 0.0))
            else:
                price_score = float("inf")

            recency_penalty = -ticker.timestamp if ticker else 0.0
            ranking.append((price_score, -health_scores.get(ex, 0.0), recency_penalty, ex))

        if not ranking:
            return None, "healthy_no_quote"

        ranking.sort()
        return ranking[0][-1], "best_price"

    def choose_exchange(self, internal_symbol: str) -> RoutingDecision:
        """
        Decide which exchange to route an order for `internal_symbol`.

        Algorithm (high-level):

        1) Collect candidate exchanges:
           - Primary from RoutingConfig.default_exchange_per_symbol.
           - Backup list from RoutingConfig.backup_exchanges_per_symbol.
           - Fallback: any exchange that has a mapping for this symbol.

        2) Fetch health scores from ExchangeHealthMonitor.
           - Filter out exchanges with score < config.min_health_score.

        3) If prefer_best_price is False:
           - Pick the first healthy candidate in order (primary then backups).

        4) If prefer_best_price is True:
           - Use MultiExchangeAggregator to fetch an aggregated quote.
           - Among healthy candidates, pick the one with the best price:
               - For buys: lowest ask.
               - For sells: highest bid.
             (For now, assume we are neutral; you can pick based on best spread
              or closeness to mid-price - document the choice.)

        5) Resolve SymbolMapping for the chosen exchange via SymbolRegistry.

        6) If no healthy exchange is found:
           - Fallback to primary even if below threshold, but mark reason accordingly.
        """

        candidates = self._get_candidates(internal_symbol)
        if not candidates:
            raise ValueError(f"No exchange candidates found for symbol: {internal_symbol}")

        health_scores = self._get_health_scores()
        healthy_candidates = [
            ex for ex in candidates if health_scores.get(ex, 0.0) >= self.config.min_health_score
        ]

        chosen_exchange: ExchangeId | None = None
        reason = "fallback_unhealthy"

        if self.config.prefer_best_price:
            chosen_exchange, reason = self._choose_best_priced(
                internal_symbol, healthy_candidates, health_scores
            )
            if chosen_exchange is None and healthy_candidates:
                chosen_exchange = healthy_candidates[0]
                reason = "healthy_no_quote"
        else:
            if healthy_candidates:
                chosen_exchange = healthy_candidates[0]
                reason = "primary" if chosen_exchange == candidates[0] else "backup"

        if chosen_exchange is None:
            chosen_exchange = candidates[0]

        symbol_mapping = self.symbol_registry.get_for_exchange(internal_symbol, chosen_exchange)
        if symbol_mapping is None:
            raise ValueError(
                f"No symbol mapping found for {internal_symbol} on exchange {chosen_exchange}"
            )

        decision_metadata = dict(self.config.metadata or {})
        if reason.startswith("fallback") and self.config.metadata:
            decision_metadata["note"] = "Used fallback due to lack of healthy exchanges."

        return RoutingDecision(
            internal_symbol=internal_symbol,
            chosen_exchange=chosen_exchange,
            symbol_mapping=symbol_mapping,
            reason=reason,
            health_scores=dict(health_scores),
            metadata=decision_metadata or None,
        )
