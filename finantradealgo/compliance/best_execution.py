from __future__ import annotations

"""
Best execution evaluation module.

This module evaluates whether executed trades align with best available prices
and a configured best execution policy. It does not place orders or modify
routing; it only analyzes executions. Integration points:
- ComplianceEngine via make_best_execution_handler.
- ComplianceReportGenerator via RegulatoryReportType.BEST_EXECUTION (future).
"""

from dataclasses import dataclass, field
from typing import Any, Iterable

from finantradealgo.compliance.models import (
    ComplianceScope,
    ComplianceSeverity,
    ComplianceCheckType,
    ComplianceCheckResult,
    RegulatoryReportType,
)
from finantradealgo.compliance.engine import ComplianceContext
from finantradealgo.exchanges import ExchangeId
from finantradealgo.exchanges.aggregator import MultiExchangeAggregator, AggregatedQuote


@dataclass
class BestExecutionPolicy:
    """
    Configuration for best execution evaluation.

    - max_price_deviation_bps:
        Maximum allowed deviation of execution price vs. best available price
        at the time of execution (in basis points).
    - require_venue_in_allowed_set:
        If True, execution must occur on one of the allowed venues.
    - allowed_venues:
        Optional whitelist of ExchangeId per symbol/account.
    """

    max_price_deviation_bps: float = 5.0
    require_venue_in_allowed_set: bool = False
    allowed_venues: dict[str, list[ExchangeId]] = field(default_factory=dict)


@dataclass
class ExecutedTrade:
    """
    Minimal execution information required for best-ex analysis.
    """

    trade_id: str
    internal_symbol: str
    venue: ExchangeId
    side: str  # "buy" or "sell"
    qty: float
    price: float
    timestamp_ts: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BestExecutionEvaluator:
    """
    Evaluates whether executions comply with best execution policy.

    Uses MultiExchangeAggregator to approximate the "best available price"
    around the time of execution.
    """

    aggregator: MultiExchangeAggregator
    policy: BestExecutionPolicy

    def evaluate_trade(self, trade: ExecutedTrade) -> ComplianceCheckResult:
        """
        Evaluate a single trade against the best execution policy.

        Strategy (simplified):

        1) Fetch an aggregated quote for trade.internal_symbol.
        2) Determine best available price side:
           - BUY: best ask across venues.
           - SELL: best bid across venues.
        3) Compute deviation in bps between execution price and best price.
        4) Check:
           - Deviation <= max_price_deviation_bps.
           - Venue in allowed set (if configured).
        5) Return a ComplianceCheckResult describing pass/fail.
        """
        quote: AggregatedQuote = self.aggregator.get_aggregated_quote(trade.internal_symbol)

        best_price: float | None
        if trade.side.lower() == "buy":
            best_price = quote.best_ask
        else:
            best_price = quote.best_bid

        details: dict[str, Any] = {
            "trade_id": trade.trade_id,
            "venue": trade.venue.name,
            "internal_symbol": trade.internal_symbol,
            "trade_price": trade.price,
            "best_bid": quote.best_bid,
            "best_ask": quote.best_ask,
            "timestamp": trade.timestamp_ts,
        }

        if best_price is None:
            # If we don't have a quote, treat as warning but not a hard violation.
            return ComplianceCheckResult(
                check_id="BEST_EXEC_NO_REFERENCE_PRICE",
                scope=ComplianceScope.TRADE,
                check_type=ComplianceCheckType.POST_TRADE,
                severity=ComplianceSeverity.WARNING,
                passed=True,
                message="No reference price available for best-ex evaluation",
                details=details,
            )

        # Deviation in bps
        deviation_bps = (trade.price - best_price) / best_price * 1e4
        details["best_reference_price"] = best_price
        details["deviation_bps"] = deviation_bps

        # Absolute deviation bound (we care about magnitude)
        abs_dev = abs(deviation_bps)
        within_deviation = abs_dev <= self.policy.max_price_deviation_bps

        # Venue whitelist check
        venue_ok = True
        if self.policy.require_venue_in_allowed_set:
            allowed = self.policy.allowed_venues.get(trade.internal_symbol)
            if allowed is not None:
                venue_ok = trade.venue in allowed
                details["allowed_venues"] = [v.name for v in allowed]

        passed = within_deviation and venue_ok
        severity = ComplianceSeverity.INFO if passed else ComplianceSeverity.VIOLATION

        msg = (
            f"Best execution check: deviation {deviation_bps:.2f} bps, "
            f"max allowed {self.policy.max_price_deviation_bps} bps"
        )
        if not venue_ok:
            msg += " (venue not in allowed set)"

        return ComplianceCheckResult(
            check_id="BEST_EXECUTION_PRICE",
            scope=ComplianceScope.TRADE,
            check_type=ComplianceCheckType.POST_TRADE,
            severity=severity,
            passed=passed,
            message=msg,
            details=details,
        )


def make_best_execution_handler(
    evaluator: BestExecutionEvaluator,
) -> Any:
    """
    Factory for a handler function that can be registered with ComplianceEngine.

    The handler expects a ComplianceContext where:
    - ctx.scope == ComplianceScope.TRADE
    - ctx.payload is an ExecutedTrade or a dict that can build one.
    """

    def handler(ctx: ComplianceContext) -> ComplianceCheckResult:
        payload = ctx.payload
        trade: ExecutedTrade
        if isinstance(payload, ExecutedTrade):
            trade = payload
        else:
            # Assume dict-like, map keys defensively
            trade = ExecutedTrade(
                trade_id=payload.get("trade_id") or payload.get("id"),
                internal_symbol=payload["internal_symbol"],
                venue=payload["venue"],
                side=payload["side"],
                qty=payload["qty"],
                price=payload["price"],
                timestamp_ts=payload.get("timestamp_ts") or payload.get("timestamp"),
                metadata=payload.get("metadata", {}),
            )
        return evaluator.evaluate_trade(trade)

    return handler
