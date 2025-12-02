from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol, Sequence

from finantradealgo.compliance.models import (
    RegulatoryReportRequest,
    RegulatoryReport,
    RegulatoryReportType,
)


class TradeRepository(Protocol):
    """
    Read-only interface for fetching trade data for compliance/reporting.
    """

    def fetch_trades(
        self,
        start_ts: float | None = None,
        end_ts: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> Sequence[dict[str, Any]]:
        ...


class PositionRepository(Protocol):
    """
    Read-only interface for fetching position snapshots.
    """

    def fetch_positions(
        self,
        as_of_ts: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> Sequence[dict[str, Any]]:
        ...


class RiskMetricsRepository(Protocol):
    """
    Read-only interface for fetching aggregated risk metrics.
    """

    def fetch_risk_metrics(
        self,
        as_of_ts: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...


@dataclass
class ComplianceReportGenerator:
    """
    Generates regulatory-style reports (MiFID-like) from underlying repositories.

    This class does NOT perform I/O by itself; it just constructs structured payloads
    that can later be serialized to CSV/JSON/XML.

    Repositories must be provided by the application:
    - DB-backed
    - In-memory
    - Or adapters over existing storage.
    """

    trades: TradeRepository
    positions: PositionRepository
    risk: RiskMetricsRepository

    def generate(self, request: RegulatoryReportRequest) -> RegulatoryReport:
        import time

        if request.report_type is RegulatoryReportType.TRADE:
            payload = self._build_trade_report(request)
        elif request.report_type is RegulatoryReportType.POSITION:
            payload = self._build_position_report(request)
        elif request.report_type is RegulatoryReportType.RISK:
            payload = self._build_risk_report(request)
        elif request.report_type is RegulatoryReportType.BEST_EXECUTION:
            # Will be implemented in best_execution module later
            payload = {
                "message": "Best execution report is generated elsewhere",
            }
        else:
            payload = {
                "message": f"Unsupported report type: {request.report_type}",
            }

        return RegulatoryReport(
            request=request,
            generated_at_ts=time.time(),
            payload=payload,
        )

    def _build_trade_report(self, request: RegulatoryReportRequest) -> list[dict[str, Any]]:
        trades = self.trades.fetch_trades(
            start_ts=request.start_ts,
            end_ts=request.end_ts,
            filters=request.scope_filter or {},
        )
        rows: list[dict[str, Any]] = []

        for t in trades:
            row = {
                # Example schema fields, can be aligned to MiFID-style later:
                "trade_id": t.get("id"),
                "timestamp": t.get("timestamp"),
                "client_id": t.get("client_id"),
                "strategy_id": t.get("strategy_id"),
                "symbol": t.get("symbol"),
                "side": t.get("side"),
                "qty": t.get("qty"),
                "price": t.get("price"),
                "venue": t.get("venue"),
                "order_type": t.get("order_type"),
            }
            rows.append(row)

        return rows

    def _build_position_report(self, request: RegulatoryReportRequest) -> list[dict[str, Any]]:
        positions = self.positions.fetch_positions(
            as_of_ts=request.as_of_ts,
            filters=request.scope_filter or {},
        )
        rows: list[dict[str, Any]] = []

        for p in positions:
            row = {
                "account_id": p.get("account_id"),
                "symbol": p.get("symbol"),
                "qty": p.get("qty"),
                "avg_entry_price": p.get("avg_entry_price"),
                "unrealized_pnl": p.get("unrealized_pnl"),
                "realized_pnl": p.get("realized_pnl"),
                "leverage": p.get("leverage"),
            }
            rows.append(row)

        return rows

    def _build_risk_report(self, request: RegulatoryReportRequest) -> dict[str, Any]:
        metrics = self.risk.fetch_risk_metrics(
            as_of_ts=request.as_of_ts,
            filters=request.scope_filter or {},
        )
        # Just pass through for now; enforcement of schema can be added later.
        return metrics
