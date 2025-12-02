from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .models import (
    ComplianceCheckResult,
    ComplianceAlert,
    RegulatoryReportRequest,
    RegulatoryReport,
    ComplianceScope,
    ComplianceCheckType,
)


@dataclass
class ComplianceContext:
    """
    Context passed into compliance checks.

    This is intentionally generic and can carry:
    - trade: execution report / order info
    - position: current portfolio snapshot
    - strategy: strategy metadata and params
    - system: risk metrics, health, etc.

    Higher-level code is responsible for building a suitable context object
    for each check.
    """

    scope: ComplianceScope
    payload: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class ComplianceEngine:
    """
    High-level compliance engine.

    Responsibilities:
    - Run pre-trade and post-trade checks.
    - Run periodic / end-of-day checks (positions, risk, limits).
    - Generate regulatory-style reports.
    - Produce alerts that can be sent to logging/audit/notification systems.

    This class is intentionally conservative: it does not send alerts by itself;
    callers can route ComplianceAlert objects to logging, DB, email, etc.
    """

    def __init__(self) -> None:
        # In a more advanced version, this could load check configurations
        # from YAML/DB, or allow registration of custom check handlers.
        self._registered_checks: dict[str, Any] = {}

    def register_check(
        self,
        check_id: str,
        handler: Any,
        *,
        scope: ComplianceScope,
        check_type: ComplianceCheckType,
    ) -> None:
        """
        Register a compliance check handler.

        handler is a callable that takes a ComplianceContext and returns
        a ComplianceCheckResult or an iterable of ComplianceCheckResult.

        Example signature:
            def max_position_check(ctx: ComplianceContext) -> ComplianceCheckResult: ...
        """
        self._registered_checks[check_id] = {
            "handler": handler,
            "scope": scope,
            "check_type": check_type,
        }

    def run_checks(
        self,
        ctx: ComplianceContext,
        *,
        check_type: ComplianceCheckType | None = None,
    ) -> list[ComplianceCheckResult]:
        """
        Run all registered checks applicable to this context.

        - If check_type is provided, only run checks of that type.
        - Scope is matched against ctx.scope.
        """
        results: list[ComplianceCheckResult] = []

        for check_id, meta in self._registered_checks.items():
            if meta["scope"] != ctx.scope:
                continue
            if check_type is not None and meta["check_type"] != check_type:
                continue

            handler = meta["handler"]
            out = handler(ctx)

            if isinstance(out, ComplianceCheckResult):
                results.append(out)
            elif out is not None:
                results.extend(out)

        return results

    def build_alerts(
        self,
        results: Iterable[ComplianceCheckResult],
    ) -> list[ComplianceAlert]:
        """
        Build high-level alerts from a list of check results.

        The default implementation:
        - Groups violations by severity and scope.
        - Creates one alert per failing check.
        """
        import time

        alerts: list[ComplianceAlert] = []
        ts = time.time()

        for res in results:
            if res.passed:
                continue
            alert = ComplianceAlert(
                alert_id=res.check_id,
                scope=res.scope,
                severity=res.severity,
                message=res.message or f"Compliance check failed: {res.check_id}",
                created_at_ts=ts,
                related_check_ids=[res.check_id],
                metadata=res.details.copy(),
            )
            alerts.append(alert)

        return alerts

    def generate_report(
        self,
        request: RegulatoryReportRequest,
    ) -> RegulatoryReport:
        """
        Generate a regulatory-style report.

        This is a high-level facade; concrete implementations should:
        - Fetch trades/positions/risk metrics from storage.
        - Transform them into the required schema for the given report_type.
        - Return a RegulatoryReport containing structured payload (rows/dicts).

        For now, this is a placeholder; later we will delegate to dedicated
        reporting modules (e.g. compliance.reporting).
        """
        import time

        # TODO: call into finantradealgo.compliance.reporting when implemented
        payload: Any = {
            "message": "Regulatory reporting not yet implemented",
            "report_type": request.report_type.name,
        }
        return RegulatoryReport(
            request=request,
            generated_at_ts=time.time(),
            payload=payload,
        )
