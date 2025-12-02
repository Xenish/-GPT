from __future__ import annotations

"""
Integration glue between compliance outputs and the application's audit/logging layer.

Application bootstrap can wire:
- ComplianceEngine -> ComplianceAuditSink -> AuditTrailLogger
so compliance results/alerts are persisted via the existing audit trail.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Protocol

from finantradealgo.compliance.models import ComplianceCheckResult, ComplianceAlert

try:
    from finantradealgo.logging.audit_trail import AuditTrailLogger
except ImportError:
    # Fallback protocol to keep this module importable if the concrete audit logger
    # implementation is not present in the current environment.
    class AuditTrailLogger(Protocol):  # type: ignore[override]
        def log_event(
            self,
            *,
            category: str,
            event_type: str,
            severity: str,
            payload: dict[str, Any],
        ) -> None:
            ...


@dataclass
class ComplianceAuditSink:
    """
    Thin wrapper that writes compliance-related events into the generic audit trail.

    This decouples ComplianceEngine from any concrete logging/storage backend.
    """

    audit_logger: AuditTrailLogger

    def log_check_results(self, results: Iterable[ComplianceCheckResult]) -> None:
        """
        Persist raw check results into the audit trail for traceability.
        """
        for res in results:
            self.audit_logger.log_event(
                category="compliance_check",
                event_type=res.check_id,
                severity=res.severity.name.lower(),
                payload={
                    "scope": res.scope.name,
                    "check_type": res.check_type.name,
                    "passed": res.passed,
                    "message": res.message,
                    "details": res.details,
                },
            )

    def log_alerts(self, alerts: Iterable[ComplianceAlert]) -> None:
        """
        Persist high-level compliance alerts into the audit trail.
        """
        for alert in alerts:
            self.audit_logger.log_event(
                category="compliance_alert",
                event_type=alert.alert_id,
                severity=alert.severity.name.lower(),
                payload={
                    "scope": alert.scope.name,
                    "message": alert.message,
                    "created_at_ts": alert.created_at_ts,
                    "related_check_ids": alert.related_check_ids,
                    "metadata": alert.metadata,
                },
            )
