from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class ComplianceScope(Enum):
    TRADE = auto()
    POSITION = auto()
    STRATEGY = auto()
    SYSTEM = auto()


class ComplianceSeverity(Enum):
    INFO = auto()
    WARNING = auto()
    VIOLATION = auto()
    BLOCKING = auto()


class ComplianceCheckType(Enum):
    PRE_TRADE = auto()
    POST_TRADE = auto()
    INTRADAY = auto()
    END_OF_DAY = auto()
    REPORTING = auto()


class RegulatoryReportType(Enum):
    TRADE = auto()
    POSITION = auto()
    RISK = auto()
    BEST_EXECUTION = auto()


@dataclass
class ComplianceCheckResult:
    """
    Result of a single compliance check.

    - check_id: stable identifier for the check (e.g. "MAX_POSITION_LIMIT").
    - scope: what the check applies to (trade/position/strategy/system).
    - severity: informational vs warning vs violation.
    - passed: True if check passed, False if violated.
    """

    check_id: str
    scope: ComplianceScope
    check_type: ComplianceCheckType
    severity: ComplianceSeverity
    passed: bool
    message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceAlert:
    """
    High-level alert generated from one or more check results.

    This is what will be sent to logging/audit systems, Slack, email, etc.
    """

    alert_id: str
    scope: ComplianceScope
    severity: ComplianceSeverity
    message: str
    created_at_ts: float
    related_check_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegulatoryReportRequest:
    """
    Request to generate a regulatory-style report (MiFID-like) for a given period.
    """

    report_type: RegulatoryReportType
    as_of_ts: float | None = None
    start_ts: float | None = None
    end_ts: float | None = None
    scope_filter: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegulatoryReport:
    """
    Generic regulatory report container.

    payload:
        Arbitrary structured data (list of rows, dicts, etc.) that can be
        serialized into CSV/JSON/XML as required by downstream systems.
    """

    request: RegulatoryReportRequest
    generated_at_ts: float
    payload: Any
    metadata: dict[str, Any] = field(default_factory=dict)
