"""
Performance Tracking & Monitoring.

Tracks live strategy performance and compares to backtest expectations.
"""

from finantradealgo.research.performance.models import (
    PerformanceMetrics,
    PerformanceSnapshot,
    PerformanceComparison,
    PerformancePeriod,
    PerformanceStatus,
    PerformanceAlert,
    PerformanceAttribution,
)
from finantradealgo.research.performance.tracker import PerformanceTracker
from finantradealgo.research.performance.comparison import PerformanceComparator
from finantradealgo.research.performance.aggregator import (
    MetricsAggregator,
    PerformanceDatabase,
)
from finantradealgo.research.performance.degradation import (
    DegradationRule,
    PerformanceDegradationDetector,
    ConsecutiveLossDetector,
)
from finantradealgo.research.performance.alerts import (
    AlertManager,
    AlertChannel,
    AlertSubscription,
    get_alert_manager,
)
from finantradealgo.research.performance.attribution import AttributionAnalyzer

__all__ = [
    # Models
    "PerformanceMetrics",
    "PerformanceSnapshot",
    "PerformanceComparison",
    "PerformancePeriod",
    "PerformanceStatus",
    "PerformanceAlert",
    "PerformanceAttribution",
    # Tracking
    "PerformanceTracker",
    "PerformanceComparator",
    # Aggregation
    "MetricsAggregator",
    "PerformanceDatabase",
    # Degradation
    "DegradationRule",
    "PerformanceDegradationDetector",
    "ConsecutiveLossDetector",
    # Alerts
    "AlertManager",
    "AlertChannel",
    "AlertSubscription",
    "get_alert_manager",
    # Attribution
    "AttributionAnalyzer",
]
