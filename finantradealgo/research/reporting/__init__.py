"""
Research Reporting Module.

Utilities for generating reports, visualizations, and documentation
from research results.
"""

from finantradealgo.research.reporting.base import (
    Report,
    ReportSection,
    ReportFormat,
    ReportGenerator,
)
from finantradealgo.research.reporting.strategy_search import (
    StrategySearchReportGenerator,
)
from finantradealgo.research.reporting.ensemble import (
    EnsembleReportGenerator,
)

__all__ = [
    "Report",
    "ReportSection",
    "ReportFormat",
    "ReportGenerator",
    "StrategySearchReportGenerator",
    "EnsembleReportGenerator",
]
