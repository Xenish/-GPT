"""
Research Reporting Module.

Utilities for generating reports, visualizations, and documentation
from research results.
"""

from finantradealgo.research.reporting.base import (
    Report,
    ReportSection,
    ReportFormat,
    ReportProfile,
    ReportGenerator,
)
from finantradealgo.research.reporting.strategy_search import (
    StrategySearchReportGenerator,
)
from finantradealgo.research.reporting.ensemble import (
    EnsembleReportGenerator,
)
from finantradealgo.research.reporting.backtest_report import (
    BacktestReportGenerator,
)
from finantradealgo.research.reporting.live_report import (
    LiveReportGenerator,
)

__all__ = [
    "Report",
    "ReportSection",
    "ReportFormat",
    "ReportProfile",
    "ReportGenerator",
    "StrategySearchReportGenerator",
    "EnsembleReportGenerator",
    "BacktestReportGenerator",
    "LiveReportGenerator",
]
