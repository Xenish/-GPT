"""
Research Visualization.

Interactive charts and plots for research analysis.
"""

from finantradealgo.research.visualization.charts import (
    ChartConfig,
    ChartType,
    create_chart,
)
from finantradealgo.research.visualization.equity import EquityCurveVisualizer
from finantradealgo.research.visualization.heatmap import ParameterHeatmapVisualizer
from finantradealgo.research.visualization.trades import TradeAnalysisVisualizer
from finantradealgo.research.visualization.dashboard import (
    StrategyDashboard,
    ParameterSearchDashboard,
    ComparisonDashboard,
)

__all__ = [
    # Base
    "ChartConfig",
    "ChartType",
    "create_chart",
    # Visualizers
    "EquityCurveVisualizer",
    "ParameterHeatmapVisualizer",
    "TradeAnalysisVisualizer",
    # Dashboards
    "StrategyDashboard",
    "ParameterSearchDashboard",
    "ComparisonDashboard",
]
