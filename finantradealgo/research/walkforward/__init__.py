"""
Walk-Forward Optimization.

Out-of-sample validation framework for strategy robustness testing.
"""

from finantradealgo.research.walkforward.models import (
    WindowType,
    OptimizationMetric,
    WalkForwardConfig,
    WalkForwardWindow,
    WalkForwardResult,
    WalkForwardComparison,
)
from finantradealgo.research.walkforward.optimizer import WalkForwardOptimizer
from finantradealgo.research.walkforward.validator import (
    OutOfSampleValidator,
    ValidationStatus,
    ValidationReport,
)
from finantradealgo.research.walkforward.analysis import (
    WalkForwardAnalyzer,
    EfficiencyMetrics,
)
from finantradealgo.research.walkforward.visualization import WalkForwardVisualizer

__all__ = [
    # Enums
    "WindowType",
    "OptimizationMetric",
    "ValidationStatus",
    # Models
    "WalkForwardConfig",
    "WalkForwardWindow",
    "WalkForwardResult",
    "WalkForwardComparison",
    "ValidationReport",
    "EfficiencyMetrics",
    # Optimizer
    "WalkForwardOptimizer",
    # Validator
    "OutOfSampleValidator",
    # Analyzer
    "WalkForwardAnalyzer",
    # Visualizer
    "WalkForwardVisualizer",
]
