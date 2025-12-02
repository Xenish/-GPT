"""
Ensemble and Multi-Armed Bandit Research Module.

This module implements meta-strategies that combine multiple base strategies
using ensemble methods and multi-armed bandit algorithms.
"""

from finantradealgo.research.ensemble.base import (
    EnsembleStrategy,
    EnsembleConfig,
    ComponentStrategy,
)
from finantradealgo.research.ensemble.weighted import (
    WeightedEnsembleStrategy,
    WeightedEnsembleConfig,
    WeightingMethod,
)
from finantradealgo.research.ensemble.bandit import (
    BanditEnsembleStrategy,
    BanditEnsembleConfig,
    BanditAlgorithm,
)
from finantradealgo.research.ensemble.portfolio import (
    PortfolioConfig,
    Portfolio,
    PortfolioWeight,
    PortfolioWeightingMethod,
    PortfolioBacktestResult,
    PortfolioComparisonResult,
    CorrelationMatrix,
    RebalanceFrequency,
)
from finantradealgo.research.ensemble.optimizer import PortfolioOptimizer
from finantradealgo.research.ensemble.correlation import (
    CorrelationAnalyzer,
    DiversificationOptimizer,
)
from finantradealgo.research.ensemble.rebalancer import (
    PortfolioRebalancer,
    AdaptiveRebalancer,
    RebalanceEvent,
)
from finantradealgo.research.ensemble.portfolio_backtest import (
    PortfolioBacktester,
    load_strategy_returns_from_files,
    calculate_contribution_attribution,
)

__all__ = [
    # Ensemble (existing)
    "EnsembleStrategy",
    "EnsembleConfig",
    "ComponentStrategy",
    "WeightedEnsembleStrategy",
    "WeightedEnsembleConfig",
    "WeightingMethod",
    "BanditEnsembleStrategy",
    "BanditEnsembleConfig",
    "BanditAlgorithm",
    # Portfolio
    "PortfolioConfig",
    "Portfolio",
    "PortfolioWeight",
    "PortfolioWeightingMethod",
    "PortfolioBacktestResult",
    "PortfolioComparisonResult",
    "CorrelationMatrix",
    "RebalanceFrequency",
    "PortfolioOptimizer",
    "CorrelationAnalyzer",
    "DiversificationOptimizer",
    "PortfolioRebalancer",
    "AdaptiveRebalancer",
    "RebalanceEvent",
    "PortfolioBacktester",
    "load_strategy_returns_from_files",
    "calculate_contribution_attribution",
]
