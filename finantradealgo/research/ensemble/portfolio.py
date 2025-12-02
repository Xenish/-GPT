"""
Portfolio Construction Models.

Data structures for combining backtested strategies into optimized portfolios.
This complements the runtime ensemble (base.py) with post-backtest portfolio optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd
import numpy as np


class PortfolioWeightingMethod(str, Enum):
    """Portfolio weighting method."""
    EQUAL = "equal"  # Equal weights
    PERFORMANCE = "performance"  # Based on returns
    SHARPE = "sharpe"  # Based on Sharpe ratio
    RISK_PARITY = "risk_parity"  # Based on inverse volatility
    MINIMUM_VARIANCE = "minimum_variance"  # Minimum portfolio variance
    MAXIMUM_SHARPE = "maximum_sharpe"  # Maximum Sharpe ratio
    HIERARCHICAL_RISK_PARITY = "hierarchical_risk_parity"  # HRP


class RebalanceFrequency(str, Enum):
    """Portfolio rebalance frequency."""
    NEVER = "never"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


@dataclass
class PortfolioWeight:
    """Individual strategy weight in portfolio."""

    strategy_id: str
    weight: float

    # Performance metrics
    sharpe_ratio: float = 0.0
    annual_return: float = 0.0
    volatility: float = 0.0
    max_drawdown: float = 0.0

    # Contribution to portfolio
    contribution_to_return: float = 0.0
    contribution_to_risk: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy_id": self.strategy_id,
            "weight": round(self.weight, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "annual_return": round(self.annual_return, 2),
            "volatility": round(self.volatility, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "contribution_to_return": round(self.contribution_to_return, 2),
            "contribution_to_risk": round(self.contribution_to_risk, 2),
        }


@dataclass
class PortfolioConfig:
    """Portfolio construction configuration."""

    portfolio_id: str
    strategy_ids: List[str]

    # Weighting
    weighting_method: PortfolioWeightingMethod = PortfolioWeightingMethod.EQUAL
    min_weight: float = 0.05  # Minimum 5% per strategy
    max_weight: float = 0.50  # Maximum 50% per strategy

    # Rebalancing
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    rebalance_threshold: float = 0.05  # Rebalance if drift > 5%

    # Constraints
    max_strategies: int = 10  # Maximum strategies in portfolio
    min_correlation: float = -1.0  # Minimum correlation to include
    max_correlation: float = 0.95  # Maximum correlation to include

    # Risk limits
    max_portfolio_volatility: float = 0.20  # 20% max volatility
    max_portfolio_drawdown: float = 0.30  # 30% max drawdown

    # Optimization (for optimization-based methods)
    lookback_periods: int = 252  # Trading days for optimization
    risk_free_rate: float = 0.02  # 2% annual

    def validate(self) -> None:
        """Validate configuration."""
        if self.min_weight < 0 or self.min_weight > 1:
            raise ValueError("min_weight must be between 0 and 1")

        if self.max_weight < self.min_weight or self.max_weight > 1:
            raise ValueError("max_weight must be >= min_weight and <= 1")

        if len(self.strategy_ids) > self.max_strategies:
            raise ValueError(f"Too many strategies: {len(self.strategy_ids)} > {self.max_strategies}")

        if self.min_correlation < -1 or self.min_correlation > 1:
            raise ValueError("min_correlation must be between -1 and 1")


@dataclass
class CorrelationMatrix:
    """Strategy correlation analysis."""

    strategy_ids: List[str]
    correlation_matrix: pd.DataFrame

    # Summary statistics
    avg_correlation: float = 0.0
    min_correlation: float = 0.0
    max_correlation: float = 0.0

    # Diversification metrics
    diversification_ratio: float = 1.0  # Portfolio vol / weighted avg vol
    effective_n: float = 1.0  # Effective number of independent strategies

    def get_correlation(self, strategy1: str, strategy2: str) -> float:
        """Get correlation between two strategies."""
        return self.correlation_matrix.loc[strategy1, strategy2]

    def get_highly_correlated_pairs(self, threshold: float = 0.8) -> List[tuple[str, str, float]]:
        """Get pairs with correlation above threshold."""
        pairs = []
        n = len(self.strategy_ids)

        for i in range(n):
            for j in range(i + 1, n):
                sid1 = self.strategy_ids[i]
                sid2 = self.strategy_ids[j]
                corr = self.correlation_matrix.iloc[i, j]

                if abs(corr) >= threshold:
                    pairs.append((sid1, sid2, corr))

        return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy_ids": self.strategy_ids,
            "correlation_matrix": self.correlation_matrix.to_dict(),
            "avg_correlation": round(self.avg_correlation, 3),
            "min_correlation": round(self.min_correlation, 3),
            "max_correlation": round(self.max_correlation, 3),
            "diversification_ratio": round(self.diversification_ratio, 3),
            "effective_n": round(self.effective_n, 2),
        }


@dataclass
class Portfolio:
    """Optimized portfolio of strategies."""

    portfolio_id: str
    config: PortfolioConfig
    weights: List[PortfolioWeight]

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_rebalance: Optional[datetime] = None
    rebalance_count: int = 0

    # Performance tracking
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    volatility: float = 0.0
    max_drawdown: float = 0.0

    # Correlation analysis
    correlation_matrix: Optional[CorrelationMatrix] = None

    def get_weight(self, strategy_id: str) -> float:
        """Get weight for strategy."""
        for w in self.weights:
            if w.strategy_id == strategy_id:
                return w.weight
        return 0.0

    def get_weights_dict(self) -> Dict[str, float]:
        """Get weights as dictionary."""
        return {w.strategy_id: w.weight for w in self.weights}

    def needs_rebalance(self, current_weights: Dict[str, float]) -> bool:
        """Check if rebalancing is needed."""
        target_weights = self.get_weights_dict()

        for strategy_id, target_weight in target_weights.items():
            current_weight = current_weights.get(strategy_id, 0.0)
            drift = abs(current_weight - target_weight)

            if drift > self.config.rebalance_threshold:
                return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "portfolio_id": self.portfolio_id,
            "strategy_ids": self.config.strategy_ids,
            "weighting_method": self.config.weighting_method.value,
            "weights": [w.to_dict() for w in self.weights],
            "total_return": round(self.total_return, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "volatility": round(self.volatility, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "rebalance_count": self.rebalance_count,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class PortfolioBacktestResult:
    """Portfolio backtest result."""

    portfolio_id: str
    config: PortfolioConfig

    # Performance
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Risk
    volatility: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0

    # Diversification
    diversification_ratio: float = 1.0
    effective_n: float = 1.0
    avg_correlation: float = 0.0

    # Trading
    total_trades: int = 0
    rebalance_count: int = 0
    turnover: float = 0.0  # Average turnover per rebalance

    # Time series
    equity_curve: Optional[pd.Series] = None
    weights_over_time: Optional[pd.DataFrame] = None
    returns: Optional[pd.Series] = None
    drawdown_series: Optional[pd.Series] = None

    # Component performance
    component_weights: List[PortfolioWeight] = field(default_factory=list)
    component_contributions: Optional[pd.DataFrame] = None

    # Comparison to individual strategies
    best_individual_sharpe: float = 0.0
    portfolio_vs_best_improvement: float = 0.0

    # Timestamps
    backtest_start: Optional[datetime] = None
    backtest_end: Optional[datetime] = None

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary."""
        return {
            "portfolio_id": self.portfolio_id,
            "weighting_method": self.config.weighting_method.value,
            "num_strategies": len(self.config.strategy_ids),
            "total_return": round(self.total_return, 2),
            "annual_return": round(self.annual_return, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "sortino_ratio": round(self.sortino_ratio, 3),
            "volatility": round(self.volatility, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "diversification_ratio": round(self.diversification_ratio, 3),
            "effective_n": round(self.effective_n, 2),
            "total_trades": self.total_trades,
            "rebalance_count": self.rebalance_count,
            "portfolio_vs_best_improvement": round(self.portfolio_vs_best_improvement, 2),
        }


@dataclass
class PortfolioComparisonResult:
    """Comparison of multiple portfolio construction methods."""

    portfolio_results: List[PortfolioBacktestResult]

    # Best performers
    best_sharpe_method: str = ""
    best_return_method: str = ""
    best_risk_adjusted_method: str = ""

    # Comparison metrics
    performance_table: Optional[pd.DataFrame] = None

    def get_best_portfolio(self, metric: str = "sharpe_ratio") -> Optional[PortfolioBacktestResult]:
        """Get best portfolio by metric."""
        if not self.portfolio_results:
            return None

        return max(self.portfolio_results, key=lambda x: getattr(x, metric, 0))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_portfolios": len(self.portfolio_results),
            "best_sharpe_method": self.best_sharpe_method,
            "best_return_method": self.best_return_method,
            "best_risk_adjusted_method": self.best_risk_adjusted_method,
            "results": [r.to_summary_dict() for r in self.portfolio_results],
        }
