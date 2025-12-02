"""
Monte Carlo Simulation Models.

Data structures for Monte Carlo risk analysis and simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Any

import pandas as pd
import numpy as np


class ResamplingMethod(str, Enum):
    """Resampling method for Monte Carlo."""

    BOOTSTRAP = "bootstrap"  # Random sampling with replacement
    BLOCK_BOOTSTRAP = "block_bootstrap"  # Block sampling for dependent data
    SHUFFLE = "shuffle"  # Random shuffle of original data
    PARAMETRIC = "parametric"  # Parametric distribution fitting


class RiskMetric(str, Enum):
    """Risk metric types."""

    VALUE_AT_RISK = "var"  # Value at Risk
    CONDITIONAL_VAR = "cvar"  # Conditional VaR (Expected Shortfall)
    MAX_DRAWDOWN = "max_drawdown"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"


@dataclass
class MonteCarloConfig:
    """
    Monte Carlo simulation configuration.

    Controls simulation parameters and risk metrics.
    """

    # Simulation parameters
    n_simulations: int = field(
        default=1000,
        metadata={"description": "Number of Monte Carlo simulations to run"},
    )
    resampling_method: ResamplingMethod = field(
        default=ResamplingMethod.BOOTSTRAP,
        metadata={"description": "Resampling method"},
    )

    # Block bootstrap parameters
    block_size: int = field(
        default=10,
        metadata={"description": "Block size for block bootstrap"},
    )

    # Risk metrics
    confidence_level: float = field(
        default=0.95,
        metadata={"description": "Confidence level for VaR/CVaR (0.95 = 95%)"},
    )

    # Constraints
    min_trades_per_sim: int = field(
        default=10,
        metadata={"description": "Minimum trades required per simulation"},
    )

    # Random seed for reproducibility
    random_seed: Optional[int] = field(
        default=None,
        metadata={"description": "Random seed for reproducibility"},
    )

    def __post_init__(self):
        """Validate configuration."""
        if self.n_simulations <= 0:
            raise ValueError("n_simulations must be positive")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")


@dataclass
class SimulationResult:
    """
    Single Monte Carlo simulation result.

    Results from one simulation run.
    """

    simulation_id: int
    trades_sampled: int
    total_pnl: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float

    # Drawdown details
    max_drawdown_duration: int = 0  # In number of trades

    # Trade sequence
    equity_curve: Optional[pd.Series] = None


@dataclass
class MonteCarloResult:
    """
    Complete Monte Carlo analysis result.

    Contains all simulations and aggregate risk metrics.
    """

    strategy_id: str
    config: MonteCarloConfig
    simulations: List[SimulationResult] = field(default_factory=list)

    # Aggregate statistics
    n_simulations: int = 0
    mean_return: float = 0.0
    median_return: float = 0.0
    std_return: float = 0.0

    # Confidence intervals
    return_ci_lower: float = 0.0  # Lower bound at confidence level
    return_ci_upper: float = 0.0  # Upper bound
    sharpe_ci_lower: float = 0.0
    sharpe_ci_upper: float = 0.0

    # Risk metrics
    value_at_risk: float = 0.0  # VaR at confidence level
    conditional_var: float = 0.0  # CVaR (Expected Shortfall)
    expected_shortfall: float = 0.0  # Alias for CVaR

    # Percentiles
    percentile_1: float = 0.0  # 1st percentile (worst case)
    percentile_5: float = 0.0  # 5th percentile
    percentile_25: float = 0.0  # Q1
    percentile_75: float = 0.0  # Q3
    percentile_95: float = 0.0  # 95th percentile
    percentile_99: float = 0.0  # 99th percentile (best case)

    # Probability metrics
    prob_profit: float = 0.0  # Probability of being profitable
    prob_loss_exceeds_10pct: float = 0.0  # Prob of losing > 10%
    prob_loss_exceeds_20pct: float = 0.0  # Prob of losing > 20%

    # Distribution shape
    skewness: float = 0.0
    kurtosis: float = 0.0

    # Timestamps
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_duration_seconds: float = 0.0

    def calculate_statistics(self):
        """Calculate aggregate statistics from all simulations."""
        if not self.simulations:
            return

        self.n_simulations = len(self.simulations)

        # Extract returns
        returns = np.array([s.total_return for s in self.simulations])
        sharpes = np.array([s.sharpe_ratio for s in self.simulations])

        # Basic statistics
        self.mean_return = float(np.mean(returns))
        self.median_return = float(np.median(returns))
        self.std_return = float(np.std(returns))

        # Confidence intervals
        alpha = 1 - self.config.confidence_level
        self.return_ci_lower = float(np.percentile(returns, alpha / 2 * 100))
        self.return_ci_upper = float(np.percentile(returns, (1 - alpha / 2) * 100))
        self.sharpe_ci_lower = float(np.percentile(sharpes, alpha / 2 * 100))
        self.sharpe_ci_upper = float(np.percentile(sharpes, (1 - alpha / 2) * 100))

        # VaR and CVaR
        var_percentile = (1 - self.config.confidence_level) * 100
        self.value_at_risk = float(np.percentile(returns, var_percentile))

        # CVaR = Expected value of returns below VaR
        below_var = returns[returns <= self.value_at_risk]
        if len(below_var) > 0:
            self.conditional_var = float(np.mean(below_var))
            self.expected_shortfall = self.conditional_var
        else:
            self.conditional_var = self.value_at_risk
            self.expected_shortfall = self.value_at_risk

        # Percentiles
        self.percentile_1 = float(np.percentile(returns, 1))
        self.percentile_5 = float(np.percentile(returns, 5))
        self.percentile_25 = float(np.percentile(returns, 25))
        self.percentile_75 = float(np.percentile(returns, 75))
        self.percentile_95 = float(np.percentile(returns, 95))
        self.percentile_99 = float(np.percentile(returns, 99))

        # Probability metrics
        self.prob_profit = float(np.sum(returns > 0) / len(returns))
        self.prob_loss_exceeds_10pct = float(np.sum(returns < -10) / len(returns))
        self.prob_loss_exceeds_20pct = float(np.sum(returns < -20) / len(returns))

        # Distribution shape
        self.skewness = float(self._calculate_skewness(returns))
        self.kurtosis = float(self._calculate_kurtosis(returns))

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness."""
        if len(data) < 3:
            return 0.0

        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return 0.0

        n = len(data)
        skew = np.sum(((data - mean) / std) ** 3) / n

        return skew

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate excess kurtosis."""
        if len(data) < 4:
            return 0.0

        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return 0.0

        n = len(data)
        kurt = np.sum(((data - mean) / std) ** 4) / n - 3  # Excess kurtosis

        return kurt

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary."""
        return {
            "strategy_id": self.strategy_id,
            "n_simulations": self.n_simulations,
            "confidence_level": self.config.confidence_level,
            # Returns
            "mean_return": round(self.mean_return, 2),
            "median_return": round(self.median_return, 2),
            "std_return": round(self.std_return, 2),
            # Confidence intervals
            "return_ci": [round(self.return_ci_lower, 2), round(self.return_ci_upper, 2)],
            "sharpe_ci": [round(self.sharpe_ci_lower, 3), round(self.sharpe_ci_upper, 3)],
            # Risk metrics
            "var": round(self.value_at_risk, 2),
            "cvar": round(self.conditional_var, 2),
            # Probabilities
            "prob_profit": round(self.prob_profit, 3),
            "prob_loss_10pct": round(self.prob_loss_exceeds_10pct, 3),
            "prob_loss_20pct": round(self.prob_loss_exceeds_20pct, 3),
            # Distribution
            "skewness": round(self.skewness, 3),
            "kurtosis": round(self.kurtosis, 3),
            # Timing
            "total_duration_seconds": round(self.total_duration_seconds, 1),
        }


@dataclass
class StressTestScenario:
    """
    Stress test scenario definition.

    Defines market conditions to stress test strategy.
    """

    scenario_id: str
    name: str
    description: str

    # Market condition modifiers
    volatility_multiplier: float = 1.0  # Multiply volatility by this factor
    trend_adjustment: float = 0.0  # Add to returns (drift)
    drawdown_multiplier: float = 1.0  # Multiply drawdowns

    # Correlation shocks
    correlation_shock: Optional[float] = None  # Force correlation to this value

    def apply_to_returns(self, returns: pd.Series) -> pd.Series:
        """
        Apply stress scenario to return series.

        Args:
            returns: Original returns

        Returns:
            Stressed returns
        """
        stressed = returns.copy()

        # Apply volatility multiplier
        mean_return = stressed.mean()
        stressed = mean_return + (stressed - mean_return) * self.volatility_multiplier

        # Apply trend adjustment
        stressed = stressed + self.trend_adjustment

        return stressed


# Predefined stress scenarios
STRESS_SCENARIOS = {
    "high_volatility": StressTestScenario(
        scenario_id="high_vol",
        name="High Volatility",
        description="2x volatility with same mean",
        volatility_multiplier=2.0,
    ),
    "market_crash": StressTestScenario(
        scenario_id="crash",
        name="Market Crash",
        description="3x volatility with -5% drift",
        volatility_multiplier=3.0,
        trend_adjustment=-0.05,
    ),
    "low_volatility": StressTestScenario(
        scenario_id="low_vol",
        name="Low Volatility",
        description="0.5x volatility",
        volatility_multiplier=0.5,
    ),
    "trending_market": StressTestScenario(
        scenario_id="trending",
        name="Strong Trend",
        description="Strong upward drift",
        trend_adjustment=0.03,
    ),
    "ranging_market": StressTestScenario(
        scenario_id="ranging",
        name="Ranging Market",
        description="Low volatility, no trend",
        volatility_multiplier=0.6,
        trend_adjustment=0.0,
    ),
}
