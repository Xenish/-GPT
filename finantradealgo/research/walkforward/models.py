"""
Walk-Forward Optimization Models.

Data structures for walk-forward analysis and validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Any

import pandas as pd


class WindowType(str, Enum):
    """Walk-forward window type."""

    ROLLING = "rolling"  # Fixed size window that rolls forward
    ANCHORED = "anchored"  # Expanding window anchored to start
    EXPANDING = "expanding"  # Alias for anchored


class OptimizationMetric(str, Enum):
    """Metric to optimize in walk-forward."""

    SHARPE_RATIO = "sharpe_ratio"
    TOTAL_RETURN = "total_return"
    PROFIT_FACTOR = "profit_factor"
    WIN_RATE = "win_rate"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"


@dataclass
class WalkForwardConfig:
    """
    Walk-forward optimization configuration.

    Defines how to split data into in-sample and out-of-sample periods.
    """

    # Window configuration
    in_sample_periods: int = field(
        default=12,
        metadata={"description": "Number of periods for in-sample optimization (e.g., 12 months)"},
    )
    out_sample_periods: int = field(
        default=3,
        metadata={"description": "Number of periods for out-of-sample testing (e.g., 3 months)"},
    )
    window_type: WindowType = field(
        default=WindowType.ROLLING,
        metadata={"description": "Type of window (rolling or anchored)"},
    )

    # Period definition
    period_unit: str = field(
        default="M",
        metadata={
            "description": "Period unit: 'D' (day), 'W' (week), 'M' (month), 'Q' (quarter), 'Y' (year)"
        },
    )

    # Optimization
    optimization_metric: OptimizationMetric = field(
        default=OptimizationMetric.SHARPE_RATIO,
        metadata={"description": "Metric to optimize during in-sample periods"},
    )

    # Constraints
    min_trades_per_period: int = field(
        default=10,
        metadata={"description": "Minimum trades required per period"},
    )
    require_profitable_is: bool = field(
        default=False,
        metadata={"description": "Require in-sample to be profitable"},
    )

    def __post_init__(self):
        """Validate configuration."""
        if self.in_sample_periods <= 0:
            raise ValueError("in_sample_periods must be positive")
        if self.out_sample_periods <= 0:
            raise ValueError("out_sample_periods must be positive")
        if self.period_unit not in ["D", "W", "M", "Q", "Y"]:
            raise ValueError(f"Invalid period_unit: {self.period_unit}")


@dataclass
class WalkForwardWindow:
    """
    Single walk-forward window.

    Represents one iteration of the walk-forward process.
    """

    window_id: int
    in_sample_start: datetime
    in_sample_end: datetime
    out_sample_start: datetime
    out_sample_end: datetime

    # Best parameters from in-sample optimization
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_is_score: float = 0.0

    # In-sample results
    is_total_trades: int = 0
    is_sharpe: float = 0.0
    is_total_return: float = 0.0
    is_max_drawdown: float = 0.0
    is_win_rate: float = 0.0

    # Out-of-sample results
    oos_total_trades: int = 0
    oos_sharpe: float = 0.0
    oos_total_return: float = 0.0
    oos_max_drawdown: float = 0.0
    oos_win_rate: float = 0.0

    # Performance comparison
    sharpe_degradation: float = 0.0  # (IS - OOS) / IS
    return_degradation: float = 0.0

    def calculate_degradation(self):
        """Calculate performance degradation from IS to OOS."""
        if self.is_sharpe != 0:
            self.sharpe_degradation = (self.is_sharpe - self.oos_sharpe) / abs(self.is_sharpe)
        else:
            self.sharpe_degradation = 0.0

        if self.is_total_return != 0:
            self.return_degradation = (
                self.is_total_return - self.oos_total_return
            ) / abs(self.is_total_return)
        else:
            self.return_degradation = 0.0


@dataclass
class WalkForwardResult:
    """
    Complete walk-forward optimization result.

    Contains all windows and aggregate statistics.
    """

    strategy_id: str
    config: WalkForwardConfig
    windows: List[WalkForwardWindow] = field(default_factory=list)

    # Aggregate metrics
    total_windows: int = 0
    avg_is_sharpe: float = 0.0
    avg_oos_sharpe: float = 0.0
    avg_sharpe_degradation: float = 0.0

    avg_is_return: float = 0.0
    avg_oos_return: float = 0.0
    avg_return_degradation: float = 0.0

    # Consistency metrics
    oos_win_rate: float = 0.0  # % of OOS periods that were profitable
    consistency_score: float = 0.0  # Based on degradation and OOS performance

    # Parameter stability
    param_stability_score: float = 0.0  # How stable parameters are across windows

    # Combined equity curve
    combined_equity_curve: Optional[pd.Series] = None

    # Timestamps
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_duration_seconds: float = 0.0

    def calculate_aggregate_metrics(self):
        """Calculate aggregate metrics from all windows."""
        if not self.windows:
            return

        self.total_windows = len(self.windows)

        # Average metrics
        self.avg_is_sharpe = sum(w.is_sharpe for w in self.windows) / self.total_windows
        self.avg_oos_sharpe = sum(w.oos_sharpe for w in self.windows) / self.total_windows
        self.avg_sharpe_degradation = (
            sum(w.sharpe_degradation for w in self.windows) / self.total_windows
        )

        self.avg_is_return = sum(w.is_total_return for w in self.windows) / self.total_windows
        self.avg_oos_return = sum(w.oos_total_return for w in self.windows) / self.total_windows
        self.avg_return_degradation = (
            sum(w.return_degradation for w in self.windows) / self.total_windows
        )

        # OOS win rate (% of profitable OOS periods)
        profitable_oos = sum(1 for w in self.windows if w.oos_total_return > 0)
        self.oos_win_rate = profitable_oos / self.total_windows

        # Consistency score (0-100)
        # Higher score = low degradation + high OOS win rate + positive OOS Sharpe
        degradation_penalty = max(0, 1 - abs(self.avg_sharpe_degradation))
        sharpe_bonus = min(1, max(0, self.avg_oos_sharpe / 2))  # Cap at Sharpe=2
        self.consistency_score = (
            degradation_penalty * 0.4 + self.oos_win_rate * 0.4 + sharpe_bonus * 0.2
        ) * 100

        # Parameter stability (calculated separately)
        self.param_stability_score = self._calculate_param_stability()

    def _calculate_param_stability(self) -> float:
        """
        Calculate parameter stability score.

        Measures how consistent parameter choices are across windows.
        Returns score 0-100 where higher = more stable.
        """
        if len(self.windows) < 2:
            return 100.0

        # Collect all parameter names
        param_names = set()
        for window in self.windows:
            param_names.update(window.best_params.keys())

        if not param_names:
            return 100.0

        # Calculate coefficient of variation for each parameter
        stabilities = []
        for param_name in param_names:
            values = []
            for window in self.windows:
                if param_name in window.best_params:
                    val = window.best_params[param_name]
                    # Handle numeric values only
                    if isinstance(val, (int, float)):
                        values.append(val)

            if len(values) > 1:
                mean_val = sum(values) / len(values)
                if mean_val != 0:
                    std_val = (sum((x - mean_val) ** 2 for x in values) / len(values)) ** 0.5
                    cv = std_val / abs(mean_val)  # Coefficient of variation
                    # Convert to stability score (lower CV = higher stability)
                    stability = max(0, 100 * (1 - min(cv, 1)))
                    stabilities.append(stability)

        if stabilities:
            return sum(stabilities) / len(stabilities)
        else:
            return 100.0

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary."""
        return {
            "strategy_id": self.strategy_id,
            "total_windows": self.total_windows,
            "window_type": self.config.window_type.value,
            "in_sample_periods": self.config.in_sample_periods,
            "out_sample_periods": self.config.out_sample_periods,
            # In-sample
            "avg_is_sharpe": round(self.avg_is_sharpe, 3),
            "avg_is_return": round(self.avg_is_return, 2),
            # Out-of-sample
            "avg_oos_sharpe": round(self.avg_oos_sharpe, 3),
            "avg_oos_return": round(self.avg_oos_return, 2),
            "oos_win_rate": round(self.oos_win_rate, 3),
            # Degradation
            "avg_sharpe_degradation": round(self.avg_sharpe_degradation, 3),
            "avg_return_degradation": round(self.avg_return_degradation, 3),
            # Scores
            "consistency_score": round(self.consistency_score, 1),
            "param_stability_score": round(self.param_stability_score, 1),
            # Timing
            "total_duration_seconds": round(self.total_duration_seconds, 1),
        }


@dataclass
class WalkForwardComparison:
    """
    Comparison of multiple walk-forward results.

    Used to compare different strategies or parameter sets.
    """

    results: List[WalkForwardResult] = field(default_factory=list)
    comparison_metric: str = "consistency_score"

    def rank_by_metric(self, metric: str = "consistency_score") -> List[tuple[str, float]]:
        """
        Rank strategies by a specific metric.

        Args:
            metric: Metric to rank by

        Returns:
            List of (strategy_id, metric_value) tuples sorted by metric
        """
        rankings = []
        for result in self.results:
            value = getattr(result, metric, 0.0)
            rankings.append((result.strategy_id, value))

        # Sort descending (higher is better for most metrics)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def get_best_strategy(self, metric: str = "consistency_score") -> Optional[WalkForwardResult]:
        """Get best strategy by metric."""
        if not self.results:
            return None

        rankings = self.rank_by_metric(metric)
        best_id = rankings[0][0]

        for result in self.results:
            if result.strategy_id == best_id:
                return result

        return None
