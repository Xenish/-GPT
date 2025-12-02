"""
Risk Metrics Calculator.

Advanced risk metrics and analysis for Monte Carlo results.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np

from finantradealgo.research.montecarlo.models import MonteCarloResult


@dataclass
class RiskAssessment:
    """
    Comprehensive risk assessment.

    Detailed risk metrics and analysis.
    """

    # Value at Risk metrics
    var_95: float  # 95% VaR
    var_99: float  # 99% VaR
    cvar_95: float  # 95% CVaR
    cvar_99: float  # 99% CVaR

    # Tail risk
    expected_shortfall: float
    tail_ratio: float  # Ratio of gains to losses in tails

    # Drawdown risk
    max_dd_mean: float
    max_dd_95: float  # 95th percentile worst drawdown
    max_dd_99: float  # 99th percentile worst drawdown

    # Return distribution
    return_mean: float
    return_median: float
    return_std: float
    return_skew: float
    return_kurtosis: float

    # Probability metrics
    prob_ruin: float  # Probability of catastrophic loss
    prob_target: float  # Probability of reaching target return

    # Risk-adjusted returns
    sharpe_mean: float
    sortino_mean: float
    calmar_mean: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "var": {
                "95pct": round(self.var_95, 2),
                "99pct": round(self.var_99, 2),
            },
            "cvar": {
                "95pct": round(self.cvar_95, 2),
                "99pct": round(self.cvar_99, 2),
            },
            "tail_risk": {
                "expected_shortfall": round(self.expected_shortfall, 2),
                "tail_ratio": round(self.tail_ratio, 3),
            },
            "drawdown": {
                "mean": round(self.max_dd_mean, 2),
                "95pct": round(self.max_dd_95, 2),
                "99pct": round(self.max_dd_99, 2),
            },
            "return_distribution": {
                "mean": round(self.return_mean, 2),
                "median": round(self.return_median, 2),
                "std": round(self.return_std, 2),
                "skew": round(self.return_skew, 3),
                "kurtosis": round(self.return_kurtosis, 3),
            },
            "probabilities": {
                "ruin": round(self.prob_ruin, 4),
                "target": round(self.prob_target, 4),
            },
            "risk_adjusted": {
                "sharpe": round(self.sharpe_mean, 3),
                "sortino": round(self.sortino_mean, 3),
                "calmar": round(self.calmar_mean, 3),
            },
        }


class RiskMetricsCalculator:
    """
    Calculate advanced risk metrics from Monte Carlo results.

    Provides comprehensive risk analysis beyond basic statistics.
    """

    def __init__(self):
        """Initialize risk metrics calculator."""
        pass

    def calculate_risk_assessment(
        self,
        result: MonteCarloResult,
        ruin_threshold: float = -50.0,  # -50% considered ruin
        target_return: float = 20.0,  # 20% target return
    ) -> RiskAssessment:
        """
        Calculate comprehensive risk assessment.

        Args:
            result: Monte Carlo result
            ruin_threshold: Return threshold considered catastrophic
            target_return: Target return for probability calculation

        Returns:
            RiskAssessment with all metrics
        """
        # Extract data
        returns = np.array([s.total_return for s in result.simulations])
        sharpes = np.array([s.sharpe_ratio for s in result.simulations])
        sortinos = np.array([s.sortino_ratio for s in result.simulations])
        max_dds = np.array([s.max_drawdown for s in result.simulations])

        # VaR and CVaR at multiple confidence levels
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)

        below_var_95 = returns[returns <= var_95]
        cvar_95 = np.mean(below_var_95) if len(below_var_95) > 0 else var_95

        below_var_99 = returns[returns <= var_99]
        cvar_99 = np.mean(below_var_99) if len(below_var_99) > 0 else var_99

        # Expected shortfall (average of worst 5%)
        worst_5pct = np.percentile(returns, 5)
        expected_shortfall = np.mean(returns[returns <= worst_5pct])

        # Tail ratio (average of best 5% / average of worst 5%)
        best_5pct = np.percentile(returns, 95)
        avg_best = np.mean(returns[returns >= best_5pct])
        avg_worst = np.mean(returns[returns <= worst_5pct])
        tail_ratio = abs(avg_best / avg_worst) if avg_worst != 0 else 0

        # Drawdown statistics
        max_dd_mean = np.mean(max_dds)
        max_dd_95 = np.percentile(max_dds, 95)  # 95% worst case
        max_dd_99 = np.percentile(max_dds, 99)  # 99% worst case

        # Return distribution
        return_mean = np.mean(returns)
        return_median = np.median(returns)
        return_std = np.std(returns)
        return_skew = self._calculate_skewness(returns)
        return_kurtosis = self._calculate_kurtosis(returns)

        # Probability metrics
        prob_ruin = np.sum(returns < ruin_threshold) / len(returns)
        prob_target = np.sum(returns > target_return) / len(returns)

        # Risk-adjusted returns
        sharpe_mean = np.mean(sharpes)
        sortino_mean = np.mean(sortinos)

        # Calmar ratio (return / max drawdown)
        calmar_ratios = []
        for s in result.simulations:
            if s.max_drawdown != 0:
                calmar = abs(s.total_return / s.max_drawdown)
                calmar_ratios.append(calmar)

        calmar_mean = np.mean(calmar_ratios) if calmar_ratios else 0.0

        return RiskAssessment(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            expected_shortfall=expected_shortfall,
            tail_ratio=tail_ratio,
            max_dd_mean=max_dd_mean,
            max_dd_95=max_dd_95,
            max_dd_99=max_dd_99,
            return_mean=return_mean,
            return_median=return_median,
            return_std=return_std,
            return_skew=return_skew,
            return_kurtosis=return_kurtosis,
            prob_ruin=prob_ruin,
            prob_target=prob_target,
            sharpe_mean=sharpe_mean,
            sortino_mean=sortino_mean,
            calmar_mean=calmar_mean,
        )

    def calculate_optimal_position_size(
        self,
        result: MonteCarloResult,
        max_risk_pct: float = 2.0,  # Max 2% risk per trade
        confidence_level: float = 0.95,
    ) -> Dict[str, float]:
        """
        Calculate optimal position sizing based on Monte Carlo results.

        Uses Kelly Criterion and risk-based approaches.

        Args:
            result: Monte Carlo result
            max_risk_pct: Maximum risk per trade as percentage
            confidence_level: Confidence level for risk metrics

        Returns:
            Dictionary with position sizing recommendations
        """
        returns = np.array([s.total_return for s in result.simulations])
        win_rates = np.array([s.win_rate for s in result.simulations])

        # Average metrics
        avg_return = np.mean(returns)
        avg_win_rate = np.mean(win_rates)

        # Kelly Criterion (simplified)
        # f = (p * b - q) / b where p=win_rate, q=loss_rate, b=win/loss ratio
        avg_wins = []
        avg_losses = []

        for sim in result.simulations:
            if sim.equity_curve is not None:
                pnl = sim.equity_curve.diff().dropna()
                wins = pnl[pnl > 0]
                losses = pnl[pnl < 0]

                if len(wins) > 0:
                    avg_wins.append(wins.mean())
                if len(losses) > 0:
                    avg_losses.append(abs(losses.mean()))

        if avg_wins and avg_losses:
            avg_win = np.mean(avg_wins)
            avg_loss = np.mean(avg_losses)
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1

            kelly_fraction = (avg_win_rate * win_loss_ratio - (1 - avg_win_rate)) / win_loss_ratio
            kelly_fraction = max(0, min(kelly_fraction, 1))  # Clamp to [0, 1]

            # Half Kelly (more conservative)
            half_kelly = kelly_fraction / 2
        else:
            kelly_fraction = 0
            half_kelly = 0

        # VaR-based position sizing
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(returns, var_percentile)

        # Position size to limit VaR to max_risk_pct
        if var < 0:
            var_position_size = max_risk_pct / abs(var)
        else:
            var_position_size = 1.0

        # Fixed fractional (based on max drawdown)
        max_dds = np.array([s.max_drawdown for s in result.simulations])
        worst_dd = np.percentile(max_dds, 95)  # 95th percentile worst case

        if worst_dd < 0:
            fixed_fraction = max_risk_pct / abs(worst_dd)
        else:
            fixed_fraction = 1.0

        return {
            "kelly_fraction": round(kelly_fraction, 4),
            "half_kelly": round(half_kelly, 4),
            "var_based": round(var_position_size, 4),
            "fixed_fractional": round(fixed_fraction, 4),
            "recommended": round(min(half_kelly, var_position_size, fixed_fraction, 1.0), 4),
        }

    def analyze_tail_risk(
        self,
        result: MonteCarloResult,
        tail_percentile: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Analyze tail risk (extreme outcomes).

        Args:
            result: Monte Carlo result
            tail_percentile: Percentile defining tail (default 5%)

        Returns:
            Tail risk analysis
        """
        returns = np.array([s.total_return for s in result.simulations])

        # Left tail (losses)
        left_tail_threshold = np.percentile(returns, tail_percentile)
        left_tail = returns[returns <= left_tail_threshold]

        # Right tail (gains)
        right_tail_threshold = np.percentile(returns, 100 - tail_percentile)
        right_tail = returns[returns >= right_tail_threshold]

        return {
            "left_tail": {
                "threshold": round(left_tail_threshold, 2),
                "mean": round(np.mean(left_tail), 2),
                "min": round(np.min(left_tail), 2),
                "count": len(left_tail),
            },
            "right_tail": {
                "threshold": round(right_tail_threshold, 2),
                "mean": round(np.mean(right_tail), 2),
                "max": round(np.max(right_tail), 2),
                "count": len(right_tail),
            },
            "tail_ratio": round(np.mean(right_tail) / abs(np.mean(left_tail)), 3)
            if np.mean(left_tail) != 0
            else 0,
            "asymmetry": round(np.mean(right_tail) + np.mean(left_tail), 2),
        }

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
        kurt = np.sum(((data - mean) / std) ** 4) / n - 3

        return kurt
