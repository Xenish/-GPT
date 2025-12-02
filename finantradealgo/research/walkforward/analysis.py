"""
Walk-Forward Analysis Tools.

Additional analysis and metrics for walk-forward results.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

import pandas as pd
import numpy as np

from finantradealgo.research.walkforward.models import (
    WalkForwardResult,
    WalkForwardWindow,
    WalkForwardComparison,
)


@dataclass
class EfficiencyMetrics:
    """
    Walk-forward efficiency metrics.

    Measures how well strategy translates from IS to OOS.
    """

    # Efficiency ratios (OOS / IS)
    sharpe_efficiency: float = 0.0  # OOS Sharpe / IS Sharpe
    return_efficiency: float = 0.0  # OOS Return / IS Return

    # Stability metrics
    sharpe_correlation: float = 0.0  # Correlation between IS and OOS Sharpe
    param_consistency: float = 0.0  # How consistent parameters are

    # Risk metrics
    oos_max_dd_avg: float = 0.0
    dd_increase_pct: float = 0.0  # How much DD increased from IS to OOS

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sharpe_efficiency": round(self.sharpe_efficiency, 3),
            "return_efficiency": round(self.return_efficiency, 3),
            "sharpe_correlation": round(self.sharpe_correlation, 3),
            "param_consistency": round(self.param_consistency, 3),
            "oos_max_dd_avg": round(self.oos_max_dd_avg, 2),
            "dd_increase_pct": round(self.dd_increase_pct, 2),
        }


class WalkForwardAnalyzer:
    """
    Analyzes walk-forward results.

    Provides detailed metrics and insights.
    """

    def __init__(self):
        """Initialize analyzer."""
        pass

    def calculate_efficiency_metrics(self, result: WalkForwardResult) -> EfficiencyMetrics:
        """
        Calculate efficiency metrics.

        Args:
            result: Walk-forward result

        Returns:
            EfficiencyMetrics with detailed analysis
        """
        if not result.windows:
            return EfficiencyMetrics()

        # Calculate efficiency ratios
        sharpe_eff = (
            result.avg_oos_sharpe / result.avg_is_sharpe
            if result.avg_is_sharpe != 0
            else 0.0
        )
        return_eff = (
            result.avg_oos_return / result.avg_is_return
            if result.avg_is_return != 0
            else 0.0
        )

        # Calculate IS-OOS correlations
        is_sharpes = [w.is_sharpe for w in result.windows]
        oos_sharpes = [w.oos_sharpe for w in result.windows]

        sharpe_corr = 0.0
        if len(is_sharpes) > 1:
            sharpe_corr = np.corrcoef(is_sharpes, oos_sharpes)[0, 1]
            if np.isnan(sharpe_corr):
                sharpe_corr = 0.0

        # Drawdown analysis
        oos_dds = [w.oos_max_drawdown for w in result.windows]
        is_dds = [w.is_max_drawdown for w in result.windows]

        oos_dd_avg = np.mean(oos_dds) if oos_dds else 0.0
        is_dd_avg = np.mean(is_dds) if is_dds else 0.0

        dd_increase = (
            ((oos_dd_avg - is_dd_avg) / abs(is_dd_avg)) * 100
            if is_dd_avg != 0
            else 0.0
        )

        return EfficiencyMetrics(
            sharpe_efficiency=sharpe_eff,
            return_efficiency=return_eff,
            sharpe_correlation=sharpe_corr,
            param_consistency=result.param_stability_score / 100.0,
            oos_max_dd_avg=oos_dd_avg,
            dd_increase_pct=dd_increase,
        )

    def identify_regime_sensitivity(
        self, result: WalkForwardResult
    ) -> Dict[str, Any]:
        """
        Identify if strategy is sensitive to specific market regimes.

        Args:
            result: Walk-forward result

        Returns:
            Regime sensitivity analysis
        """
        if not result.windows:
            return {}

        # Group windows by performance quartiles
        oos_returns = [w.oos_total_return for w in result.windows]
        q1 = np.percentile(oos_returns, 25)
        q3 = np.percentile(oos_returns, 75)

        best_windows = [w for w in result.windows if w.oos_total_return >= q3]
        worst_windows = [w for w in result.windows if w.oos_total_return <= q1]

        # Analyze characteristics
        def analyze_window_group(windows: List[WalkForwardWindow]) -> Dict[str, float]:
            if not windows:
                return {}

            return {
                "avg_oos_sharpe": np.mean([w.oos_sharpe for w in windows]),
                "avg_oos_return": np.mean([w.oos_total_return for w in windows]),
                "avg_win_rate": np.mean([w.oos_win_rate for w in windows]),
                "avg_trades": np.mean([w.oos_total_trades for w in windows]),
            }

        return {
            "best_quartile": {
                "count": len(best_windows),
                "characteristics": analyze_window_group(best_windows),
            },
            "worst_quartile": {
                "count": len(worst_windows),
                "characteristics": analyze_window_group(worst_windows),
            },
            "performance_range": {
                "min_oos_return": min(oos_returns),
                "max_oos_return": max(oos_returns),
                "range": max(oos_returns) - min(oos_returns),
                "iqr": q3 - q1,
            },
        }

    def analyze_parameter_drift(self, result: WalkForwardResult) -> Dict[str, Any]:
        """
        Analyze how parameters change across windows.

        Args:
            result: Walk-forward result

        Returns:
            Parameter drift analysis
        """
        if not result.windows:
            return {}

        # Collect parameters across windows
        param_history = {}
        for window in result.windows:
            for param_name, param_value in window.best_params.items():
                if param_name not in param_history:
                    param_history[param_name] = []
                param_history[param_name].append(param_value)

        # Analyze each parameter
        drift_analysis = {}
        for param_name, values in param_history.items():
            # Only analyze numeric parameters
            numeric_values = [v for v in values if isinstance(v, (int, float))]

            if len(numeric_values) > 1:
                drift_analysis[param_name] = {
                    "mean": np.mean(numeric_values),
                    "std": np.std(numeric_values),
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "range": max(numeric_values) - min(numeric_values),
                    "cv": np.std(numeric_values) / np.mean(numeric_values)
                    if np.mean(numeric_values) != 0
                    else 0,
                    "trend": self._calculate_trend(numeric_values),
                }

        return drift_analysis

    def calculate_walk_forward_sharpe(self, result: WalkForwardResult) -> float:
        """
        Calculate overall walk-forward Sharpe ratio.

        Uses only OOS returns for true out-of-sample performance.

        Args:
            result: Walk-forward result

        Returns:
            Walk-forward Sharpe ratio
        """
        if not result.windows or not result.combined_equity_curve:
            return 0.0

        # Calculate returns from combined OOS equity curve
        returns = result.combined_equity_curve.pct_change().dropna()

        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        # Annualized Sharpe
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

        return sharpe

    def generate_summary_report(self, result: WalkForwardResult) -> Dict[str, Any]:
        """
        Generate comprehensive summary report.

        Args:
            result: Walk-forward result

        Returns:
            Dictionary with complete analysis
        """
        efficiency = self.calculate_efficiency_metrics(result)
        regime_sensitivity = self.identify_regime_sensitivity(result)
        param_drift = self.analyze_parameter_drift(result)
        wf_sharpe = self.calculate_walk_forward_sharpe(result)

        # Window-by-window details
        window_details = []
        for w in result.windows:
            window_details.append({
                "window_id": w.window_id,
                "period": f"{w.out_sample_start.date()} to {w.out_sample_end.date()}",
                "is_sharpe": round(w.is_sharpe, 2),
                "oos_sharpe": round(w.oos_sharpe, 2),
                "degradation": round(w.sharpe_degradation, 3),
                "oos_return": round(w.oos_total_return, 2),
                "best_params": w.best_params,
            })

        return {
            "strategy_id": result.strategy_id,
            "overall_metrics": result.to_summary_dict(),
            "walk_forward_sharpe": round(wf_sharpe, 3),
            "efficiency_metrics": efficiency.to_dict(),
            "regime_sensitivity": regime_sensitivity,
            "parameter_drift": param_drift,
            "window_details": window_details,
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction."""
        if len(values) < 3:
            return "insufficient_data"

        x = np.arange(len(values))
        y = np.array(values)

        slope = np.polyfit(x, y, 1)[0]

        if abs(slope) < 0.01:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"

    def export_to_dataframe(self, result: WalkForwardResult) -> pd.DataFrame:
        """
        Export walk-forward results to DataFrame.

        Args:
            result: Walk-forward result

        Returns:
            DataFrame with all window results
        """
        data = []
        for window in result.windows:
            data.append({
                "window_id": window.window_id,
                "is_start": window.in_sample_start,
                "is_end": window.in_sample_end,
                "oos_start": window.out_sample_start,
                "oos_end": window.out_sample_end,
                # IS metrics
                "is_sharpe": window.is_sharpe,
                "is_return": window.is_total_return,
                "is_max_dd": window.is_max_drawdown,
                "is_win_rate": window.is_win_rate,
                "is_trades": window.is_total_trades,
                # OOS metrics
                "oos_sharpe": window.oos_sharpe,
                "oos_return": window.oos_total_return,
                "oos_max_dd": window.oos_max_drawdown,
                "oos_win_rate": window.oos_win_rate,
                "oos_trades": window.oos_total_trades,
                # Degradation
                "sharpe_degradation": window.sharpe_degradation,
                "return_degradation": window.return_degradation,
            })

        return pd.DataFrame(data)

    def compare_strategies(
        self, results: List[WalkForwardResult]
    ) -> pd.DataFrame:
        """
        Compare multiple walk-forward results.

        Args:
            results: List of walk-forward results

        Returns:
            Comparison DataFrame
        """
        comparison_data = []

        for result in results:
            efficiency = self.calculate_efficiency_metrics(result)
            wf_sharpe = self.calculate_walk_forward_sharpe(result)

            comparison_data.append({
                "strategy_id": result.strategy_id,
                "total_windows": result.total_windows,
                "avg_oos_sharpe": result.avg_oos_sharpe,
                "avg_oos_return": result.avg_oos_return,
                "oos_win_rate": result.oos_win_rate,
                "sharpe_degradation": result.avg_sharpe_degradation,
                "consistency_score": result.consistency_score,
                "param_stability": result.param_stability_score,
                "sharpe_efficiency": efficiency.sharpe_efficiency,
                "wf_sharpe": wf_sharpe,
            })

        df = pd.DataFrame(comparison_data)

        # Sort by consistency score
        df = df.sort_values("consistency_score", ascending=False)

        return df
