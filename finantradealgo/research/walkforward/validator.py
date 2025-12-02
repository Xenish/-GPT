"""
Out-of-Sample Validator.

Validates and analyzes out-of-sample performance.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np

from finantradealgo.research.walkforward.models import (
    WalkForwardResult,
    WalkForwardWindow,
)


class ValidationStatus(str, Enum):
    """Validation status levels."""

    EXCELLENT = "excellent"  # Strong OOS performance, low degradation
    GOOD = "good"  # Acceptable OOS performance
    WARNING = "warning"  # Significant degradation or poor OOS
    FAILED = "failed"  # Strategy failed validation


@dataclass
class ValidationReport:
    """
    Validation report for walk-forward result.

    Assesses strategy robustness based on OOS performance.
    """

    status: ValidationStatus
    overall_score: float  # 0-100
    passed_checks: List[str]
    failed_checks: List[str]
    warnings: List[str]
    recommendations: List[str]

    # Detailed scores
    degradation_score: float = 0.0  # 0-100, higher is better
    consistency_score: float = 0.0
    oos_performance_score: float = 0.0
    stability_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "overall_score": round(self.overall_score, 1),
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "scores": {
                "degradation": round(self.degradation_score, 1),
                "consistency": round(self.consistency_score, 1),
                "oos_performance": round(self.oos_performance_score, 1),
                "stability": round(self.stability_score, 1),
            },
        }


class OutOfSampleValidator:
    """
    Validates walk-forward results for robustness.

    Checks various criteria to determine if strategy is production-ready.
    """

    def __init__(
        self,
        max_sharpe_degradation: float = 0.5,  # Max 50% degradation
        min_oos_win_rate: float = 0.4,  # At least 40% of OOS periods profitable
        min_oos_sharpe: float = 0.5,  # Minimum OOS Sharpe
        max_param_cv: float = 0.5,  # Max coefficient of variation for parameters
    ):
        """
        Initialize validator.

        Args:
            max_sharpe_degradation: Maximum acceptable Sharpe degradation
            min_oos_win_rate: Minimum OOS win rate (profitable periods)
            min_oos_sharpe: Minimum average OOS Sharpe ratio
            max_param_cv: Maximum parameter coefficient of variation
        """
        self.max_sharpe_degradation = max_sharpe_degradation
        self.min_oos_win_rate = min_oos_win_rate
        self.min_oos_sharpe = min_oos_sharpe
        self.max_param_cv = max_param_cv

    def validate(self, result: WalkForwardResult) -> ValidationReport:
        """
        Validate walk-forward result.

        Args:
            result: Walk-forward result to validate

        Returns:
            ValidationReport with detailed assessment
        """
        passed_checks = []
        failed_checks = []
        warnings = []
        recommendations = []

        # 1. Check Sharpe degradation
        degradation_score = self._check_degradation(
            result, passed_checks, failed_checks, warnings
        )

        # 2. Check OOS win rate
        consistency_score = self._check_consistency(
            result, passed_checks, failed_checks, warnings
        )

        # 3. Check OOS performance
        oos_performance_score = self._check_oos_performance(
            result, passed_checks, failed_checks, warnings
        )

        # 4. Check parameter stability
        stability_score = self._check_stability(
            result, passed_checks, failed_checks, warnings
        )

        # Calculate overall score
        overall_score = (
            degradation_score * 0.3
            + consistency_score * 0.3
            + oos_performance_score * 0.25
            + stability_score * 0.15
        )

        # Determine status
        if overall_score >= 80:
            status = ValidationStatus.EXCELLENT
        elif overall_score >= 60:
            status = ValidationStatus.GOOD
        elif overall_score >= 40:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.FAILED

        # Generate recommendations
        recommendations = self._generate_recommendations(
            result, degradation_score, consistency_score, oos_performance_score, stability_score
        )

        return ValidationReport(
            status=status,
            overall_score=overall_score,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warnings=warnings,
            recommendations=recommendations,
            degradation_score=degradation_score,
            consistency_score=consistency_score,
            oos_performance_score=oos_performance_score,
            stability_score=stability_score,
        )

    def _check_degradation(
        self,
        result: WalkForwardResult,
        passed: List[str],
        failed: List[str],
        warnings: List[str],
    ) -> float:
        """Check performance degradation from IS to OOS."""
        avg_degradation = abs(result.avg_sharpe_degradation)

        if avg_degradation <= self.max_sharpe_degradation * 0.5:
            # Excellent: < 25% degradation
            passed.append(f"Low Sharpe degradation: {avg_degradation:.1%}")
            score = 100
        elif avg_degradation <= self.max_sharpe_degradation:
            # Acceptable: 25-50% degradation
            passed.append(f"Acceptable Sharpe degradation: {avg_degradation:.1%}")
            score = 70
        elif avg_degradation <= self.max_sharpe_degradation * 1.5:
            # Warning: 50-75% degradation
            warnings.append(f"High Sharpe degradation: {avg_degradation:.1%}")
            score = 40
        else:
            # Failed: > 75% degradation
            failed.append(f"Excessive Sharpe degradation: {avg_degradation:.1%}")
            score = 10

        return score

    def _check_consistency(
        self,
        result: WalkForwardResult,
        passed: List[str],
        failed: List[str],
        warnings: List[str],
    ) -> float:
        """Check consistency of OOS performance."""
        oos_win_rate = result.oos_win_rate

        if oos_win_rate >= self.min_oos_win_rate * 1.5:
            # Excellent: > 60% of OOS periods profitable
            passed.append(f"High OOS win rate: {oos_win_rate:.1%}")
            score = 100
        elif oos_win_rate >= self.min_oos_win_rate:
            # Acceptable: 40-60%
            passed.append(f"Acceptable OOS win rate: {oos_win_rate:.1%}")
            score = 70
        elif oos_win_rate >= self.min_oos_win_rate * 0.75:
            # Warning: 30-40%
            warnings.append(f"Low OOS win rate: {oos_win_rate:.1%}")
            score = 40
        else:
            # Failed: < 30%
            failed.append(f"Very low OOS win rate: {oos_win_rate:.1%}")
            score = 10

        return score

    def _check_oos_performance(
        self,
        result: WalkForwardResult,
        passed: List[str],
        failed: List[str],
        warnings: List[str],
    ) -> float:
        """Check absolute OOS performance."""
        avg_oos_sharpe = result.avg_oos_sharpe

        if avg_oos_sharpe >= self.min_oos_sharpe * 2:
            # Excellent: Sharpe > 1.0
            passed.append(f"Strong OOS Sharpe: {avg_oos_sharpe:.2f}")
            score = 100
        elif avg_oos_sharpe >= self.min_oos_sharpe:
            # Acceptable: Sharpe 0.5-1.0
            passed.append(f"Acceptable OOS Sharpe: {avg_oos_sharpe:.2f}")
            score = 70
        elif avg_oos_sharpe >= 0:
            # Warning: Sharpe 0-0.5
            warnings.append(f"Low OOS Sharpe: {avg_oos_sharpe:.2f}")
            score = 40
        else:
            # Failed: Negative Sharpe
            failed.append(f"Negative OOS Sharpe: {avg_oos_sharpe:.2f}")
            score = 0

        return score

    def _check_stability(
        self,
        result: WalkForwardResult,
        passed: List[str],
        failed: List[str],
        warnings: List[str],
    ) -> float:
        """Check parameter stability across windows."""
        stability = result.param_stability_score

        if stability >= 80:
            passed.append(f"High parameter stability: {stability:.0f}/100")
            score = 100
        elif stability >= 60:
            passed.append(f"Acceptable parameter stability: {stability:.0f}/100")
            score = 70
        elif stability >= 40:
            warnings.append(f"Low parameter stability: {stability:.0f}/100")
            score = 40
        else:
            failed.append(f"Very low parameter stability: {stability:.0f}/100")
            score = 10

        return score

    def _generate_recommendations(
        self,
        result: WalkForwardResult,
        degradation_score: float,
        consistency_score: float,
        oos_performance_score: float,
        stability_score: float,
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if degradation_score < 50:
            recommendations.append(
                "High IS-to-OOS degradation suggests overfitting. "
                "Consider: 1) Simplifying strategy, 2) Reducing parameters, "
                "3) Adding regularization"
            )

        if consistency_score < 50:
            recommendations.append(
                "Low OOS win rate indicates inconsistent performance. "
                "Consider: 1) Testing on longer periods, 2) Adding filters, "
                "3) Market regime detection"
            )

        if oos_performance_score < 50:
            recommendations.append(
                "Poor absolute OOS performance. "
                "Consider: 1) Revisiting strategy logic, 2) Better entry/exit rules, "
                "3) Risk management improvements"
            )

        if stability_score < 50:
            recommendations.append(
                "Parameter instability across windows. "
                "Consider: 1) Widening parameter ranges, 2) Using robust optimization, "
                "3) Parameter regularization"
            )

        if result.avg_oos_sharpe > 0 and degradation_score > 60 and consistency_score > 60:
            recommendations.append(
                "Strategy shows good robustness. "
                "Next steps: 1) Paper trading, 2) Risk sizing, 3) Production deployment"
            )

        if not recommendations:
            recommendations.append("All validation checks passed. Strategy appears robust.")

        return recommendations

    def analyze_window_pattern(self, result: WalkForwardResult) -> Dict[str, Any]:
        """
        Analyze patterns across walk-forward windows.

        Args:
            result: Walk-forward result

        Returns:
            Dictionary with pattern analysis
        """
        if not result.windows:
            return {}

        # Extract time series of metrics
        oos_sharpes = [w.oos_sharpe for w in result.windows]
        oos_returns = [w.oos_total_return for w in result.windows]
        degradations = [w.sharpe_degradation for w in result.windows]

        # Trend analysis
        sharpe_trend = self._calculate_trend(oos_sharpes)
        return_trend = self._calculate_trend(oos_returns)

        # Volatility analysis
        sharpe_volatility = np.std(oos_sharpes) if len(oos_sharpes) > 1 else 0
        return_volatility = np.std(oos_returns) if len(oos_returns) > 1 else 0

        # Worst periods
        worst_windows = sorted(result.windows, key=lambda w: w.oos_sharpe)[:3]

        return {
            "trends": {
                "oos_sharpe_trend": sharpe_trend,
                "oos_return_trend": return_trend,
            },
            "volatility": {
                "oos_sharpe_volatility": round(sharpe_volatility, 3),
                "oos_return_volatility": round(return_volatility, 2),
            },
            "worst_periods": [
                {
                    "window_id": w.window_id,
                    "period": f"{w.out_sample_start.date()} to {w.out_sample_end.date()}",
                    "oos_sharpe": round(w.oos_sharpe, 2),
                    "oos_return": round(w.oos_total_return, 2),
                }
                for w in worst_windows
            ],
            "degradation_stats": {
                "mean": round(np.mean(degradations), 3),
                "std": round(np.std(degradations), 3),
                "max": round(max(degradations), 3),
                "min": round(min(degradations), 3),
            },
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction (improving, stable, declining)."""
        if len(values) < 3:
            return "insufficient_data"

        # Simple linear regression
        x = np.arange(len(values))
        y = np.array(values)

        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]

        # Determine trend
        if abs(slope) < 0.01:
            return "stable"
        elif slope > 0:
            return "improving"
        else:
            return "declining"

    def compare_to_benchmark(
        self,
        result: WalkForwardResult,
        benchmark_result: WalkForwardResult,
    ) -> Dict[str, Any]:
        """
        Compare walk-forward result to benchmark.

        Args:
            result: Strategy result
            benchmark_result: Benchmark result

        Returns:
            Comparison metrics
        """
        return {
            "oos_sharpe_diff": result.avg_oos_sharpe - benchmark_result.avg_oos_sharpe,
            "oos_return_diff": result.avg_oos_return - benchmark_result.avg_oos_return,
            "degradation_diff": result.avg_sharpe_degradation
            - benchmark_result.avg_sharpe_degradation,
            "consistency_diff": result.consistency_score - benchmark_result.consistency_score,
            "better_windows": sum(
                1
                for w1, w2 in zip(result.windows, benchmark_result.windows)
                if w1.oos_sharpe > w2.oos_sharpe
            ),
            "total_windows": len(result.windows),
        }
