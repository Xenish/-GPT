"""
Performance Comparison.

Compares live performance to backtest expectations and detects degradation.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from finantradealgo.research.performance.models import (
    PerformanceMetrics,
    PerformanceComparison,
    PerformanceStatus,
)


class PerformanceComparator:
    """
    Compare live performance to backtest expectations.

    Detects performance degradation and provides status assessment.
    """

    def __init__(
        self,
        acceptable_sharpe_degradation: float = 0.3,  # 30% degradation is acceptable
        acceptable_return_degradation: float = 0.4,  # 40% degradation is acceptable
        acceptable_drawdown_increase: float = 0.5,  # 50% increase in DD is acceptable
        warning_threshold: float = 0.5,  # 50% degradation triggers warning
        critical_threshold: float = 0.7,  # 70% degradation triggers critical
    ):
        """
        Initialize performance comparator.

        Args:
            acceptable_sharpe_degradation: Acceptable Sharpe ratio degradation (0-1)
            acceptable_return_degradation: Acceptable return degradation (0-1)
            acceptable_drawdown_increase: Acceptable max DD increase (0-1)
            warning_threshold: Threshold for warning status (0-1)
            critical_threshold: Threshold for critical status (0-1)
        """
        self.acceptable_sharpe_degradation = acceptable_sharpe_degradation
        self.acceptable_return_degradation = acceptable_return_degradation
        self.acceptable_drawdown_increase = acceptable_drawdown_increase
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    def compare(
        self,
        strategy_id: str,
        live_metrics: PerformanceMetrics,
        backtest_metrics: PerformanceMetrics,
    ) -> PerformanceComparison:
        """
        Compare live and backtest performance.

        Args:
            strategy_id: Strategy identifier
            live_metrics: Live trading performance metrics
            backtest_metrics: Backtest performance metrics

        Returns:
            Performance comparison with status and warnings
        """
        # Calculate deltas
        sharpe_delta = live_metrics.sharpe_ratio - backtest_metrics.sharpe_ratio
        return_delta = live_metrics.total_return - backtest_metrics.total_return
        drawdown_delta = live_metrics.max_drawdown - backtest_metrics.max_drawdown
        win_rate_delta = live_metrics.win_rate - backtest_metrics.win_rate

        # Calculate relative deviations (percentage)
        sharpe_deviation_pct = self._calculate_deviation_pct(
            live_metrics.sharpe_ratio,
            backtest_metrics.sharpe_ratio,
        )

        return_deviation_pct = self._calculate_deviation_pct(
            live_metrics.total_return,
            backtest_metrics.total_return,
        )

        drawdown_deviation_pct = self._calculate_deviation_pct(
            live_metrics.max_drawdown,
            backtest_metrics.max_drawdown,
            inverse=True,  # Drawdown is negative, so worse = higher
        )

        win_rate_deviation_pct = self._calculate_deviation_pct(
            live_metrics.win_rate,
            backtest_metrics.win_rate,
        )

        # Assess overall status
        status, warnings, critical_issues = self._assess_status(
            live_metrics=live_metrics,
            backtest_metrics=backtest_metrics,
            sharpe_deviation_pct=sharpe_deviation_pct,
            return_deviation_pct=return_deviation_pct,
            drawdown_deviation_pct=drawdown_deviation_pct,
            win_rate_deviation_pct=win_rate_deviation_pct,
        )

        # Create comparison
        comparison = PerformanceComparison(
            strategy_id=strategy_id,
            comparison_time=datetime.utcnow(),
            live_metrics=live_metrics,
            backtest_metrics=backtest_metrics,
            sharpe_delta=sharpe_delta,
            return_delta=return_delta,
            drawdown_delta=drawdown_delta,
            win_rate_delta=win_rate_delta,
            sharpe_deviation_pct=sharpe_deviation_pct,
            return_deviation_pct=return_deviation_pct,
            drawdown_deviation_pct=drawdown_deviation_pct,
            win_rate_deviation_pct=win_rate_deviation_pct,
            overall_status=status,
            warnings=warnings,
            critical_issues=critical_issues,
            live_period_days=live_metrics.duration_days,
            backtest_period_days=backtest_metrics.duration_days,
        )

        return comparison

    def _calculate_deviation_pct(
        self,
        live_value: float,
        backtest_value: float,
        inverse: bool = False,
    ) -> float:
        """
        Calculate percentage deviation from expected value.

        Args:
            live_value: Live metric value
            backtest_value: Expected (backtest) metric value
            inverse: If True, higher live value is worse (e.g., for drawdown)

        Returns:
            Deviation percentage (negative = underperformance)
        """
        if backtest_value == 0:
            return 0.0

        deviation = (live_value - backtest_value) / abs(backtest_value)

        if inverse:
            deviation = -deviation

        return deviation

    def _assess_status(
        self,
        live_metrics: PerformanceMetrics,
        backtest_metrics: PerformanceMetrics,
        sharpe_deviation_pct: float,
        return_deviation_pct: float,
        drawdown_deviation_pct: float,
        win_rate_deviation_pct: float,
    ) -> tuple[PerformanceStatus, List[str], List[str]]:
        """
        Assess overall performance status.

        Returns:
            (status, warnings, critical_issues)
        """
        warnings = []
        critical_issues = []

        # Check Sharpe ratio
        if sharpe_deviation_pct < -self.critical_threshold:
            critical_issues.append(
                f"Sharpe ratio critically low: {live_metrics.sharpe_ratio:.4f} "
                f"vs expected {backtest_metrics.sharpe_ratio:.4f} "
                f"({sharpe_deviation_pct * 100:.1f}% deviation)"
            )
        elif sharpe_deviation_pct < -self.warning_threshold:
            warnings.append(
                f"Sharpe ratio below expectations: {live_metrics.sharpe_ratio:.4f} "
                f"vs expected {backtest_metrics.sharpe_ratio:.4f} "
                f"({sharpe_deviation_pct * 100:.1f}% deviation)"
            )

        # Check returns
        if return_deviation_pct < -self.critical_threshold:
            critical_issues.append(
                f"Returns critically low: {live_metrics.total_return:.2f}% "
                f"vs expected {backtest_metrics.total_return:.2f}% "
                f"({return_deviation_pct * 100:.1f}% deviation)"
            )
        elif return_deviation_pct < -self.warning_threshold:
            warnings.append(
                f"Returns below expectations: {live_metrics.total_return:.2f}% "
                f"vs expected {backtest_metrics.total_return:.2f}% "
                f"({return_deviation_pct * 100:.1f}% deviation)"
            )

        # Check drawdown
        if drawdown_deviation_pct < -self.critical_threshold:
            critical_issues.append(
                f"Max drawdown critically high: {live_metrics.max_drawdown:.2f}% "
                f"vs expected {backtest_metrics.max_drawdown:.2f}% "
                f"({abs(drawdown_deviation_pct) * 100:.1f}% worse)"
            )
        elif drawdown_deviation_pct < -self.warning_threshold:
            warnings.append(
                f"Max drawdown higher than expected: {live_metrics.max_drawdown:.2f}% "
                f"vs expected {backtest_metrics.max_drawdown:.2f}% "
                f"({abs(drawdown_deviation_pct) * 100:.1f}% worse)"
            )

        # Check win rate
        if win_rate_deviation_pct < -self.warning_threshold:
            warnings.append(
                f"Win rate below expectations: {live_metrics.win_rate:.2%} "
                f"vs expected {backtest_metrics.win_rate:.2%} "
                f"({win_rate_deviation_pct * 100:.1f}% deviation)"
            )

        # Check consecutive losses
        if live_metrics.consecutive_losses > 5:
            warnings.append(
                f"High consecutive losses: {live_metrics.consecutive_losses} in a row"
            )

        if live_metrics.consecutive_losses > 10:
            critical_issues.append(
                f"Critically high consecutive losses: {live_metrics.consecutive_losses} in a row"
            )

        # Check current drawdown
        if abs(live_metrics.current_drawdown) > abs(backtest_metrics.max_drawdown) * 1.5:
            critical_issues.append(
                f"Current drawdown exceeds backtest max DD by 50%: "
                f"{live_metrics.current_drawdown:.2f}% vs {backtest_metrics.max_drawdown:.2f}%"
            )

        # Determine overall status
        if critical_issues:
            status = PerformanceStatus.CRITICAL
        elif warnings:
            # Check severity of warnings
            if sharpe_deviation_pct < -self.acceptable_sharpe_degradation:
                status = PerformanceStatus.WARNING
            elif return_deviation_pct < -self.acceptable_return_degradation:
                status = PerformanceStatus.WARNING
            else:
                status = PerformanceStatus.ACCEPTABLE
        else:
            # No issues - check if performing well
            if sharpe_deviation_pct > 0.1 and return_deviation_pct > 0.1:
                status = PerformanceStatus.EXCELLENT
            elif sharpe_deviation_pct >= 0 or return_deviation_pct >= 0:
                status = PerformanceStatus.GOOD
            else:
                status = PerformanceStatus.ACCEPTABLE

        return status, warnings, critical_issues

    def generate_comparison_report(self, comparison: PerformanceComparison) -> str:
        """
        Generate human-readable comparison report.

        Args:
            comparison: Performance comparison

        Returns:
            Formatted report string
        """
        lines = []

        lines.append("=" * 70)
        lines.append(f"PERFORMANCE COMPARISON REPORT: {comparison.strategy_id}")
        lines.append("=" * 70)
        lines.append(f"Comparison Time: {comparison.comparison_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(f"Overall Status: {comparison.overall_status.value.upper()}")
        lines.append("")

        # Metrics comparison table
        lines.append("METRICS COMPARISON:")
        lines.append("-" * 70)
        lines.append(f"{'Metric':<20} {'Live':<15} {'Backtest':<15} {'Delta':<15}")
        lines.append("-" * 70)

        metrics = [
            ("Sharpe Ratio", "sharpe_ratio", "{:.4f}"),
            ("Total Return", "total_return", "{:.2f}%"),
            ("Max Drawdown", "max_drawdown", "{:.2f}%"),
            ("Win Rate", "win_rate", "{:.2%}"),
            ("Trade Count", "total_trades", "{:d}"),
            ("Profit Factor", "profit_factor", "{:.2f}"),
        ]

        for label, attr, fmt in metrics:
            live_val = getattr(comparison.live_metrics, attr)
            backtest_val = getattr(comparison.backtest_metrics, attr)

            if attr in ["total_trades"]:
                delta_str = f"{live_val - backtest_val:+d}"
            elif "%" in fmt:
                delta_str = f"{live_val - backtest_val:+.2f}pp"
            else:
                delta_str = f"{live_val - backtest_val:+.4f}"

            lines.append(f"{label:<20} {fmt.format(live_val):<15} {fmt.format(backtest_val):<15} {delta_str:<15}")

        lines.append("")

        # Deviation analysis
        lines.append("DEVIATION ANALYSIS:")
        lines.append("-" * 70)
        lines.append(f"Sharpe Deviation:     {comparison.sharpe_deviation_pct:+.1%}")
        lines.append(f"Return Deviation:     {comparison.return_deviation_pct:+.1%}")
        lines.append(f"Drawdown Deviation:   {comparison.drawdown_deviation_pct:+.1%}")
        lines.append(f"Win Rate Deviation:   {comparison.win_rate_deviation_pct:+.1%}")
        lines.append("")

        # Warnings
        if comparison.warnings:
            lines.append("WARNINGS:")
            lines.append("-" * 70)
            for warning in comparison.warnings:
                lines.append(f"⚠  {warning}")
            lines.append("")

        # Critical issues
        if comparison.critical_issues:
            lines.append("CRITICAL ISSUES:")
            lines.append("-" * 70)
            for issue in comparison.critical_issues:
                lines.append(f"❌ {issue}")
            lines.append("")

        # Recommendations
        lines.append("RECOMMENDATIONS:")
        lines.append("-" * 70)

        if comparison.overall_status == PerformanceStatus.CRITICAL:
            lines.append("🛑 IMMEDIATE ACTION REQUIRED:")
            lines.append("   1. Stop trading immediately")
            lines.append("   2. Review recent trades for anomalies")
            lines.append("   3. Check if market conditions have changed significantly")
            lines.append("   4. Consider re-optimizing strategy parameters")
        elif comparison.overall_status == PerformanceStatus.WARNING:
            lines.append("⚠️  MONITORING REQUIRED:")
            lines.append("   1. Closely monitor next 10-20 trades")
            lines.append("   2. Consider reducing position size")
            lines.append("   3. Review if current market regime matches backtest period")
        elif comparison.overall_status == PerformanceStatus.ACCEPTABLE:
            lines.append("✅ Performance is acceptable")
            lines.append("   - Continue monitoring regularly")
            lines.append("   - Keep tracking metrics vs backtest")
        elif comparison.overall_status in [PerformanceStatus.GOOD, PerformanceStatus.EXCELLENT]:
            lines.append("🎯 Performance is meeting or exceeding expectations")
            lines.append("   - Continue current strategy")
            lines.append("   - Consider gradually increasing position size if risk allows")

        lines.append("=" * 70)

        return "\n".join(lines)
