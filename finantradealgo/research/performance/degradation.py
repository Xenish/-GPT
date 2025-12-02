"""
Performance Degradation Detection.

Monitors strategy performance for degradation and triggers alerts.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from finantradealgo.research.performance.models import (
    PerformanceMetrics,
    PerformanceAlert,
    PerformanceStatus,
)
from finantradealgo.research.performance.tracker import PerformanceTracker
from finantradealgo.research.performance.comparison import PerformanceComparator
from finantradealgo.research.performance.aggregator import MetricsAggregator


class DegradationRule:
    """
    Rule for detecting performance degradation.
    """

    def __init__(
        self,
        name: str,
        metric: str,
        threshold: float,
        comparison_type: str = "absolute",  # "absolute", "relative", "trend"
        severity: str = "warning",
        message_template: str = "",
    ):
        """
        Initialize degradation rule.

        Args:
            name: Rule name
            metric: Metric to monitor
            threshold: Threshold value
            comparison_type: How to compare ("absolute", "relative", "trend")
            severity: Alert severity ("warning", "critical")
            message_template: Alert message template
        """
        self.name = name
        self.metric = metric
        self.threshold = threshold
        self.comparison_type = comparison_type
        self.severity = severity
        self.message_template = message_template

    def check(
        self,
        current_metrics: PerformanceMetrics,
        baseline_metrics: Optional[PerformanceMetrics] = None,
        historical_metrics: Optional[pd.DataFrame] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if rule is violated.

        Args:
            current_metrics: Current performance metrics
            baseline_metrics: Baseline (expected) metrics
            historical_metrics: Historical metrics for trend analysis

        Returns:
            (is_violated, alert_message)
        """
        current_value = getattr(current_metrics, self.metric, None)

        if current_value is None:
            return False, None

        violated = False
        message = None

        if self.comparison_type == "absolute":
            # Absolute threshold (e.g., Sharpe < 0.3)
            if current_value < self.threshold:
                violated = True
                message = self.message_template.format(
                    metric=self.metric,
                    current=current_value,
                    threshold=self.threshold,
                )

        elif self.comparison_type == "relative" and baseline_metrics:
            # Relative to baseline (e.g., Sharpe 30% lower than backtest)
            baseline_value = getattr(baseline_metrics, self.metric, None)

            if baseline_value is not None and baseline_value != 0:
                deviation = (current_value - baseline_value) / abs(baseline_value)

                if deviation < -self.threshold:  # Negative deviation = worse
                    violated = True
                    message = self.message_template.format(
                        metric=self.metric,
                        current=current_value,
                        baseline=baseline_value,
                        deviation=deviation * 100,
                    )

        elif self.comparison_type == "trend" and historical_metrics is not None:
            # Trend analysis
            if self.metric in historical_metrics.columns:
                recent = historical_metrics[self.metric].tail(5)

                if len(recent) >= 3:
                    # Check if declining trend
                    slope = np.polyfit(range(len(recent)), recent.values, 1)[0]

                    if slope < -self.threshold:
                        violated = True
                        message = self.message_template.format(
                            metric=self.metric,
                            slope=slope,
                            threshold=self.threshold,
                        )

        return violated, message


class PerformanceDegradationDetector:
    """
    Detect performance degradation using predefined rules.

    Monitors strategy performance and triggers alerts when rules are violated.
    """

    DEFAULT_RULES = [
        # Absolute thresholds
        DegradationRule(
            name="low_sharpe",
            metric="sharpe_ratio",
            threshold=0.3,
            comparison_type="absolute",
            severity="warning",
            message_template="Sharpe ratio is critically low: {current:.4f} (threshold: {threshold:.4f})",
        ),
        DegradationRule(
            name="high_drawdown",
            metric="max_drawdown",
            threshold=-30.0,  # -30%
            comparison_type="absolute",
            severity="critical",
            message_template="Max drawdown exceeds limit: {current:.2f}% (limit: {threshold:.2f}%)",
        ),
        DegradationRule(
            name="low_win_rate",
            metric="win_rate",
            threshold=0.35,
            comparison_type="absolute",
            severity="warning",
            message_template="Win rate is too low: {current:.2%} (threshold: {threshold:.2%})",
        ),
        # Relative thresholds (vs backtest)
        DegradationRule(
            name="sharpe_degradation",
            metric="sharpe_ratio",
            threshold=0.4,  # 40% degradation
            comparison_type="relative",
            severity="warning",
            message_template="Sharpe ratio degraded significantly: {current:.4f} vs expected {baseline:.4f} ({deviation:.1f}% worse)",
        ),
        DegradationRule(
            name="return_degradation",
            metric="total_return",
            threshold=0.5,  # 50% degradation
            comparison_type="relative",
            severity="warning",
            message_template="Returns degraded significantly: {current:.2f}% vs expected {baseline:.2f}% ({deviation:.1f}% worse)",
        ),
        # Trend-based rules
        DegradationRule(
            name="declining_sharpe_trend",
            metric="sharpe_ratio",
            threshold=0.05,  # Declining at 0.05 per period
            comparison_type="trend",
            severity="warning",
            message_template="Sharpe ratio is declining (slope: {slope:.4f})",
        ),
    ]

    def __init__(
        self,
        tracker: PerformanceTracker,
        baseline_metrics: Optional[PerformanceMetrics] = None,
        rules: Optional[List[DegradationRule]] = None,
    ):
        """
        Initialize degradation detector.

        Args:
            tracker: Performance tracker
            baseline_metrics: Baseline (backtest) metrics
            rules: List of degradation rules (default: DEFAULT_RULES)
        """
        self.tracker = tracker
        self.baseline_metrics = baseline_metrics
        self.rules = rules if rules is not None else self.DEFAULT_RULES.copy()
        self.aggregator = MetricsAggregator(tracker)

        # Alert history
        self.alerts: List[PerformanceAlert] = []

    def check_degradation(
        self,
        current_metrics: Optional[PerformanceMetrics] = None,
    ) -> List[PerformanceAlert]:
        """
        Check for performance degradation.

        Args:
            current_metrics: Current metrics (default: use tracker's current)

        Returns:
            List of triggered alerts
        """
        if current_metrics is None:
            current_metrics = self.tracker.current_metrics

        if current_metrics is None:
            return []

        alerts = []

        # Get historical metrics for trend analysis
        historical_metrics = {}
        for metric in ["sharpe_ratio", "total_return", "max_drawdown", "win_rate"]:
            hist = self.tracker.get_metrics_history(metric)
            if not hist.empty:
                historical_metrics[metric] = hist[metric]

        historical_df = pd.DataFrame(historical_metrics) if historical_metrics else None

        # Check each rule
        for rule in self.rules:
            violated, message = rule.check(
                current_metrics=current_metrics,
                baseline_metrics=self.baseline_metrics,
                historical_metrics=historical_df,
            )

            if violated:
                alert = PerformanceAlert(
                    strategy_id=self.tracker.strategy_id,
                    alert_time=datetime.utcnow(),
                    alert_type=rule.name,
                    severity=rule.severity,
                    message=message,
                    metrics_snapshot=current_metrics,
                    recommended_action=self._get_recommended_action(rule.name),
                )

                alerts.append(alert)
                self.alerts.append(alert)

        return alerts

    def _get_recommended_action(self, alert_type: str) -> str:
        """Get recommended action for alert type."""
        recommendations = {
            "low_sharpe": "Consider reducing position size or stopping trading until performance improves",
            "high_drawdown": "STOP TRADING IMMEDIATELY. Review strategy logic and market conditions",
            "low_win_rate": "Review recent losing trades for patterns. Consider regime change",
            "sharpe_degradation": "Monitor closely for next 10-20 trades. Compare to backtest assumptions",
            "return_degradation": "Check if market conditions differ from backtest period",
            "declining_sharpe_trend": "Performance is trending down. Consider reducing position size",
        }

        return recommendations.get(alert_type, "Monitor closely and investigate root cause")

    def get_degradation_report(self) -> Dict:
        """
        Generate degradation detection report.

        Returns:
            Report dictionary
        """
        recent_alerts = [a for a in self.alerts if a.alert_time >= datetime.utcnow() - timedelta(days=7)]

        report = {
            "strategy_id": self.tracker.strategy_id,
            "report_time": datetime.utcnow().isoformat(),
            "total_alerts": len(self.alerts),
            "recent_alerts_7d": len(recent_alerts),
            "critical_alerts": sum(1 for a in recent_alerts if a.severity == "critical"),
            "warning_alerts": sum(1 for a in recent_alerts if a.severity == "warning"),
            "active_rules": len(self.rules),
        }

        # Add recent alerts
        report["recent_alerts"] = [
            {
                "alert_time": a.alert_time.isoformat(),
                "alert_type": a.alert_type,
                "severity": a.severity,
                "message": a.message,
                "recommended_action": a.recommended_action,
            }
            for a in sorted(recent_alerts, key=lambda x: x.alert_time, reverse=True)[:10]
        ]

        # Add rule violation counts
        alert_counts = {}
        for alert in self.alerts:
            alert_counts[alert.alert_type] = alert_counts.get(alert.alert_type, 0) + 1

        report["alert_counts_by_type"] = alert_counts

        return report

    def add_custom_rule(self, rule: DegradationRule) -> None:
        """
        Add custom degradation rule.

        Args:
            rule: Degradation rule to add
        """
        self.rules.append(rule)

    def remove_rule(self, rule_name: str) -> bool:
        """
        Remove degradation rule by name.

        Args:
            rule_name: Name of rule to remove

        Returns:
            True if removed, False if not found
        """
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                del self.rules[i]
                return True

        return False

    def clear_alerts(self, days: Optional[int] = None) -> None:
        """
        Clear alert history.

        Args:
            days: Clear alerts older than N days (None = clear all)
        """
        if days is None:
            self.alerts = []
        else:
            cutoff = datetime.utcnow() - timedelta(days=days)
            self.alerts = [a for a in self.alerts if a.alert_time >= cutoff]


class ConsecutiveLossDetector:
    """
    Specialized detector for consecutive loss streaks.

    Triggers alerts when losing streaks exceed thresholds.
    """

    def __init__(
        self,
        strategy_id: str,
        warning_threshold: int = 5,
        critical_threshold: int = 10,
    ):
        """
        Initialize consecutive loss detector.

        Args:
            strategy_id: Strategy identifier
            warning_threshold: Number of consecutive losses for warning
            critical_threshold: Number of consecutive losses for critical alert
        """
        self.strategy_id = strategy_id
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

        self.current_streak = 0
        self.max_streak = 0

    def update(self, trade_pnl: float) -> Optional[PerformanceAlert]:
        """
        Update with new trade result.

        Args:
            trade_pnl: Trade PnL (positive = win, negative = loss)

        Returns:
            Alert if threshold exceeded, None otherwise
        """
        if trade_pnl < 0:
            # Loss
            self.current_streak += 1
            self.max_streak = max(self.max_streak, self.current_streak)

            # Check thresholds
            if self.current_streak >= self.critical_threshold:
                return PerformanceAlert(
                    strategy_id=self.strategy_id,
                    alert_time=datetime.utcnow(),
                    alert_type="critical_loss_streak",
                    severity="critical",
                    message=f"CRITICAL: {self.current_streak} consecutive losses",
                    metrics_snapshot=PerformanceMetrics(consecutive_losses=self.current_streak),
                    recommended_action="STOP TRADING IMMEDIATELY and investigate",
                )
            elif self.current_streak >= self.warning_threshold:
                return PerformanceAlert(
                    strategy_id=self.strategy_id,
                    alert_time=datetime.utcnow(),
                    alert_type="warning_loss_streak",
                    severity="warning",
                    message=f"WARNING: {self.current_streak} consecutive losses",
                    metrics_snapshot=PerformanceMetrics(consecutive_losses=self.current_streak),
                    recommended_action="Monitor closely. Consider reducing position size",
                )
        else:
            # Win - reset streak
            self.current_streak = 0

        return None

    def get_status(self) -> Dict:
        """Get current streak status."""
        return {
            "strategy_id": self.strategy_id,
            "current_streak": self.current_streak,
            "max_streak": self.max_streak,
            "warning_threshold": self.warning_threshold,
            "critical_threshold": self.critical_threshold,
        }
