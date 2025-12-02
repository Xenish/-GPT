"""
Performance Metrics Aggregator.

Aggregates performance metrics across different time periods and provides
rolling window calculations.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from finantradealgo.research.performance.models import (
    PerformanceMetrics,
    PerformancePeriod,
    PerformanceSnapshot,
)
from finantradealgo.research.performance.tracker import PerformanceTracker


class MetricsAggregator:
    """
    Aggregate performance metrics across time periods.

    Provides rolling window calculations and time-based aggregations.
    """

    def __init__(self, tracker: PerformanceTracker):
        """
        Initialize metrics aggregator.

        Args:
            tracker: Performance tracker instance
        """
        self.tracker = tracker

    def aggregate_by_period(
        self,
        period: PerformancePeriod,
        metric_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Aggregate metrics by time period.

        Args:
            period: Time period for aggregation
            metric_name: Name of metric to aggregate
            start_time: Start time for filtering
            end_time: End time for filtering

        Returns:
            DataFrame with aggregated metrics
        """
        # Get metric history
        history = self.tracker.get_metrics_history(
            metric_name=metric_name,
            period=period,
        )

        if history.empty:
            return pd.DataFrame()

        # Filter by time if specified
        if start_time:
            history = history[history["timestamp"] >= start_time]
        if end_time:
            history = history[history["timestamp"] <= end_time]

        # Determine resampling frequency based on period
        freq_map = {
            PerformancePeriod.DAILY: "D",
            PerformancePeriod.WEEKLY: "W",
            PerformancePeriod.MONTHLY: "M",
        }

        freq = freq_map.get(period, "D")

        # Set timestamp as index for resampling
        history = history.set_index("timestamp")

        # Aggregate
        aggregated = history.resample(freq).agg({
            metric_name: ["mean", "std", "min", "max", "count"]
        })

        aggregated.columns = ["mean", "std", "min", "max", "count"]
        aggregated = aggregated.reset_index()

        return aggregated

    def calculate_rolling_metrics(
        self,
        metric_name: str,
        window_days: int = 7,
        min_periods: int = 1,
    ) -> pd.DataFrame:
        """
        Calculate rolling window statistics for a metric.

        Args:
            metric_name: Name of metric
            window_days: Rolling window size in days
            min_periods: Minimum number of observations required

        Returns:
            DataFrame with rolling statistics
        """
        # Get full metric history
        history = self.tracker.get_metrics_history(metric_name)

        if history.empty:
            return pd.DataFrame()

        # Set timestamp as index
        history = history.set_index("timestamp")

        # Calculate rolling statistics
        window = f"{window_days}D"

        rolling = pd.DataFrame({
            "timestamp": history.index,
            f"{metric_name}": history[metric_name],
            f"{metric_name}_rolling_mean": history[metric_name].rolling(
                window=window,
                min_periods=min_periods,
            ).mean(),
            f"{metric_name}_rolling_std": history[metric_name].rolling(
                window=window,
                min_periods=min_periods,
            ).std(),
            f"{metric_name}_rolling_min": history[metric_name].rolling(
                window=window,
                min_periods=min_periods,
            ).min(),
            f"{metric_name}_rolling_max": history[metric_name].rolling(
                window=window,
                min_periods=min_periods,
            ).max(),
        })

        rolling = rolling.reset_index(drop=True)
        return rolling

    def calculate_period_over_period_change(
        self,
        metric_name: str,
        period: PerformancePeriod = PerformancePeriod.DAILY,
    ) -> pd.DataFrame:
        """
        Calculate period-over-period change for a metric.

        Args:
            metric_name: Name of metric
            period: Time period for comparison

        Returns:
            DataFrame with period-over-period changes
        """
        # Get aggregated data
        aggregated = self.aggregate_by_period(period, metric_name)

        if aggregated.empty or len(aggregated) < 2:
            return pd.DataFrame()

        # Calculate change
        aggregated["change"] = aggregated["mean"].diff()
        aggregated["change_pct"] = aggregated["mean"].pct_change() * 100

        return aggregated

    def calculate_cumulative_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """
        Calculate cumulative metrics over a time period.

        Args:
            start_time: Start time (None = beginning)
            end_time: End time (None = now)

        Returns:
            Dictionary of cumulative metrics
        """
        snapshots = self.tracker.get_snapshots(
            start_time=start_time,
            end_time=end_time,
        )

        if not snapshots:
            return {}

        # Aggregate across all snapshots
        total_pnl = sum(s.metrics.total_pnl for s in snapshots)
        total_trades = sum(s.metrics.total_trades for s in snapshots)
        winning_trades = sum(s.metrics.winning_trades for s in snapshots)

        # Calculate win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # Get latest metrics for current state
        latest = snapshots[-1].metrics

        return {
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": win_rate,
            "current_sharpe": latest.sharpe_ratio,
            "current_drawdown": latest.current_drawdown,
            "max_drawdown": latest.max_drawdown,
            "period_start": snapshots[0].snapshot_time.isoformat(),
            "period_end": snapshots[-1].snapshot_time.isoformat(),
        }

    def get_performance_trend(
        self,
        metric_name: str,
        window_days: int = 30,
    ) -> str:
        """
        Determine performance trend for a metric.

        Args:
            metric_name: Name of metric
            window_days: Window for trend analysis

        Returns:
            Trend description: "improving", "stable", "declining", "insufficient_data"
        """
        # Get rolling metrics
        rolling = self.calculate_rolling_metrics(
            metric_name=metric_name,
            window_days=window_days,
        )

        if rolling.empty or len(rolling) < 3:
            return "insufficient_data"

        # Get recent values (last 3 data points)
        recent = rolling.tail(3)

        mean_col = f"{metric_name}_rolling_mean"

        if mean_col not in recent.columns:
            return "insufficient_data"

        values = recent[mean_col].dropna()

        if len(values) < 2:
            return "insufficient_data"

        # Calculate trend
        # Simple linear regression slope
        x = np.arange(len(values))
        y = values.values

        if len(x) < 2:
            return "insufficient_data"

        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]

        # Determine trend based on slope
        # Normalize by value magnitude to get relative slope
        avg_value = abs(y.mean())

        if avg_value == 0:
            return "stable"

        relative_slope = slope / avg_value

        if relative_slope > 0.05:  # 5% improvement
            return "improving"
        elif relative_slope < -0.05:  # 5% decline
            return "declining"
        else:
            return "stable"

    def generate_performance_report(
        self,
        periods: List[PerformancePeriod] = None,
    ) -> Dict:
        """
        Generate comprehensive performance report.

        Args:
            periods: List of periods to include (default: all)

        Returns:
            Performance report dictionary
        """
        if periods is None:
            periods = [
                PerformancePeriod.DAILY,
                PerformancePeriod.WEEKLY,
                PerformancePeriod.MONTHLY,
            ]

        report = {
            "strategy_id": self.tracker.strategy_id,
            "is_live": self.tracker.is_live,
            "report_time": datetime.utcnow().isoformat(),
            "periods": {},
        }

        # Add metrics for each period
        key_metrics = ["sharpe_ratio", "total_return", "max_drawdown", "win_rate"]

        for period in periods:
            period_data = {}

            for metric in key_metrics:
                # Get aggregated data
                agg = self.aggregate_by_period(period, metric)

                if not agg.empty:
                    period_data[metric] = {
                        "current": float(agg["mean"].iloc[-1]) if len(agg) > 0 else 0.0,
                        "avg": float(agg["mean"].mean()),
                        "min": float(agg["min"].min()),
                        "max": float(agg["max"].max()),
                    }

                # Get trend
                trend = self.get_performance_trend(metric, window_days=30)
                period_data[f"{metric}_trend"] = trend

            report["periods"][period.value] = period_data

        # Add cumulative metrics
        report["cumulative"] = self.calculate_cumulative_metrics()

        # Add current snapshot
        if self.tracker.current_metrics:
            report["current"] = self.tracker.current_metrics.to_dict()

        return report


class PerformanceDatabase:
    """
    Persistent storage for performance metrics.

    Stores metrics in parquet files for efficient querying.
    """

    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize performance database.

        Args:
            storage_dir: Directory for database files (default: outputs/performance_db)
        """
        from pathlib import Path

        if storage_dir is None:
            storage_dir = Path("outputs") / "performance_db"

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_metrics(
        self,
        strategy_id: str,
        metrics: PerformanceMetrics,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Save performance metrics to database.

        Args:
            strategy_id: Strategy identifier
            metrics: Performance metrics
            timestamp: Timestamp (default: now)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        # Convert metrics to row
        row = metrics.to_dict()
        row["strategy_id"] = strategy_id
        row["timestamp"] = timestamp

        # Create DataFrame
        df = pd.DataFrame([row])

        # File path
        file_path = self.storage_dir / f"{strategy_id}_metrics.parquet"

        # Append or create
        if file_path.exists():
            existing = pd.read_parquet(file_path)
            df = pd.concat([existing, df], ignore_index=True)

        # Save
        df.to_parquet(file_path, index=False)

    def load_metrics(
        self,
        strategy_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Load performance metrics from database.

        Args:
            strategy_id: Strategy identifier
            start_time: Filter start time
            end_time: Filter end time

        Returns:
            DataFrame with metrics
        """
        file_path = self.storage_dir / f"{strategy_id}_metrics.parquet"

        if not file_path.exists():
            return pd.DataFrame()

        df = pd.read_parquet(file_path)

        # Filter by time
        if start_time:
            df = df[df["timestamp"] >= start_time]
        if end_time:
            df = df[df["timestamp"] <= end_time]

        return df

    def list_strategies(self) -> List[str]:
        """
        List all strategies in database.

        Returns:
            List of strategy IDs
        """
        strategy_ids = []

        for file_path in self.storage_dir.glob("*_metrics.parquet"):
            strategy_id = file_path.stem.replace("_metrics", "")
            strategy_ids.append(strategy_id)

        return strategy_ids
