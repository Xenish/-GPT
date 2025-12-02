"""
Performance Tracker.

Tracks strategy performance over time with snapshots and historical data.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from finantradealgo.research.performance.models import (
    PerformanceMetrics,
    PerformanceSnapshot,
    PerformancePeriod,
)


class PerformanceTracker:
    """
    Track strategy performance over time.

    Stores performance snapshots and provides historical analysis.
    """

    def __init__(
        self,
        strategy_id: str,
        storage_dir: Optional[Path] = None,
        is_live: bool = True,
    ):
        """
        Initialize performance tracker.

        Args:
            strategy_id: Unique identifier for strategy
            storage_dir: Directory to store performance data (default: outputs/performance)
            is_live: Whether this is tracking live trading (True) or backtest (False)
        """
        self.strategy_id = strategy_id
        self.is_live = is_live

        # Storage
        if storage_dir is None:
            storage_dir = Path("outputs") / "performance" / strategy_id
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot storage
        self.snapshots: List[PerformanceSnapshot] = []

        # Current metrics
        self.current_metrics: Optional[PerformanceMetrics] = None

        # Load existing snapshots
        self._load_snapshots()

    def record_snapshot(
        self,
        metrics: PerformanceMetrics,
        period: PerformancePeriod = PerformancePeriod.DAILY,
        market_condition: Optional[str] = None,
        regime: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> PerformanceSnapshot:
        """
        Record a performance snapshot.

        Args:
            metrics: Performance metrics at this point in time
            period: Time period this snapshot represents
            market_condition: Current market condition (optional)
            regime: Current market regime (optional)
            notes: Additional notes (optional)

        Returns:
            Created snapshot
        """
        snapshot = PerformanceSnapshot(
            strategy_id=self.strategy_id,
            snapshot_time=datetime.utcnow(),
            metrics=metrics,
            period=period,
            is_live=self.is_live,
            market_condition=market_condition,
            regime=regime,
            notes=notes,
        )

        self.snapshots.append(snapshot)
        self.current_metrics = metrics

        # Save to disk
        self._save_snapshot(snapshot)

        return snapshot

    def update_from_trades(self, trades_df: pd.DataFrame) -> PerformanceMetrics:
        """
        Calculate and record metrics from trade history.

        Args:
            trades_df: DataFrame with trade data (must have 'pnl', 'entry_time', 'exit_time' columns)

        Returns:
            Calculated performance metrics
        """
        if trades_df.empty:
            return PerformanceMetrics()

        # Calculate metrics
        metrics = self._calculate_metrics_from_trades(trades_df)

        # Record snapshot
        self.record_snapshot(metrics, period=PerformancePeriod.ALL_TIME)

        return metrics

    def _calculate_metrics_from_trades(self, trades_df: pd.DataFrame) -> PerformanceMetrics:
        """Calculate performance metrics from trade data."""
        # Basic trade stats
        total_trades = len(trades_df)
        winning_trades = (trades_df["pnl"] > 0).sum()
        losing_trades = (trades_df["pnl"] < 0).sum()

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # PnL stats
        total_pnl = trades_df["pnl"].sum()
        avg_win = trades_df[trades_df["pnl"] > 0]["pnl"].mean() if winning_trades > 0 else 0.0
        avg_loss = trades_df[trades_df["pnl"] < 0]["pnl"].mean() if losing_trades > 0 else 0.0

        # Profit factor
        gross_profit = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
        gross_loss = abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Calculate returns (assuming starting capital)
        # TODO: Make this configurable
        starting_capital = 10000.0
        total_return = (total_pnl / starting_capital) * 100

        # Calculate Sharpe ratio
        if len(trades_df) > 1:
            returns = trades_df["pnl"].values
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        # Calculate drawdown
        cumulative_pnl = trades_df["pnl"].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        max_drawdown = (drawdown.min() / starting_capital) * 100 if len(drawdown) > 0 else 0.0
        current_drawdown = (drawdown.iloc[-1] / starting_capital) * 100 if len(drawdown) > 0 else 0.0

        # Calculate consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0

        current_win_streak = 0
        current_loss_streak = 0

        for pnl in trades_df["pnl"]:
            if pnl > 0:
                current_win_streak += 1
                current_loss_streak = 0
                max_consecutive_wins = max(max_consecutive_wins, current_win_streak)
            elif pnl < 0:
                current_loss_streak += 1
                current_win_streak = 0
                max_consecutive_losses = max(max_consecutive_losses, current_loss_streak)

        consecutive_wins = current_win_streak
        consecutive_losses = current_loss_streak

        # Calculate trade duration
        if "entry_time" in trades_df.columns and "exit_time" in trades_df.columns:
            trades_df["duration"] = pd.to_datetime(trades_df["exit_time"]) - pd.to_datetime(trades_df["entry_time"])
            avg_duration_hours = trades_df["duration"].mean().total_seconds() / 3600 if len(trades_df) > 0 else 0.0
        else:
            avg_duration_hours = 0.0

        # Time period
        if "entry_time" in trades_df.columns:
            start_time = pd.to_datetime(trades_df["entry_time"].min())
            end_time = pd.to_datetime(trades_df["entry_time"].max())
            duration_days = (end_time - start_time).total_seconds() / 86400
        else:
            start_time = None
            end_time = None
            duration_days = 0.0

        # Calculate daily and annualized returns
        if duration_days > 0:
            daily_return = total_return / duration_days
            annualized_return = daily_return * 365
        else:
            daily_return = 0.0
            annualized_return = 0.0

        # Volatility
        volatility = trades_df["pnl"].std() if len(trades_df) > 1 else 0.0

        # Sortino ratio (downside deviation)
        downside_returns = trades_df[trades_df["pnl"] < 0]["pnl"]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0.0
        sortino_ratio = (trades_df["pnl"].mean() / downside_deviation) * np.sqrt(252) if downside_deviation > 0 else 0.0

        return PerformanceMetrics(
            total_pnl=total_pnl,
            total_return=total_return,
            daily_return=daily_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            volatility=volatility,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_trade_duration_hours=avg_duration_hours,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            start_time=start_time,
            end_time=end_time,
            duration_days=duration_days,
        )

    def get_snapshots(
        self,
        period: Optional[PerformancePeriod] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[PerformanceSnapshot]:
        """
        Get performance snapshots with optional filtering.

        Args:
            period: Filter by period type
            start_time: Filter snapshots after this time
            end_time: Filter snapshots before this time

        Returns:
            List of filtered snapshots
        """
        filtered = self.snapshots

        if period:
            filtered = [s for s in filtered if s.period == period]

        if start_time:
            filtered = [s for s in filtered if s.snapshot_time >= start_time]

        if end_time:
            filtered = [s for s in filtered if s.snapshot_time <= end_time]

        return filtered

    def get_metrics_history(
        self,
        metric_name: str,
        period: Optional[PerformancePeriod] = None,
    ) -> pd.DataFrame:
        """
        Get time series of a specific metric.

        Args:
            metric_name: Name of metric (e.g., "sharpe_ratio", "total_pnl")
            period: Filter by period type

        Returns:
            DataFrame with timestamp and metric values
        """
        snapshots = self.get_snapshots(period=period)

        if not snapshots:
            return pd.DataFrame(columns=["timestamp", metric_name])

        data = []
        for snapshot in snapshots:
            metrics_dict = snapshot.metrics.to_dict()
            if metric_name in metrics_dict:
                data.append({
                    "timestamp": snapshot.snapshot_time,
                    metric_name: metrics_dict[metric_name],
                })

        df = pd.DataFrame(data)
        df = df.sort_values("timestamp")
        return df

    def get_performance_summary(self) -> Dict:
        """
        Get summary of current performance.

        Returns:
            Dictionary with performance summary
        """
        if not self.current_metrics:
            return {"status": "no_data"}

        summary = {
            "strategy_id": self.strategy_id,
            "is_live": self.is_live,
            "last_update": datetime.utcnow().isoformat(),
            "total_snapshots": len(self.snapshots),
            "current_metrics": self.current_metrics.to_dict(),
        }

        # Add historical context if we have snapshots
        if len(self.snapshots) > 1:
            # Get sharpe ratio trend
            sharpe_history = self.get_metrics_history("sharpe_ratio")
            if not sharpe_history.empty:
                summary["sharpe_trend"] = {
                    "current": sharpe_history["sharpe_ratio"].iloc[-1],
                    "7d_avg": sharpe_history["sharpe_ratio"].tail(7).mean(),
                    "30d_avg": sharpe_history["sharpe_ratio"].tail(30).mean(),
                    "all_time_avg": sharpe_history["sharpe_ratio"].mean(),
                }

        return summary

    def _save_snapshot(self, snapshot: PerformanceSnapshot) -> None:
        """Save snapshot to disk."""
        # Save individual snapshot
        snapshot_file = self.storage_dir / f"snapshot_{snapshot.snapshot_time.strftime('%Y%m%d_%H%M%S')}.json"

        with snapshot_file.open("w", encoding="utf-8") as f:
            json.dump(snapshot.to_dict(), f, indent=2)

        # Also update the snapshots index
        self._save_snapshots_index()

    def _save_snapshots_index(self) -> None:
        """Save index of all snapshots."""
        index_file = self.storage_dir / "snapshots_index.json"

        index_data = {
            "strategy_id": self.strategy_id,
            "is_live": self.is_live,
            "total_snapshots": len(self.snapshots),
            "last_updated": datetime.utcnow().isoformat(),
            "snapshots": [s.to_dict() for s in self.snapshots],
        }

        with index_file.open("w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)

    def _load_snapshots(self) -> None:
        """Load existing snapshots from disk."""
        index_file = self.storage_dir / "snapshots_index.json"

        if not index_file.exists():
            return

        try:
            with index_file.open("r", encoding="utf-8") as f:
                index_data = json.load(f)

            self.snapshots = [
                PerformanceSnapshot.from_dict(s)
                for s in index_data.get("snapshots", [])
            ]

            # Set current metrics to latest snapshot
            if self.snapshots:
                self.current_metrics = self.snapshots[-1].metrics

        except Exception as e:
            print(f"[WARN] Failed to load snapshots: {e}")
            self.snapshots = []
