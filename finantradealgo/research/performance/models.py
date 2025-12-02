"""
Performance Tracking Data Models.

Defines data structures for tracking and comparing strategy performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class PerformancePeriod(str, Enum):
    """Time periods for performance aggregation."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ALL_TIME = "all_time"


class PerformanceStatus(str, Enum):
    """Performance status classification."""

    EXCELLENT = "excellent"  # Performing better than expected
    GOOD = "good"  # Within expected range (upper half)
    ACCEPTABLE = "acceptable"  # Within expected range (lower half)
    WARNING = "warning"  # Below expectations but not critical
    CRITICAL = "critical"  # Significantly underperforming


@dataclass
class PerformanceMetrics:
    """
    Core performance metrics for a strategy.

    These metrics can represent either backtest or live performance.
    """

    # Returns
    total_pnl: float = 0.0
    total_return: float = 0.0  # As percentage
    daily_return: float = 0.0
    annualized_return: float = 0.0

    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    volatility: float = 0.0

    # Trade metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    avg_trade_duration_hours: float = 0.0

    # Consistency metrics
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Time tracking
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_days: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_pnl": self.total_pnl,
            "total_return": self.total_return,
            "daily_return": self.daily_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "current_drawdown": self.current_drawdown,
            "volatility": self.volatility,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "avg_trade_duration_hours": self.avg_trade_duration_hours,
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_days": self.duration_days,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> PerformanceMetrics:
        """Create from dictionary."""
        # Handle datetime conversion
        if data.get("start_time"):
            data["start_time"] = datetime.fromisoformat(data["start_time"])
        if data.get("end_time"):
            data["end_time"] = datetime.fromisoformat(data["end_time"])

        return cls(**data)


@dataclass
class PerformanceSnapshot:
    """
    Point-in-time snapshot of strategy performance.

    Used for tracking performance over time.
    """

    strategy_id: str
    snapshot_time: datetime
    metrics: PerformanceMetrics
    period: PerformancePeriod
    is_live: bool  # True for live trading, False for backtest

    # Optional context
    market_condition: Optional[str] = None  # e.g., "bull", "bear", "sideways"
    regime: Optional[str] = None  # Market regime at snapshot time
    notes: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "strategy_id": self.strategy_id,
            "snapshot_time": self.snapshot_time.isoformat(),
            "metrics": self.metrics.to_dict(),
            "period": self.period.value,
            "is_live": self.is_live,
            "market_condition": self.market_condition,
            "regime": self.regime,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> PerformanceSnapshot:
        """Create from dictionary."""
        data["snapshot_time"] = datetime.fromisoformat(data["snapshot_time"])
        data["metrics"] = PerformanceMetrics.from_dict(data["metrics"])
        data["period"] = PerformancePeriod(data["period"])
        return cls(**data)


@dataclass
class PerformanceComparison:
    """
    Comparison between live and backtest performance.

    Highlights deviations and provides status assessment.
    """

    strategy_id: str
    comparison_time: datetime

    # Performance data
    live_metrics: PerformanceMetrics
    backtest_metrics: PerformanceMetrics

    # Deviations (live - backtest)
    sharpe_delta: float = 0.0
    return_delta: float = 0.0
    drawdown_delta: float = 0.0
    win_rate_delta: float = 0.0

    # Relative deviations (percentage)
    sharpe_deviation_pct: float = 0.0
    return_deviation_pct: float = 0.0
    drawdown_deviation_pct: float = 0.0
    win_rate_deviation_pct: float = 0.0

    # Status assessment
    overall_status: PerformanceStatus = PerformanceStatus.ACCEPTABLE

    # Warnings and issues
    warnings: List[str] = field(default_factory=list)
    critical_issues: List[str] = field(default_factory=list)

    # Time context
    live_period_days: float = 0.0
    backtest_period_days: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "strategy_id": self.strategy_id,
            "comparison_time": self.comparison_time.isoformat(),
            "live_metrics": self.live_metrics.to_dict(),
            "backtest_metrics": self.backtest_metrics.to_dict(),
            "sharpe_delta": self.sharpe_delta,
            "return_delta": self.return_delta,
            "drawdown_delta": self.drawdown_delta,
            "win_rate_delta": self.win_rate_delta,
            "sharpe_deviation_pct": self.sharpe_deviation_pct,
            "return_deviation_pct": self.return_deviation_pct,
            "drawdown_deviation_pct": self.drawdown_deviation_pct,
            "win_rate_deviation_pct": self.win_rate_deviation_pct,
            "overall_status": self.overall_status.value,
            "warnings": self.warnings,
            "critical_issues": self.critical_issues,
            "live_period_days": self.live_period_days,
            "backtest_period_days": self.backtest_period_days,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> PerformanceComparison:
        """Create from dictionary."""
        data["comparison_time"] = datetime.fromisoformat(data["comparison_time"])
        data["live_metrics"] = PerformanceMetrics.from_dict(data["live_metrics"])
        data["backtest_metrics"] = PerformanceMetrics.from_dict(data["backtest_metrics"])
        data["overall_status"] = PerformanceStatus(data["overall_status"])
        return cls(**data)


@dataclass
class PerformanceAlert:
    """
    Alert triggered by performance degradation or issues.
    """

    strategy_id: str
    alert_time: datetime
    alert_type: str  # e.g., "underperformance", "high_drawdown", "low_win_rate"
    severity: str  # "warning", "critical"
    message: str
    metrics_snapshot: PerformanceMetrics
    recommended_action: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "strategy_id": self.strategy_id,
            "alert_time": self.alert_time.isoformat(),
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "metrics_snapshot": self.metrics_snapshot.to_dict(),
            "recommended_action": self.recommended_action,
        }


@dataclass
class PerformanceAttribution:
    """
    Performance attribution analysis.

    Breaks down performance by contributing factors.
    """

    strategy_id: str
    period_start: datetime
    period_end: datetime

    # Attribution by factor
    total_pnl: float = 0.0
    pnl_by_regime: Dict[str, float] = field(default_factory=dict)  # e.g., {"bull": 100, "bear": -20}
    pnl_by_timeframe: Dict[str, float] = field(default_factory=dict)  # e.g., {"15m": 80, "1h": 40}
    pnl_by_symbol: Dict[str, float] = field(default_factory=dict)  # e.g., {"BTCUSDT": 60, "ETHUSDT": 60}

    # Trade attribution
    top_trades: List[Dict] = field(default_factory=list)  # Top contributing trades
    worst_trades: List[Dict] = field(default_factory=list)  # Worst trades

    # Component attribution (for ensembles)
    pnl_by_component: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "strategy_id": self.strategy_id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_pnl": self.total_pnl,
            "pnl_by_regime": self.pnl_by_regime,
            "pnl_by_timeframe": self.pnl_by_timeframe,
            "pnl_by_symbol": self.pnl_by_symbol,
            "top_trades": self.top_trades,
            "worst_trades": self.worst_trades,
            "pnl_by_component": self.pnl_by_component,
        }
