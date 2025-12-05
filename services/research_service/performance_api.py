"""
Performance Tracking API Endpoints.

REST endpoints for performance monitoring, comparison, and alerts.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from finantradealgo.research.reporting import LiveReportGenerator
from finantradealgo.system.config_loader import load_config

router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================

class RecordMetricsRequest(BaseModel):
    """Request to record performance metrics."""
    strategy_id: str = Field(..., description="Strategy identifier")
    trades: List[Dict[str, Any]] = Field(..., description="List of trades with 'pnl', 'entry_time', 'exit_time'")
    period: str = Field("daily", description="Snapshot period: 'daily', 'weekly', 'monthly'")
    market_condition: Optional[str] = Field(None, description="Current market condition")
    regime: Optional[str] = Field(None, description="Current market regime")


class ComparePerformanceRequest(BaseModel):
    """Request to compare live vs backtest performance."""
    strategy_id: str = Field(..., description="Strategy identifier")
    live_trades: List[Dict[str, Any]] = Field(..., description="Live trades")
    backtest_trades: List[Dict[str, Any]] = Field(..., description="Backtest trades")


class SubscribeAlertsRequest(BaseModel):
    """Request to subscribe to performance alerts."""
    strategy_id: str = Field(..., description="Strategy identifier")
    channels: List[str] = Field(..., description="Alert channels: 'console', 'file', 'slack', 'webhook'")
    severity_filter: Optional[List[str]] = Field(None, description="Filter by severity: 'warning', 'critical'")
    alert_type_filter: Optional[List[str]] = Field(None, description="Filter by alert type")
    min_interval_minutes: int = Field(15, description="Minimum interval between same alerts")


class AttributionAnalysisRequest(BaseModel):
    """Request for attribution analysis."""
    strategy_id: str = Field(..., description="Strategy identifier")
    trades: List[Dict[str, Any]] = Field(..., description="Trades for analysis")
    group_by: Optional[List[str]] = Field(None, description="Columns to group by")


class PerformanceMetricsResponse(BaseModel):
    """Performance metrics response."""
    strategy_id: str
    total_pnl: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    timestamp: str


class PerformanceComparisonResponse(BaseModel):
    """Performance comparison response."""
    strategy_id: str
    comparison_time: str
    live_sharpe: float
    backtest_sharpe: float
    sharpe_delta: float
    sharpe_deviation_pct: float
    overall_status: str
    warnings: List[str]
    critical_issues: List[str]


class AlertSubscriptionResponse(BaseModel):
    """Alert subscription response."""
    success: bool
    strategy_id: str
    channels: List[str]
    message: str


class AttributionResponse(BaseModel):
    """Attribution analysis response."""
    strategy_id: str
    total_pnl: float
    pnl_by_regime: Optional[Dict[str, float]]
    pnl_by_symbol: Optional[Dict[str, float]]
    pnl_by_component: Optional[Dict[str, float]]
    top_trades: List[Dict[str, Any]]
    worst_trades: List[Dict[str, Any]]


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/track", response_model=PerformanceMetricsResponse)
async def track_performance(request: RecordMetricsRequest):
    """
    Record and track strategy performance.

    This endpoint:
    1. Calculates performance metrics from trades
    2. Records a performance snapshot
    3. Stores metrics in database
    4. Returns current performance

    Args:
        request: Performance tracking request

    Returns:
        Current performance metrics
    """
    try:
        import pandas as pd
        from finantradealgo.research.performance import (
            PerformanceTracker,
            PerformancePeriod,
            PerformanceDatabase,
        )

        # Convert trades to DataFrame
        trades_df = pd.DataFrame(request.trades)

        if trades_df.empty:
            raise HTTPException(status_code=400, detail="No trades provided")

        # Create tracker
        tracker = PerformanceTracker(
            strategy_id=request.strategy_id,
            is_live=True,
        )

        # Update from trades
        metrics = tracker.update_from_trades(trades_df)

        # Record snapshot
        period_map = {
            "daily": PerformancePeriod.DAILY,
            "weekly": PerformancePeriod.WEEKLY,
            "monthly": PerformancePeriod.MONTHLY,
        }

        period = period_map.get(request.period, PerformancePeriod.DAILY)

        tracker.record_snapshot(
            metrics=metrics,
            period=period,
            market_condition=request.market_condition,
            regime=request.regime,
        )

        # Save to database
        db = PerformanceDatabase()
        db.save_metrics(
            strategy_id=request.strategy_id,
            metrics=metrics,
        )

        # Return response
        return PerformanceMetricsResponse(
            strategy_id=request.strategy_id,
            total_pnl=metrics.total_pnl,
            total_return=metrics.total_return,
            sharpe_ratio=metrics.sharpe_ratio,
            max_drawdown=metrics.max_drawdown,
            total_trades=metrics.total_trades,
            win_rate=metrics.win_rate,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Performance tracking failed: {str(e)}"
        )


@router.post("/compare", response_model=PerformanceComparisonResponse)
async def compare_performance(request: ComparePerformanceRequest):
    """
    Compare live vs backtest performance.

    This endpoint:
    1. Calculates metrics for live and backtest trades
    2. Performs comparison analysis
    3. Detects performance degradation
    4. Returns status and warnings

    Args:
        request: Comparison request

    Returns:
        Performance comparison with status
    """
    try:
        import pandas as pd
        from finantradealgo.research.performance import (
            PerformanceTracker,
            PerformanceComparator,
        )

        # Convert trades to DataFrames
        live_df = pd.DataFrame(request.live_trades)
        backtest_df = pd.DataFrame(request.backtest_trades)

        if live_df.empty or backtest_df.empty:
            raise HTTPException(status_code=400, detail="Insufficient trade data")

        # Create trackers
        live_tracker = PerformanceTracker(strategy_id=f"{request.strategy_id}_live", is_live=True)
        backtest_tracker = PerformanceTracker(strategy_id=f"{request.strategy_id}_backtest", is_live=False)

        # Calculate metrics
        live_metrics = live_tracker._calculate_metrics_from_trades(live_df)
        backtest_metrics = backtest_tracker._calculate_metrics_from_trades(backtest_df)

        # Compare
        comparator = PerformanceComparator()
        comparison = comparator.compare(
            strategy_id=request.strategy_id,
            live_metrics=live_metrics,
            backtest_metrics=backtest_metrics,
        )

        # Return response
        return PerformanceComparisonResponse(
            strategy_id=comparison.strategy_id,
            comparison_time=comparison.comparison_time.isoformat(),
            live_sharpe=comparison.live_metrics.sharpe_ratio,
            backtest_sharpe=comparison.backtest_metrics.sharpe_ratio,
            sharpe_delta=comparison.sharpe_delta,
            sharpe_deviation_pct=comparison.sharpe_deviation_pct,
            overall_status=comparison.overall_status.value,
            warnings=comparison.warnings,
            critical_issues=comparison.critical_issues,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Performance comparison failed: {str(e)}"
        )


@router.post("/alerts/subscribe", response_model=AlertSubscriptionResponse)
async def subscribe_to_alerts(request: SubscribeAlertsRequest):
    """
    Subscribe to performance alerts.

    This endpoint:
    1. Creates alert subscription for strategy
    2. Configures delivery channels
    3. Sets up filters and rate limiting

    Args:
        request: Subscription request

    Returns:
        Subscription confirmation
    """
    try:
        from finantradealgo.research.performance.alerts import (
            get_alert_manager,
            AlertChannel,
        )

        # Convert channel strings to enums
        channel_map = {
            "console": AlertChannel.CONSOLE,
            "file": AlertChannel.FILE,
            "slack": AlertChannel.SLACK,
            "webhook": AlertChannel.WEBHOOK,
        }

        channels = []
        for channel_str in request.channels:
            if channel_str in channel_map:
                channels.append(channel_map[channel_str])
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid channel: {channel_str}"
                )

        # Create subscription
        manager = get_alert_manager()
        subscription = manager.subscribe(
            strategy_id=request.strategy_id,
            channels=channels,
            severity_filter=request.severity_filter,
            alert_type_filter=request.alert_type_filter,
            min_interval_minutes=request.min_interval_minutes,
        )

        return AlertSubscriptionResponse(
            success=True,
            strategy_id=subscription.strategy_id,
            channels=request.channels,
            message=f"Successfully subscribed to alerts for {request.strategy_id}",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Alert subscription failed: {str(e)}"
        )


@router.get("/alerts/history/{strategy_id}")
async def get_alert_history(
    strategy_id: str,
    hours: Optional[int] = 24,
    severity: Optional[str] = None,
):
    """
    Get alert history for strategy.

    Args:
        strategy_id: Strategy identifier
        hours: Time window in hours
        severity: Filter by severity

    Returns:
        List of recent alerts
    """
    try:
        from finantradealgo.research.performance.alerts import get_alert_manager

        manager = get_alert_manager()
        alerts = manager.get_alert_history(
            strategy_id=strategy_id,
            hours=hours,
            severity=severity,
        )

        return {
            "strategy_id": strategy_id,
            "hours": hours,
            "total_alerts": len(alerts),
            "alerts": [
                {
                    "alert_time": a.alert_time.isoformat(),
                    "alert_type": a.alert_type,
                    "severity": a.severity,
                    "message": a.message,
                    "recommended_action": a.recommended_action,
                }
                for a in alerts
            ],
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve alert history: {str(e)}"
        )


@router.post("/attribution", response_model=AttributionResponse)
async def analyze_attribution(request: AttributionAnalysisRequest):
    """
    Analyze performance attribution.

    This endpoint:
    1. Breaks down PnL by contributing factors
    2. Identifies top/worst trades
    3. Analyzes regime, symbol, component contributions

    Args:
        request: Attribution request

    Returns:
        Attribution analysis
    """
    try:
        import pandas as pd
        from finantradealgo.research.performance.attribution import AttributionAnalyzer

        # Convert trades to DataFrame
        trades_df = pd.DataFrame(request.trades)

        if trades_df.empty:
            raise HTTPException(status_code=400, detail="No trades provided")

        # Analyze attribution
        analyzer = AttributionAnalyzer(strategy_id=request.strategy_id)
        attribution = analyzer.analyze_trades(
            trades_df=trades_df,
            group_by=request.group_by,
        )

        return AttributionResponse(
            strategy_id=attribution.strategy_id,
            total_pnl=attribution.total_pnl,
            pnl_by_regime=attribution.pnl_by_regime if attribution.pnl_by_regime else None,
            pnl_by_symbol=attribution.pnl_by_symbol if attribution.pnl_by_symbol else None,
            pnl_by_component=attribution.pnl_by_component if attribution.pnl_by_component else None,
            top_trades=attribution.top_trades,
            worst_trades=attribution.worst_trades,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Attribution analysis failed: {str(e)}"
        )


@router.get("/summary/{strategy_id}")
async def get_performance_summary(strategy_id: str):
    """
    Get comprehensive performance summary.

    Args:
        strategy_id: Strategy identifier

    Returns:
        Performance summary with current metrics, trends, alerts
    """
    try:
        from finantradealgo.research.performance import (
            PerformanceTracker,
            MetricsAggregator,
        )
        from finantradealgo.research.performance.alerts import get_alert_manager

        # Load tracker
        tracker = PerformanceTracker(strategy_id=strategy_id, is_live=True)

        if not tracker.current_metrics:
            raise HTTPException(status_code=404, detail=f"No performance data for {strategy_id}")

        # Get aggregator
        aggregator = MetricsAggregator(tracker)

        # Get performance report
        performance_report = aggregator.generate_performance_report()

        # Get alert summary
        alert_manager = get_alert_manager()
        alert_summary = alert_manager.get_alert_summary(hours=24)

        # Combine
        summary = {
            "strategy_id": strategy_id,
            "timestamp": datetime.utcnow().isoformat(),
            "performance": performance_report,
            "alerts_24h": alert_summary,
        }

        return summary

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate performance summary: {str(e)}"
        )


@router.get("/list")
async def list_tracked_strategies():
    """
    List all strategies being tracked.

    Returns:
        List of strategy IDs with performance data
    """
    try:
        from finantradealgo.research.performance import PerformanceDatabase

        db = PerformanceDatabase()
        strategy_ids = db.list_strategies()

        return {
            "total_strategies": len(strategy_ids),
            "strategy_ids": strategy_ids,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list strategies: {str(e)}"
        )


def _resolve_live_snapshot_path(run_id: str | None) -> Path:
    cfg = load_config("research")
    live_cfg = cfg.get("live", {}) or {}
    live_dir = Path(live_cfg.get("state_dir", "outputs/live"))
    latest_path = live_cfg.get("latest_state_path") or live_cfg.get("state_path") or live_dir / "live_state.json"
    if run_id:
        return Path(live_cfg.get("state_path") or live_dir / f"live_state_{run_id}.json")
    return Path(latest_path)


def _resolve_live_trades_path(run_id: str | None) -> Optional[Path]:
    cfg = load_config("research")
    live_cfg = cfg.get("live", {}) or {}
    live_dir = Path(live_cfg.get("state_dir", "outputs/live"))
    trades_override = live_cfg.get("trades_path")
    if trades_override:
        return Path(trades_override)
    if run_id:
        path = live_dir / f"trades_{run_id}.csv"
        if path.exists():
            return path
    default_path = live_dir / "trades_latest.csv"
    return default_path if default_path.exists() else None


@router.get("/live/report")
async def live_report(format: str = "html", run_id: Optional[str] = None):
    """
    Generate live performance/health report from latest snapshot and recent trades.
    """
    snapshot_path = _resolve_live_snapshot_path(run_id)
    if not snapshot_path.exists():
        raise HTTPException(status_code=404, detail=f"Live snapshot not found: {snapshot_path}")

    trades_path = _resolve_live_trades_path(run_id)

    try:
        generator = LiveReportGenerator()
        report = generator.generate(
            snapshot=snapshot_path,
            trades=trades_path if trades_path else None,
            snapshot_path=snapshot_path,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc))

    fmt = format.lower()
    if fmt in ("markdown", "md"):
        content = report.to_markdown()
        return {"run_id": run_id or report.run_id, "format": "markdown", "content": content}
    if fmt == "json":
        return {"run_id": run_id or report.run_id, "format": "json", "content": report.to_dict()}

    content = report.to_html()
    return {"run_id": run_id or report.run_id, "format": "html", "content": content}
