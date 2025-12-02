"""
Visualization API Endpoints.

REST API for generating and serving research visualizations.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
import json

from fastapi import APIRouter, HTTPException, Query, Body
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
import pandas as pd

from finantradealgo.research.visualization import (
    EquityCurveVisualizer,
    ParameterHeatmapVisualizer,
    TradeAnalysisVisualizer,
    StrategyDashboard,
    ParameterSearchDashboard,
    ComparisonDashboard,
    ChartConfig,
)
from finantradealgo.research.performance.models import PerformanceMetrics


router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================


class EquityCurveRequest(BaseModel):
    """Request to generate equity curve visualization."""

    trades_df: List[Dict[str, Any]] = Field(..., description="List of trade records")
    starting_capital: float = Field(10000.0, description="Starting capital")
    show_trades: bool = Field(True, description="Show individual trade markers")
    show_drawdown: bool = Field(True, description="Show drawdown subplot")
    output_format: str = Field("html", description="Output format (html, png, svg)")


class TradeAnalysisRequest(BaseModel):
    """Request to generate trade analysis visualization."""

    trades_df: List[Dict[str, Any]] = Field(..., description="List of trade records")
    chart_type: str = Field(
        "overview",
        description="Chart type (overview, pnl_by_time, win_loss, consecutive, sizes)",
    )
    group_by: Optional[str] = Field(None, description="Time grouping (hour, day, month)")
    output_format: str = Field("html", description="Output format")


class ParameterHeatmapRequest(BaseModel):
    """Request to generate parameter heatmap."""

    results_df: List[Dict[str, Any]] = Field(..., description="Parameter search results")
    param1: str = Field(..., description="First parameter")
    param2: str = Field(..., description="Second parameter")
    metric: str = Field("sharpe_ratio", description="Target metric")
    output_format: str = Field("html", description="Output format")


class StrategyDashboardRequest(BaseModel):
    """Request to generate strategy dashboard."""

    trades_df: List[Dict[str, Any]] = Field(..., description="List of trade records")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Performance metrics")
    dashboard_type: str = Field("full", description="Dashboard type (full, summary)")
    starting_capital: float = Field(10000.0, description="Starting capital")
    output_format: str = Field("html", description="Output format")


class ParameterSearchDashboardRequest(BaseModel):
    """Request to generate parameter search dashboard."""

    results_df: List[Dict[str, Any]] = Field(..., description="Parameter search results")
    metric: str = Field("sharpe_ratio", description="Target metric")
    param1: Optional[str] = Field(None, description="First parameter (auto-detect if None)")
    param2: Optional[str] = Field(None, description="Second parameter (auto-detect if None)")
    output_format: str = Field("html", description="Output format")


class ComparisonDashboardRequest(BaseModel):
    """Request to generate strategy comparison dashboard."""

    strategies_data: Dict[str, List[Dict[str, Any]]] = Field(
        ..., description="Dict mapping strategy names to trade records"
    )
    starting_capital: float = Field(10000.0, description="Starting capital")
    output_format: str = Field("html", description="Output format")


class VisualizationResponse(BaseModel):
    """Response for visualization generation."""

    success: bool
    file_path: Optional[str] = None
    format: str
    message: str
    visualization_id: str


# ============================================================================
# Helper Functions
# ============================================================================


def _get_viz_dir() -> Path:
    """Get visualization output directory."""
    viz_dir = Path("data/visualizations")
    viz_dir.mkdir(parents=True, exist_ok=True)
    return viz_dir


def _generate_viz_id() -> str:
    """Generate unique visualization ID."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _df_from_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert list of records to DataFrame."""
    return pd.DataFrame(records)


def _save_visualization(
    fig,
    viz_type: str,
    viz_id: str,
    output_format: str = "html",
) -> Path:
    """Save visualization to file."""
    viz_dir = _get_viz_dir()
    filename = viz_dir / f"{viz_type}_{viz_id}.{output_format}"

    if output_format == "html":
        fig.write_html(str(filename))
    elif output_format in ["png", "svg", "pdf"]:
        fig.write_image(str(filename), format=output_format)
    else:
        raise ValueError(f"Unsupported format: {output_format}")

    return filename


# ============================================================================
# Equity Curve Endpoints
# ============================================================================


@router.post("/equity-curve", response_model=VisualizationResponse)
async def generate_equity_curve(request: EquityCurveRequest):
    """
    Generate equity curve visualization.

    Creates an interactive equity curve with optional drawdown and trade markers.
    """
    try:
        trades_df = _df_from_records(request.trades_df)
        viz_id = _generate_viz_id()

        visualizer = EquityCurveVisualizer()
        fig = visualizer.plot_equity_curve(
            trades_df=trades_df,
            starting_capital=request.starting_capital,
            show_trades=request.show_trades,
            show_drawdown=request.show_drawdown,
        )

        file_path = _save_visualization(fig, "equity_curve", viz_id, request.output_format)

        return VisualizationResponse(
            success=True,
            file_path=str(file_path),
            format=request.output_format,
            message="Equity curve generated successfully",
            visualization_id=viz_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate equity curve: {str(e)}")


@router.post("/returns-distribution", response_model=VisualizationResponse)
async def generate_returns_distribution(
    trades_df: List[Dict[str, Any]] = Body(...),
    bins: int = Body(50),
    output_format: str = Body("html"),
):
    """
    Generate returns distribution visualization.

    Creates histogram and box plot of trade returns.
    """
    try:
        df = _df_from_records(trades_df)
        viz_id = _generate_viz_id()

        visualizer = EquityCurveVisualizer()
        fig = visualizer.plot_returns_distribution(df, bins=bins)

        file_path = _save_visualization(fig, "returns_dist", viz_id, output_format)

        return VisualizationResponse(
            success=True,
            file_path=str(file_path),
            format=output_format,
            message="Returns distribution generated successfully",
            visualization_id=viz_id,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate returns distribution: {str(e)}"
        )


@router.post("/rolling-metrics", response_model=VisualizationResponse)
async def generate_rolling_metrics(
    trades_df: List[Dict[str, Any]] = Body(...),
    window: int = Body(20),
    metrics: List[str] = Body(["sharpe", "win_rate"]),
    output_format: str = Body("html"),
):
    """
    Generate rolling metrics visualization.

    Plots rolling Sharpe ratio, win rate, and other metrics.
    """
    try:
        df = _df_from_records(trades_df)
        viz_id = _generate_viz_id()

        visualizer = EquityCurveVisualizer()
        fig = visualizer.plot_rolling_metrics(df, window=window, metrics=metrics)

        file_path = _save_visualization(fig, "rolling_metrics", viz_id, output_format)

        return VisualizationResponse(
            success=True,
            file_path=str(file_path),
            format=output_format,
            message="Rolling metrics generated successfully",
            visualization_id=viz_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate rolling metrics: {str(e)}")


# ============================================================================
# Trade Analysis Endpoints
# ============================================================================


@router.post("/trade-analysis", response_model=VisualizationResponse)
async def generate_trade_analysis(request: TradeAnalysisRequest):
    """
    Generate trade analysis visualization.

    Supports multiple chart types: overview, pnl_by_time, win_loss, consecutive, sizes.
    """
    try:
        trades_df = _df_from_records(request.trades_df)
        viz_id = _generate_viz_id()

        visualizer = TradeAnalysisVisualizer()

        if request.chart_type == "overview":
            fig = visualizer.plot_trade_overview(trades_df)
        elif request.chart_type == "pnl_by_time":
            fig = visualizer.plot_pnl_by_time(
                trades_df, group_by=request.group_by or "hour"
            )
        elif request.chart_type == "win_loss":
            fig = visualizer.plot_win_loss_analysis(trades_df)
        elif request.chart_type == "consecutive":
            fig = visualizer.plot_consecutive_analysis(trades_df)
        elif request.chart_type == "sizes":
            fig = visualizer.plot_trade_sizes(trades_df)
        else:
            raise ValueError(f"Unknown chart type: {request.chart_type}")

        file_path = _save_visualization(
            fig, f"trade_{request.chart_type}", viz_id, request.output_format
        )

        return VisualizationResponse(
            success=True,
            file_path=str(file_path),
            format=request.output_format,
            message=f"Trade analysis ({request.chart_type}) generated successfully",
            visualization_id=viz_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate trade analysis: {str(e)}")


# ============================================================================
# Parameter Heatmap Endpoints
# ============================================================================


@router.post("/parameter-heatmap", response_model=VisualizationResponse)
async def generate_parameter_heatmap(request: ParameterHeatmapRequest):
    """
    Generate 2D parameter heatmap.

    Visualizes parameter space as a heatmap colored by target metric.
    """
    try:
        results_df = _df_from_records(request.results_df)
        viz_id = _generate_viz_id()

        visualizer = ParameterHeatmapVisualizer()
        fig = visualizer.plot_2d_heatmap(
            results_df=results_df,
            param1=request.param1,
            param2=request.param2,
            metric=request.metric,
        )

        file_path = _save_visualization(fig, "param_heatmap", viz_id, request.output_format)

        return VisualizationResponse(
            success=True,
            file_path=str(file_path),
            format=request.output_format,
            message="Parameter heatmap generated successfully",
            visualization_id=viz_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate parameter heatmap: {str(e)}")


@router.post("/parameter-sensitivity", response_model=VisualizationResponse)
async def generate_parameter_sensitivity(
    results_df: List[Dict[str, Any]] = Body(...),
    metric: str = Body("sharpe_ratio"),
    output_format: str = Body("html"),
):
    """
    Generate parameter sensitivity analysis.

    Shows how each parameter affects the target metric.
    """
    try:
        df = _df_from_records(results_df)
        viz_id = _generate_viz_id()

        visualizer = ParameterHeatmapVisualizer()
        fig = visualizer.plot_parameter_sensitivity(df, metric=metric)

        file_path = _save_visualization(fig, "param_sensitivity", viz_id, output_format)

        return VisualizationResponse(
            success=True,
            file_path=str(file_path),
            format=output_format,
            message="Parameter sensitivity analysis generated successfully",
            visualization_id=viz_id,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate parameter sensitivity: {str(e)}"
        )


# ============================================================================
# Dashboard Endpoints
# ============================================================================


@router.post("/dashboard/strategy", response_model=VisualizationResponse)
async def generate_strategy_dashboard(request: StrategyDashboardRequest):
    """
    Generate comprehensive strategy dashboard.

    Combines equity curve, trade analysis, and metrics into a single view.
    """
    try:
        trades_df = _df_from_records(request.trades_df)
        viz_id = _generate_viz_id()

        dashboard = StrategyDashboard()

        # Convert metrics dict to PerformanceMetrics if provided
        metrics = None
        if request.metrics:
            metrics = PerformanceMetrics(**request.metrics)

        if request.dashboard_type == "full":
            fig = dashboard.create_full_dashboard(
                trades_df=trades_df,
                metrics=metrics,
                starting_capital=request.starting_capital,
            )
        elif request.dashboard_type == "summary":
            if not metrics:
                raise ValueError("Metrics required for summary dashboard")
            fig = dashboard.create_quick_summary(trades_df=trades_df, metrics=metrics)
        else:
            raise ValueError(f"Unknown dashboard type: {request.dashboard_type}")

        file_path = _save_visualization(
            fig, f"strategy_dashboard_{request.dashboard_type}", viz_id, request.output_format
        )

        return VisualizationResponse(
            success=True,
            file_path=str(file_path),
            format=request.output_format,
            message=f"Strategy dashboard ({request.dashboard_type}) generated successfully",
            visualization_id=viz_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate strategy dashboard: {str(e)}")


@router.post("/dashboard/parameter-search", response_model=VisualizationResponse)
async def generate_parameter_search_dashboard(request: ParameterSearchDashboardRequest):
    """
    Generate parameter search dashboard.

    Visualizes parameter optimization results with heatmaps and rankings.
    """
    try:
        results_df = _df_from_records(request.results_df)
        viz_id = _generate_viz_id()

        dashboard = ParameterSearchDashboard()
        fig = dashboard.create_search_dashboard(
            results_df=results_df,
            metric=request.metric,
            param1=request.param1,
            param2=request.param2,
        )

        file_path = _save_visualization(
            fig, "param_search_dashboard", viz_id, request.output_format
        )

        return VisualizationResponse(
            success=True,
            file_path=str(file_path),
            format=request.output_format,
            message="Parameter search dashboard generated successfully",
            visualization_id=viz_id,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate parameter search dashboard: {str(e)}"
        )


@router.post("/dashboard/comparison", response_model=VisualizationResponse)
async def generate_comparison_dashboard(request: ComparisonDashboardRequest):
    """
    Generate strategy comparison dashboard.

    Compares multiple strategies side by side.
    """
    try:
        # Convert strategies data to DataFrames
        strategies_data = {
            name: _df_from_records(trades)
            for name, trades in request.strategies_data.items()
        }

        viz_id = _generate_viz_id()

        dashboard = ComparisonDashboard()
        fig = dashboard.create_comparison_dashboard(
            strategies_data=strategies_data,
            starting_capital=request.starting_capital,
        )

        file_path = _save_visualization(
            fig, "comparison_dashboard", viz_id, request.output_format
        )

        return VisualizationResponse(
            success=True,
            file_path=str(file_path),
            format=request.output_format,
            message="Comparison dashboard generated successfully",
            visualization_id=viz_id,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate comparison dashboard: {str(e)}"
        )


# ============================================================================
# Utility Endpoints
# ============================================================================


@router.get("/list")
async def list_visualizations():
    """
    List all generated visualizations.

    Returns a list of available visualizations with metadata.
    """
    try:
        viz_dir = _get_viz_dir()
        visualizations = []

        for file_path in viz_dir.glob("*"):
            if file_path.is_file():
                stat = file_path.stat()
                visualizations.append({
                    "filename": file_path.name,
                    "size_bytes": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "format": file_path.suffix[1:],
                })

        return {
            "success": True,
            "count": len(visualizations),
            "visualizations": sorted(
                visualizations, key=lambda x: x["created_at"], reverse=True
            ),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list visualizations: {str(e)}")


@router.get("/view/{filename}")
async def view_visualization(filename: str):
    """
    View a specific visualization.

    Returns the HTML file for browser display.
    """
    try:
        viz_dir = _get_viz_dir()
        file_path = viz_dir / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Visualization not found")

        if file_path.suffix == ".html":
            return FileResponse(file_path, media_type="text/html")
        elif file_path.suffix == ".png":
            return FileResponse(file_path, media_type="image/png")
        elif file_path.suffix == ".svg":
            return FileResponse(file_path, media_type="image/svg+xml")
        elif file_path.suffix == ".pdf":
            return FileResponse(file_path, media_type="application/pdf")
        else:
            return FileResponse(file_path)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to view visualization: {str(e)}")


@router.delete("/delete/{filename}")
async def delete_visualization(filename: str):
    """
    Delete a specific visualization.
    """
    try:
        viz_dir = _get_viz_dir()
        file_path = viz_dir / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Visualization not found")

        file_path.unlink()

        return {
            "success": True,
            "message": f"Visualization '{filename}' deleted successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete visualization: {str(e)}")
