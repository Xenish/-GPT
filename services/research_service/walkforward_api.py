"""
Walk-Forward Optimization API Endpoints.

REST API for walk-forward analysis and validation.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
import pandas as pd

from finantradealgo.research.walkforward import (
    WalkForwardConfig,
    WalkForwardOptimizer,
    OutOfSampleValidator,
    WalkForwardAnalyzer,
    WalkForwardVisualizer,
    WindowType,
    OptimizationMetric,
)


router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================


class WalkForwardRequest(BaseModel):
    """Request to run walk-forward optimization."""

    strategy_id: str = Field(..., description="Strategy identifier")
    param_grid: Dict[str, List[Any]] = Field(..., description="Parameter grid to search")
    data: List[Dict[str, Any]] = Field(..., description="Price data with datetime index")

    # Walk-forward config
    in_sample_periods: int = Field(12, description="In-sample periods")
    out_sample_periods: int = Field(3, description="Out-of-sample periods")
    window_type: str = Field("rolling", description="Window type (rolling/anchored)")
    period_unit: str = Field("M", description="Period unit (D/W/M/Q/Y)")
    optimization_metric: str = Field("sharpe_ratio", description="Optimization metric")

    # Constraints
    min_trades_per_period: int = Field(10, description="Minimum trades per period")
    require_profitable_is: bool = Field(False, description="Require profitable IS")


class ValidationRequest(BaseModel):
    """Request to validate walk-forward result."""

    result_id: str = Field(..., description="Walk-forward result ID")

    # Validation thresholds
    max_sharpe_degradation: float = Field(0.5, description="Max acceptable degradation")
    min_oos_win_rate: float = Field(0.4, description="Min OOS win rate")
    min_oos_sharpe: float = Field(0.5, description="Min OOS Sharpe")


class AnalysisRequest(BaseModel):
    """Request for walk-forward analysis."""

    result_id: str = Field(..., description="Walk-forward result ID")
    analysis_types: List[str] = Field(
        ["efficiency", "regime_sensitivity", "param_drift"],
        description="Analysis types to perform",
    )


class VisualizationRequest(BaseModel):
    """Request to generate walk-forward visualization."""

    result_id: str = Field(..., description="Walk-forward result ID")
    chart_type: str = Field(
        "performance_comparison",
        description="Chart type (performance_comparison, equity_curve, parameter_stability, robustness_dashboard)",
    )
    output_format: str = Field("html", description="Output format")


class WalkForwardResponse(BaseModel):
    """Response for walk-forward operation."""

    success: bool
    result_id: str
    message: str
    summary: Optional[Dict[str, Any]] = None


# ============================================================================
# Helper Functions
# ============================================================================


def _get_wf_dir() -> Path:
    """Get walk-forward results directory."""
    wf_dir = Path("data/walkforward")
    wf_dir.mkdir(parents=True, exist_ok=True)
    return wf_dir


def _generate_result_id(strategy_id: str) -> str:
    """Generate unique result ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{strategy_id}_wf_{timestamp}"


def _save_result(result, result_id: str):
    """Save walk-forward result."""
    result_dir = _get_wf_dir() / result_id
    result_dir.mkdir(parents=True, exist_ok=True)

    optimizer = WalkForwardOptimizer(result.config)
    optimizer.save_result(result, result_dir)

    return result_dir


def _load_result(result_id: str):
    """Load walk-forward result from disk."""
    import json

    result_dir = _get_wf_dir() / result_id

    if not result_dir.exists():
        raise HTTPException(status_code=404, detail=f"Result not found: {result_id}")

    # Load summary
    summary_path = result_dir / "summary.json"
    with open(summary_path, "r") as f:
        summary = json.load(f)

    # Load windows
    windows_path = result_dir / "windows.json"
    with open(windows_path, "r") as f:
        windows_data = json.load(f)

    return summary, windows_data


# ============================================================================
# Walk-Forward Optimization Endpoints
# ============================================================================


@router.post("/run", response_model=WalkForwardResponse)
async def run_walk_forward(request: WalkForwardRequest):
    """
    Run walk-forward optimization.

    Performs rolling or anchored window optimization with out-of-sample testing.
    """
    try:
        # Create configuration
        config = WalkForwardConfig(
            in_sample_periods=request.in_sample_periods,
            out_sample_periods=request.out_sample_periods,
            window_type=WindowType(request.window_type),
            period_unit=request.period_unit,
            optimization_metric=OptimizationMetric(request.optimization_metric),
            min_trades_per_period=request.min_trades_per_period,
            require_profitable_is=request.require_profitable_is,
        )

        # Convert data to DataFrame
        data_df = pd.DataFrame(request.data)
        if 'datetime' in data_df.columns:
            data_df['datetime'] = pd.to_datetime(data_df['datetime'])
            data_df = data_df.set_index('datetime')

        # TODO: Need backtest function - this would be provided by user
        # For now, return error message
        raise HTTPException(
            status_code=501,
            detail="Walk-forward optimization requires a backtest function. "
                   "This endpoint needs to be integrated with your strategy execution engine.",
        )

        # Example of how it would work:
        # optimizer = WalkForwardOptimizer(config)
        # result = optimizer.run_walk_forward(
        #     strategy_id=request.strategy_id,
        #     param_grid=request.param_grid,
        #     data_df=data_df,
        #     backtest_function=backtest_fn,
        # )
        #
        # result_id = _generate_result_id(request.strategy_id)
        # _save_result(result, result_id)
        #
        # return WalkForwardResponse(
        #     success=True,
        #     result_id=result_id,
        #     message="Walk-forward optimization completed",
        #     summary=result.to_summary_dict(),
        # )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Walk-forward optimization failed: {str(e)}")


@router.get("/list")
async def list_walk_forward_results():
    """
    List all walk-forward results.

    Returns list of available results with metadata.
    """
    try:
        wf_dir = _get_wf_dir()
        results = []

        for result_dir in wf_dir.iterdir():
            if result_dir.is_dir():
                summary_path = result_dir / "summary.json"
                if summary_path.exists():
                    import json
                    with open(summary_path, "r") as f:
                        summary = json.load(f)

                    results.append({
                        "result_id": result_dir.name,
                        "strategy_id": summary.get("strategy_id"),
                        "total_windows": summary.get("total_windows"),
                        "avg_oos_sharpe": summary.get("avg_oos_sharpe"),
                        "consistency_score": summary.get("consistency_score"),
                    })

        return {
            "success": True,
            "count": len(results),
            "results": results,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list results: {str(e)}")


@router.get("/result/{result_id}")
async def get_walk_forward_result(result_id: str):
    """
    Get detailed walk-forward result.

    Returns complete result including all windows.
    """
    try:
        summary, windows_data = _load_result(result_id)

        return {
            "success": True,
            "result_id": result_id,
            "summary": summary,
            "windows": windows_data,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load result: {str(e)}")


# ============================================================================
# Validation Endpoints
# ============================================================================


@router.post("/validate")
async def validate_walk_forward(request: ValidationRequest):
    """
    Validate walk-forward result.

    Assesses strategy robustness based on OOS performance.
    """
    try:
        summary, windows_data = _load_result(request.result_id)

        # Create validator
        validator = OutOfSampleValidator(
            max_sharpe_degradation=request.max_sharpe_degradation,
            min_oos_win_rate=request.min_oos_win_rate,
            min_oos_sharpe=request.min_oos_sharpe,
        )

        # TODO: Need to reconstruct WalkForwardResult from saved data
        # For now, return summary-based validation

        return {
            "success": True,
            "result_id": request.result_id,
            "validation": {
                "status": "requires_implementation",
                "message": "Full validation requires WalkForwardResult reconstruction",
                "quick_assessment": {
                    "avg_oos_sharpe": summary.get("avg_oos_sharpe"),
                    "consistency_score": summary.get("consistency_score"),
                    "param_stability": summary.get("param_stability_score"),
                },
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


# ============================================================================
# Analysis Endpoints
# ============================================================================


@router.post("/analyze")
async def analyze_walk_forward(request: AnalysisRequest):
    """
    Perform detailed walk-forward analysis.

    Calculates efficiency metrics, regime sensitivity, parameter drift, etc.
    """
    try:
        summary, windows_data = _load_result(request.result_id)

        analyzer = WalkForwardAnalyzer()

        # Return available summary data
        analysis_result = {
            "result_id": request.result_id,
            "summary": summary,
            "windows_count": len(windows_data),
            "requested_analyses": request.analysis_types,
            "note": "Full analysis requires WalkForwardResult reconstruction",
        }

        return {
            "success": True,
            "analysis": analysis_result,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ============================================================================
# Visualization Endpoints
# ============================================================================


@router.post("/visualize")
async def visualize_walk_forward(request: VisualizationRequest):
    """
    Generate walk-forward visualization.

    Creates interactive charts showing IS/OOS performance, degradation, etc.
    """
    try:
        summary, windows_data = _load_result(request.result_id)

        # Return placeholder
        return {
            "success": False,
            "message": "Visualization requires WalkForwardResult reconstruction",
            "result_id": request.result_id,
            "chart_type": request.chart_type,
            "data_available": {
                "summary": summary,
                "windows": len(windows_data),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")


# ============================================================================
# Utility Endpoints
# ============================================================================


@router.delete("/delete/{result_id}")
async def delete_walk_forward_result(result_id: str):
    """
    Delete walk-forward result.
    """
    try:
        result_dir = _get_wf_dir() / result_id

        if not result_dir.exists():
            raise HTTPException(status_code=404, detail="Result not found")

        # Delete directory
        import shutil
        shutil.rmtree(result_dir)

        return {
            "success": True,
            "message": f"Result '{result_id}' deleted successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete result: {str(e)}")


@router.get("/windows/{result_id}")
async def get_walk_forward_windows(result_id: str):
    """
    Get detailed window information.

    Returns IS/OOS metrics for each window.
    """
    try:
        summary, windows_data = _load_result(result_id)

        return {
            "success": True,
            "result_id": result_id,
            "total_windows": len(windows_data),
            "windows": windows_data,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get windows: {str(e)}")
