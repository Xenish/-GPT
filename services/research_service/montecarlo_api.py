"""
Monte Carlo API Endpoints.

REST API for Monte Carlo simulation and risk analysis.
"""

from __future__ import annotations

from typing import List, Dict, Any
from pathlib import Path

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
import pandas as pd

from finantradealgo.research.montecarlo import (
    MonteCarloConfig,
    BootstrapResampler,
    RiskMetricsCalculator,
    ResamplingMethod,
)

router = APIRouter()


class MonteCarloRequest(BaseModel):
    """Request for Monte Carlo simulation."""
    strategy_id: str
    trades: List[Dict[str, Any]]
    n_simulations: int = Field(1000, description="Number of simulations")
    resampling_method: str = Field("bootstrap", description="Resampling method")
    confidence_level: float = Field(0.95, description="Confidence level")


@router.post("/run")
async def run_monte_carlo(request: MonteCarloRequest):
    """Run Monte Carlo simulation."""
    try:
        trades_df = pd.DataFrame(request.trades)

        config = MonteCarloConfig(
            n_simulations=request.n_simulations,
            resampling_method=ResamplingMethod(request.resampling_method),
            confidence_level=request.confidence_level,
        )

        resampler = BootstrapResampler(config)
        result = resampler.run_monte_carlo(request.strategy_id, trades_df)

        # Calculate risk assessment
        risk_calc = RiskMetricsCalculator()
        risk_assessment = risk_calc.calculate_risk_assessment(result)

        summary = result.to_summary_dict()
        metrics = {
            "median_return": summary.get("median_return"),
            "p5_return": summary.get("p5_return"),
            "p95_return": summary.get("p95_return"),
            "worst_case_dd": summary.get("worst_drawdown"),
        }

        return {
            "success": True,
            "summary": summary,
            "risk_assessment": risk_assessment.to_dict(),
            "metrics": metrics,
        }

    except Exception as e:
        # Surface clear error for tests/clients
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/position-size")
async def calculate_position_size(
    trades: List[Dict[str, Any]] = Body(...),
    max_risk_pct: float = Body(2.0),
):
    """Calculate optimal position sizing."""
    try:
        trades_df = pd.DataFrame(trades)
        config = MonteCarloConfig(n_simulations=1000)

        resampler = BootstrapResampler(config)
        result = resampler.run_monte_carlo("temp", trades_df)

        risk_calc = RiskMetricsCalculator()
        sizing = risk_calc.calculate_optimal_position_size(result, max_risk_pct)

        return {"success": True, "position_sizing": sizing}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
