"""
Scenario analysis API endpoints.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from finantradealgo.backtester.scenario_engine import Scenario, run_scenarios
from finantradealgo.system.config_loader import load_system_config

router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================

class ScenarioRequest(BaseModel):
    """Single scenario specification."""
    label: str = Field(..., description="Scenario label/name")
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe (e.g., 15m, 1h)")
    strategy: str = Field(..., description="Strategy name")
    params: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
    risk_params: Optional[Dict[str, Any]] = Field(None, description="Risk parameters override")
    feature_preset: Optional[str] = Field(None, description="Feature preset override")


class RunScenariosRequest(BaseModel):
    """Request to run multiple scenarios."""
    scenarios: List[ScenarioRequest] = Field(..., description="List of scenarios to run")
    base_symbol: Optional[str] = Field(None, description="Base symbol (if not specified per scenario)")
    base_timeframe: Optional[str] = Field(None, description="Base timeframe")


class ScenarioResult(BaseModel):
    """Result of a single scenario."""
    scenario_id: str
    label: str
    symbol: str
    timeframe: str
    strategy: str
    params: Dict[str, Any]
    cum_return: float
    sharpe: float
    max_drawdown: float
    trade_count: int


class RunScenariosResponse(BaseModel):
    """Response from running scenarios."""
    n_scenarios: int
    results: List[ScenarioResult]


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/run", response_model=RunScenariosResponse)
async def run_scenarios_endpoint(request: RunScenariosRequest):
    """
    Run multiple backtest scenarios and compare results.

    This endpoint allows comparing different strategies, parameters,
    or configurations side-by-side.
    """
    if not request.scenarios:
        raise HTTPException(status_code=400, detail="At least one scenario required")

    # Load system config
    try:
        sys_cfg = load_system_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Config load failed: {str(e)}")

    # Convert request to Scenario objects
    scenarios = []
    for sc_req in request.scenarios:
        scenario = Scenario(
            symbol=sc_req.symbol or request.base_symbol,
            timeframe=sc_req.timeframe or request.base_timeframe,
            strategy=sc_req.strategy,
            params=sc_req.params,
            label=sc_req.label,
            feature_preset=sc_req.feature_preset,
            risk_params=sc_req.risk_params,
        )

        # Validate scenario has required fields
        if not scenario.symbol or not scenario.timeframe:
            raise HTTPException(
                status_code=400,
                detail=f"Scenario '{sc_req.label}' missing symbol or timeframe"
            )

        scenarios.append(scenario)

    # Run scenarios
    try:
        results_df = run_scenarios(sys_cfg, scenarios)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scenario execution failed: {str(e)}")

    # Convert to response
    results = []
    for _, row in results_df.iterrows():
        results.append(ScenarioResult(
            scenario_id=row["scenario_id"],
            label=row["label"],
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            strategy=row["strategy"],
            params=row["params"],
            cum_return=float(row["cum_return"]),
            sharpe=float(row["sharpe"]),
            max_drawdown=float(row["max_drawdown"]),
            trade_count=int(row["trade_count"]),
        ))

    return RunScenariosResponse(
        n_scenarios=len(results),
        results=results,
    )


@router.get("/{symbol}/{timeframe}")
async def get_scenario_results(symbol: str, timeframe: str):
    """
    Get previously saved scenario results (placeholder).

    In future versions, this will load cached scenario results.
    """
    raise HTTPException(
        status_code=501,
        detail="Scenario result caching not yet implemented. Use POST /run for now."
    )
