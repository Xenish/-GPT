"""
Ensemble Strategy API Endpoints.

Endpoints for running and analyzing ensemble strategies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================

class ComponentStrategyRequest(BaseModel):
    """Component strategy configuration."""
    strategy_name: str = Field(..., description="Strategy name (e.g., 'rule', 'trend_continuation')")
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Strategy parameters")
    weight: float = Field(1.0, description="Component weight")
    label: Optional[str] = Field(None, description="Component label")


class RunEnsembleRequest(BaseModel):
    """Request to run ensemble backtest."""
    ensemble_type: str = Field(..., description="Ensemble type: 'weighted' or 'bandit'")
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe (e.g., '15m', '1h')")
    components: List[ComponentStrategyRequest] = Field(..., description="Component strategies")

    # Weighted ensemble params
    weighting_method: Optional[str] = Field("equal", description="Weighting method for weighted ensembles")
    reweight_period: Optional[int] = Field(0, description="Reweight period (0 = static)")

    # Bandit ensemble params
    bandit_algorithm: Optional[str] = Field("epsilon_greedy", description="Bandit algorithm")
    epsilon: Optional[float] = Field(0.1, description="Epsilon for epsilon-greedy")
    ucb_c: Optional[float] = Field(2.0, description="C parameter for UCB1")
    update_period: Optional[int] = Field(20, description="Update period for bandit")

    # Common params
    warmup_bars: Optional[int] = Field(100, description="Warmup bars before trading")


class ComponentMetric(BaseModel):
    """Metrics for a component strategy."""
    component: str
    cum_return: float
    sharpe: float
    max_dd: float
    trade_count: int
    win_rate: float


class EnsembleMetrics(BaseModel):
    """Ensemble performance metrics."""
    cum_return: float
    sharpe: float
    max_dd: float
    trade_count: int
    win_rate: float


class RunEnsembleResponse(BaseModel):
    """Response from ensemble backtest."""
    ensemble_type: str
    symbol: str
    timeframe: str
    n_components: int
    ensemble_metrics: EnsembleMetrics
    component_metrics: List[ComponentMetric]
    bandit_stats: Optional[List[Dict[str, Any]]] = None
    weight_history: Optional[List[Dict[str, Any]]] = None


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/run", response_model=RunEnsembleResponse)
async def run_ensemble_backtest(request: RunEnsembleRequest):
    """
    Run ensemble strategy backtest.

    This endpoint:
    1. Loads OHLCV data for symbol/timeframe
    2. Creates component strategies
    3. Runs ensemble backtest
    4. Returns performance metrics

    NOTE: Currently synchronous (blocks until complete).
    """
    from datetime import datetime
    from finantradealgo.data.ohlcv_loader import load_ohlcv
    from finantradealgo.research.ensemble.backtest import run_ensemble_backtest
    from finantradealgo.research.ensemble.weighted import (
        WeightedEnsembleStrategy,
        WeightedEnsembleConfig,
        WeightingMethod,
    )
    from finantradealgo.research.ensemble.bandit import (
        BanditEnsembleStrategy,
        BanditEnsembleConfig,
        BanditAlgorithm,
    )
    from finantradealgo.research.ensemble.base import ComponentStrategy
    from finantradealgo.system.config_loader import load_config

    # Load system config
    try:
        sys_cfg = load_config("research")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Config load failed: {str(e)}")

    # Load OHLCV data
    try:
        df = load_ohlcv(symbol=request.symbol, timeframe=request.timeframe, sys_cfg=sys_cfg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data load failed: {str(e)}")

    # Prepare component configs
    components = [
        {
            "strategy_name": comp.strategy_name,
            "params": comp.params,
            "weight": comp.weight,
            "label": comp.label or comp.strategy_name,
        }
        for comp in request.components
    ]

    component_objs = [
        ComponentStrategy(
            strategy_name=comp["strategy_name"],
            strategy_params=comp["params"],
            weight=comp["weight"],
            label=comp["label"],
        )
        for comp in components
    ]

    # Create ensemble strategy
    try:
        if request.ensemble_type == "weighted":
            config = WeightedEnsembleConfig(
                components=component_objs,
                warmup_bars=request.warmup_bars or 100,
                weighting_method=WeightingMethod(request.weighting_method or "equal"),
                reweight_period=request.reweight_period or 0,
            )
            ensemble = WeightedEnsembleStrategy(config)

        elif request.ensemble_type == "bandit":
            config = BanditEnsembleConfig(
                components=component_objs,
                warmup_bars=request.warmup_bars or 100,
                bandit_algorithm=BanditAlgorithm(request.bandit_algorithm or "epsilon_greedy"),
                epsilon=request.epsilon or 0.1,
                ucb_c=request.ucb_c or 2.0,
                update_period=request.update_period or 20,
            )
            ensemble = BanditEnsembleStrategy(config)

        else:
            raise HTTPException(status_code=400, detail=f"Invalid ensemble_type: {request.ensemble_type}")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ensemble creation failed: {str(e)}")

    # Run backtest
    try:
        result = run_ensemble_backtest(
            ensemble_strategy=ensemble,
            df=df,
            components=components,
            sys_cfg=sys_cfg,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")

    # Format response
    ensemble_metrics = EnsembleMetrics(**result.ensemble_metrics)

    component_metrics_list = [
        ComponentMetric(**row)
        for row in result.component_metrics.to_dict(orient="records")
    ]

    # Bandit stats
    bandit_stats = None
    if result.bandit_stats is not None:
        bandit_stats = result.bandit_stats.to_dict(orient="records")

    # Weight history
    weight_history = None
    if result.weight_history is not None:
        weight_history = result.weight_history.to_dict(orient="records")

    return RunEnsembleResponse(
        ensemble_type=request.ensemble_type,
        symbol=request.symbol,
        timeframe=request.timeframe,
        n_components=len(components),
        ensemble_metrics=ensemble_metrics,
        component_metrics=component_metrics_list,
        bandit_stats=bandit_stats,
        weight_history=weight_history,
    )


@router.get("/algorithms")
async def get_ensemble_algorithms():
    """
    Get list of available ensemble algorithms.

    Returns:
        Dictionary of ensemble types and their algorithms.
    """
    return {
        "weighted": {
            "description": "Weighted ensemble that combines all component signals",
            "weighting_methods": [
                {"name": "equal", "description": "Equal weight for all components"},
                {"name": "sharpe", "description": "Weight by historical Sharpe ratio"},
                {"name": "inverse_vol", "description": "Weight by inverse volatility"},
                {"name": "return", "description": "Weight by historical return"},
                {"name": "custom", "description": "Use custom weights from config"},
            ],
        },
        "bandit": {
            "description": "Multi-armed bandit that selects one component at a time",
            "algorithms": [
                {"name": "epsilon_greedy", "description": "Explore with probability epsilon"},
                {"name": "ucb1", "description": "Upper Confidence Bound"},
                {"name": "thompson_sampling", "description": "Bayesian Thompson Sampling"},
            ],
        },
    }
