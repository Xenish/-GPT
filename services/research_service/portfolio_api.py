"""
Portfolio Construction API Endpoints.

Endpoints for portfolio optimization and analysis.
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

class PortfolioOptimizeRequest(BaseModel):
    """Request to optimize portfolio weights."""
    portfolio_id: str = Field(..., description="Portfolio identifier")
    strategy_ids: List[str] = Field(..., description="List of strategy IDs to include")
    weighting_method: str = Field("sharpe", description="Weighting method")

    # Constraints
    min_weight: Optional[float] = Field(0.05, description="Minimum weight per strategy")
    max_weight: Optional[float] = Field(0.50, description="Maximum weight per strategy")
    max_correlation: Optional[float] = Field(0.95, description="Maximum correlation threshold")

    # Rebalancing
    rebalance_frequency: Optional[str] = Field("monthly", description="Rebalance frequency")
    rebalance_threshold: Optional[float] = Field(0.05, description="Drift threshold for rebalancing")

    # Data source
    returns_data: Optional[Dict[str, List[float]]] = Field(None, description="Strategy returns data")
    file_paths: Optional[Dict[str, str]] = Field(None, description="Strategy file paths")


class PortfolioWeightResponse(BaseModel):
    """Portfolio weight for a strategy."""
    strategy_id: str
    weight: float
    sharpe_ratio: float
    annual_return: float
    volatility: float
    max_drawdown: float


class PortfolioOptimizeResponse(BaseModel):
    """Response from portfolio optimization."""
    portfolio_id: str
    weighting_method: str
    num_strategies: int
    weights: List[PortfolioWeightResponse]

    # Portfolio metrics
    total_return: float
    sharpe_ratio: float
    volatility: float
    max_drawdown: float
    metrics: Dict[str, float] | None = None
    sections: List[Dict[str, Any]] | None = None


class PortfolioBacktestRequest(BaseModel):
    """Request to backtest portfolio."""
    portfolio_id: str
    strategy_ids: List[str]
    weighting_method: str = "sharpe"

    # Constraints
    min_weight: Optional[float] = 0.05
    max_weight: Optional[float] = 0.50

    # Rebalancing
    rebalance_frequency: str = "monthly"
    rebalance_threshold: float = 0.05

    # Data
    returns_data: Optional[Dict[str, List[float]]] = None
    file_paths: Optional[Dict[str, str]] = None


class PortfolioBacktestResponse(BaseModel):
    """Response from portfolio backtest."""
    portfolio_id: str
    weighting_method: str
    num_strategies: int

    # Performance
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Risk
    volatility: float
    max_drawdown: float
    var_95: float
    cvar_95: float

    # Diversification
    diversification_ratio: float
    effective_n: float
    avg_correlation: float

    # Trading
    rebalance_count: int
    turnover: float
    metrics: Dict[str, float] | None = None
    sections: List[Dict[str, Any]] | None = None


class CorrelationAnalysisRequest(BaseModel):
    """Request for correlation analysis."""
    strategy_ids: List[str]
    returns_data: Optional[Dict[str, List[float]]] = None
    file_paths: Optional[Dict[str, str]] = None
    method: str = "pearson"


class CorrelationPair(BaseModel):
    """Correlation between two strategies."""
    strategy1: str
    strategy2: str
    correlation: float


class CorrelationAnalysisResponse(BaseModel):
    """Response from correlation analysis."""
    strategy_ids: List[str]
    correlation_matrix: Dict[str, Dict[str, float]]
    avg_correlation: float
    min_correlation: float
    max_correlation: float
    diversification_ratio: float
    effective_n: float
    highly_correlated_pairs: List[CorrelationPair]


class CompareMethodsRequest(BaseModel):
    """Request to compare weighting methods."""
    portfolio_id: str
    strategy_ids: List[str]
    returns_data: Optional[Dict[str, List[float]]] = None
    file_paths: Optional[Dict[str, str]] = None


class MethodComparisonRow(BaseModel):
    """Comparison row for a method."""
    method: str
    total_return: float
    sharpe_ratio: float
    volatility: float
    max_drawdown: float
    diversification_ratio: float
    rebalance_count: int


class CompareMethodsResponse(BaseModel):
    """Response from method comparison."""
    portfolio_id: str
    num_strategies: int
    best_sharpe_method: str
    best_return_method: str
    best_risk_adjusted_method: str
    results: List[MethodComparisonRow]


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/optimize", response_model=PortfolioOptimizeResponse)
async def optimize_portfolio(request: PortfolioOptimizeRequest):
    """
    Optimize portfolio weights using specified method.

    Supported methods:
    - equal: Equal weights
    - performance: Weight by returns
    - sharpe: Weight by Sharpe ratio
    - risk_parity: Inverse volatility
    - minimum_variance: Minimize portfolio variance
    - maximum_sharpe: Maximize Sharpe ratio
    - hierarchical_risk_parity: HRP algorithm
    """
    import pandas as pd
    from finantradealgo.research.ensemble.portfolio import (
        PortfolioConfig,
        PortfolioWeightingMethod,
        RebalanceFrequency,
    )
    from finantradealgo.research.ensemble.optimizer import PortfolioOptimizer
    from finantradealgo.research.ensemble.portfolio_backtest import (
        load_strategy_returns_from_files,
    )

    # Load strategy returns
    try:
        if request.returns_data:
            # Convert lists to pd.Series
            strategy_returns = {
                sid: pd.Series(returns)
                for sid, returns in request.returns_data.items()
            }
        elif request.file_paths:
            strategy_returns = load_strategy_returns_from_files(request.file_paths)
        else:
            raise HTTPException(status_code=400, detail="Must provide returns_data or file_paths")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load returns: {str(e)}")

    # Create portfolio config
    try:
        config = PortfolioConfig(
            portfolio_id=request.portfolio_id,
            strategy_ids=request.strategy_ids,
            weighting_method=PortfolioWeightingMethod(request.weighting_method),
            min_weight=request.min_weight or 0.05,
            max_weight=request.max_weight or 0.50,
            max_correlation=request.max_correlation or 0.95,
            rebalance_frequency=RebalanceFrequency(request.rebalance_frequency or "monthly"),
            rebalance_threshold=request.rebalance_threshold or 0.05,
        )
        config.validate()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid config: {str(e)}")

    # Optimize
    try:
        optimizer = PortfolioOptimizer(config)
        portfolio = optimizer.optimize_weights(strategy_returns)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

    # Format response
    weights = [
        PortfolioWeightResponse(
            strategy_id=w.strategy_id,
            weight=round(w.weight, 4),
            sharpe_ratio=round(w.sharpe_ratio, 3),
            annual_return=round(w.annual_return, 2),
            volatility=round(w.volatility, 2),
            max_drawdown=round(w.max_drawdown, 2),
        )
        for w in portfolio.weights
    ]

    return PortfolioOptimizeResponse(
        portfolio_id=portfolio.portfolio_id,
        weighting_method=request.weighting_method,
        num_strategies=len(request.strategy_ids),
        weights=weights,
        total_return=round(portfolio.total_return, 2),
        sharpe_ratio=round(portfolio.sharpe_ratio, 3),
        volatility=round(portfolio.volatility, 2),
        max_drawdown=round(portfolio.max_drawdown, 2),
        metrics={
            "sharpe_ratio": round(portfolio.sharpe_ratio, 3),
            "volatility": round(portfolio.volatility, 2),
            "max_drawdown": round(portfolio.max_drawdown, 2),
            "risk_parity_weight": getattr(portfolio, "risk_parity_weight", 0.0) or 0.0,
        },
        sections=[{"title": "Portfolio Overview"}],
    )


@router.post("/backtest", response_model=PortfolioBacktestResponse)
async def backtest_portfolio(request: PortfolioBacktestRequest):
    """
    Backtest portfolio with rebalancing.

    Returns comprehensive performance metrics including:
    - Returns, Sharpe, Sortino, Calmar ratios
    - Risk metrics (volatility, drawdown, VaR, CVaR)
    - Diversification metrics
    - Rebalancing statistics
    """
    import pandas as pd
    from finantradealgo.research.ensemble.portfolio import (
        PortfolioConfig,
        PortfolioWeightingMethod,
        RebalanceFrequency,
    )
    from finantradealgo.research.ensemble.portfolio_backtest import (
        PortfolioBacktester,
        load_strategy_returns_from_files,
    )

    # Load returns
    try:
        if request.returns_data:
            strategy_returns = {
                sid: pd.Series(returns)
                for sid, returns in request.returns_data.items()
            }
        elif request.file_paths:
            strategy_returns = load_strategy_returns_from_files(request.file_paths)
        else:
            raise HTTPException(status_code=400, detail="Must provide returns_data or file_paths")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load returns: {str(e)}")

    # Create config
    try:
        config = PortfolioConfig(
            portfolio_id=request.portfolio_id,
            strategy_ids=request.strategy_ids,
            weighting_method=PortfolioWeightingMethod(request.weighting_method),
            min_weight=request.min_weight or 0.05,
            max_weight=request.max_weight or 0.50,
            rebalance_frequency=RebalanceFrequency(request.rebalance_frequency),
            rebalance_threshold=request.rebalance_threshold,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid config: {str(e)}")

    # Backtest
    try:
        backtester = PortfolioBacktester(config)
        result = backtester.backtest_portfolio(strategy_returns)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")

    return PortfolioBacktestResponse(
        portfolio_id=result.portfolio_id,
        weighting_method=result.config.weighting_method.value,
        num_strategies=len(result.config.strategy_ids),
        total_return=round(result.total_return, 2),
        annual_return=round(result.annual_return, 2),
        sharpe_ratio=round(result.sharpe_ratio, 3),
        sortino_ratio=round(result.sortino_ratio, 3),
        calmar_ratio=round(result.calmar_ratio, 3),
        volatility=round(result.volatility, 2),
        max_drawdown=round(result.max_drawdown, 2),
        var_95=round(result.var_95, 2),
        cvar_95=round(result.cvar_95, 2),
        diversification_ratio=round(result.diversification_ratio, 3),
        effective_n=round(result.effective_n, 2),
        avg_correlation=round(result.avg_correlation, 3),
        rebalance_count=result.rebalance_count,
        turnover=round(result.turnover, 2),
        metrics={
            "sharpe_ratio": round(result.sharpe_ratio, 3),
            "volatility": round(result.volatility, 2),
            "max_drawdown": round(result.max_drawdown, 2),
        },
        sections=[{"title": "Portfolio Overview"}],
    )


@router.post("/correlation", response_model=CorrelationAnalysisResponse)
async def analyze_correlation(request: CorrelationAnalysisRequest):
    """
    Analyze correlations between strategies.

    Returns:
    - Full correlation matrix
    - Diversification metrics
    - Highly correlated pairs
    """
    import pandas as pd
    from finantradealgo.research.ensemble.correlation import CorrelationAnalyzer
    from finantradealgo.research.ensemble.portfolio_backtest import (
        load_strategy_returns_from_files,
    )

    # Load returns
    try:
        if request.returns_data:
            strategy_returns = {
                sid: pd.Series(returns)
                for sid, returns in request.returns_data.items()
            }
        elif request.file_paths:
            strategy_returns = load_strategy_returns_from_files(request.file_paths)
        else:
            raise HTTPException(status_code=400, detail="Must provide returns_data or file_paths")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load returns: {str(e)}")

    # Analyze
    try:
        analyzer = CorrelationAnalyzer()
        corr_matrix = analyzer.calculate_correlation_matrix(strategy_returns, request.method)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    # Get highly correlated pairs
    high_corr_pairs = corr_matrix.get_highly_correlated_pairs(threshold=0.7)

    pairs = [
        CorrelationPair(strategy1=s1, strategy2=s2, correlation=round(corr, 3))
        for s1, s2, corr in high_corr_pairs
    ]

    return CorrelationAnalysisResponse(
        strategy_ids=corr_matrix.strategy_ids,
        correlation_matrix=corr_matrix.correlation_matrix.to_dict(),
        avg_correlation=round(corr_matrix.avg_correlation, 3),
        min_correlation=round(corr_matrix.min_correlation, 3),
        max_correlation=round(corr_matrix.max_correlation, 3),
        diversification_ratio=round(corr_matrix.diversification_ratio, 3),
        effective_n=round(corr_matrix.effective_n, 2),
        highly_correlated_pairs=pairs,
    )


@router.post("/compare", response_model=CompareMethodsResponse)
async def compare_weighting_methods(request: CompareMethodsRequest):
    """
    Compare all portfolio weighting methods.

    Backtests each method and returns performance comparison table.
    """
    import pandas as pd
    from finantradealgo.research.ensemble.portfolio import (
        PortfolioConfig,
        RebalanceFrequency,
    )
    from finantradealgo.research.ensemble.portfolio_backtest import (
        PortfolioBacktester,
        load_strategy_returns_from_files,
    )

    # Load returns
    try:
        if request.returns_data:
            strategy_returns = {
                sid: pd.Series(returns)
                for sid, returns in request.returns_data.items()
            }
        elif request.file_paths:
            strategy_returns = load_strategy_returns_from_files(request.file_paths)
        else:
            raise HTTPException(status_code=400, detail="Must provide returns_data or file_paths")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load returns: {str(e)}")

    # Create config
    config = PortfolioConfig(
        portfolio_id=request.portfolio_id,
        strategy_ids=request.strategy_ids,
        rebalance_frequency=RebalanceFrequency.MONTHLY,
    )

    # Compare methods
    try:
        backtester = PortfolioBacktester(config)
        comparison = backtester.compare_weighting_methods(strategy_returns)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

    # Format results
    results = [
        MethodComparisonRow(
            method=row["method"],
            total_return=round(row["total_return"], 2),
            sharpe_ratio=round(row["sharpe_ratio"], 3),
            volatility=round(row["volatility"], 2),
            max_drawdown=round(row["max_drawdown"], 2),
            diversification_ratio=round(row["diversification_ratio"], 3),
            rebalance_count=int(row["rebalance_count"]),
        )
        for row in comparison.performance_table.to_dict(orient="records")
    ]

    return CompareMethodsResponse(
        portfolio_id=request.portfolio_id,
        num_strategies=len(request.strategy_ids),
        best_sharpe_method=comparison.best_sharpe_method,
        best_return_method=comparison.best_return_method,
        best_risk_adjusted_method=comparison.best_risk_adjusted_method,
        results=results,
    )


@router.get("/methods")
async def get_weighting_methods():
    """
    Get list of available portfolio weighting methods.

    Returns:
        Dictionary of weighting methods and their descriptions.
    """
    return {
        "equal": {
            "name": "Equal Weight",
            "description": "Equal allocation to all strategies",
        },
        "performance": {
            "name": "Performance-Based",
            "description": "Weight by historical returns",
        },
        "sharpe": {
            "name": "Sharpe-Based",
            "description": "Weight by Sharpe ratio",
        },
        "risk_parity": {
            "name": "Risk Parity",
            "description": "Inverse volatility weighting",
        },
        "minimum_variance": {
            "name": "Minimum Variance",
            "description": "Minimize portfolio variance (optimization-based)",
        },
        "maximum_sharpe": {
            "name": "Maximum Sharpe",
            "description": "Maximize Sharpe ratio (optimization-based)",
        },
        "hierarchical_risk_parity": {
            "name": "Hierarchical Risk Parity (HRP)",
            "description": "Cluster-based allocation using correlation hierarchy",
        },
    }
