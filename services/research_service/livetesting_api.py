"""
Live Testing API Endpoints.

REST API for paper trading and production readiness validation.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field

from finantradealgo.research.livetesting import (
    LiveTestConfig,
    OrderSide,
    OrderType,
    SlippageModel,
)
from finantradealgo.research.livetesting.paper_trader import PaperTradingEngine
from finantradealgo.research.livetesting.readiness import ProductionReadinessValidator

router = APIRouter()


class LiveTestRequest(BaseModel):
    """Request to start live test."""
    strategy_id: str
    starting_capital: float = Field(10000.0, description="Starting capital")
    commission_pct: float = Field(0.001, description="Commission %")
    slippage_pct: float = Field(0.0005, description="Slippage %")


class OrderRequest(BaseModel):
    """Request to place order."""
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    order_type: str = Field("market", description="Order type")
    limit_price: Optional[float] = None


class ReadinessRequest(BaseModel):
    """Request for production readiness check."""
    min_test_duration_hours: float = Field(24.0)
    min_trades: int = Field(50)
    min_sharpe: float = Field(0.5)


# Store active paper trading sessions
_active_sessions: Dict[str, PaperTradingEngine] = {}


@router.post("/start")
async def start_live_test(request: LiveTestRequest):
    """Start live testing session."""
    try:
        config = LiveTestConfig(
            strategy_id=request.strategy_id,
            starting_capital=request.starting_capital,
            commission_pct=request.commission_pct,
            slippage_pct=request.slippage_pct,
        )

        engine = PaperTradingEngine(config)
        _active_sessions[request.strategy_id] = engine

        return {
            "success": True,
            "strategy_id": request.strategy_id,
            "starting_capital": request.starting_capital,
            "message": "Live test session started",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/order/{strategy_id}")
async def place_order(strategy_id: str, order: OrderRequest, market_price: float = Body(...)):
    """Place order in live test."""
    try:
        if strategy_id not in _active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        engine = _active_sessions[strategy_id]

        paper_order = engine.place_order(
            symbol=order.symbol,
            side=OrderSide(order.side),
            quantity=order.quantity,
            order_type=OrderType(order.order_type),
            limit_price=order.limit_price,
        )

        # Execute immediately at market price
        engine.execute_order(paper_order, market_price)

        return {
            "success": True,
            "order": paper_order.to_dict(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{strategy_id}")
async def get_status(strategy_id: str):
    """Get live test status."""
    try:
        if strategy_id not in _active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        engine = _active_sessions[strategy_id]

        return {
            "success": True,
            "strategy_id": strategy_id,
            "cash": engine.cash,
            "equity": engine.get_equity(),
            "positions": {k: {
                "quantity": v.quantity,
                "avg_price": v.avg_entry_price,
                "market_value": v.market_value,
                "unrealized_pnl": v.unrealized_pnl,
            } for k, v in engine.positions.items()},
            "total_trades": len(engine.trades),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate/{strategy_id}")
async def validate_readiness(strategy_id: str, request: ReadinessRequest):
    """Validate production readiness."""
    try:
        if strategy_id not in _active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        engine = _active_sessions[strategy_id]
        result = engine.calculate_result()

        validator = ProductionReadinessValidator(
            min_test_duration_hours=request.min_test_duration_hours,
            min_trades=request.min_trades,
            min_sharpe=request.min_sharpe,
        )

        readiness = validator.validate(result)

        return {
            "success": True,
            "result": result.to_summary_dict(),
            "readiness": readiness.to_dict(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/stop/{strategy_id}")
async def stop_live_test(strategy_id: str):
    """Stop live testing session."""
    try:
        if strategy_id not in _active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        engine = _active_sessions[strategy_id]
        result = engine.calculate_result()

        del _active_sessions[strategy_id]

        return {
            "success": True,
            "message": "Live test session stopped",
            "result": result.to_summary_dict(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
