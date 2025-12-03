"""
Strategies API.

Provides a registry-backed list of available strategies and their metadata.
"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel

from finantradealgo.strategies.strategy_engine import list_strategies

router = APIRouter()


class StrategyInfo(BaseModel):
    name: str
    family: str
    status: str
    is_searchable: bool
    uses_ml: bool
    uses_microstructure: bool
    uses_market_structure: bool
    description: str | None = None


@router.get("/", response_model=List[StrategyInfo])
async def list_strategy_meta():
    """List all registered strategies from the registry (single source of truth)."""
    strategies = list_strategies()
    response: List[Dict[str, Any]] = []
    for name, meta in strategies.items():
        response.append({
            "name": name,
            "family": meta.family,
            "status": meta.status,
            "is_searchable": bool(getattr(meta, "is_searchable", False)),
            "uses_ml": bool(getattr(meta, "uses_ml", False)),
            "uses_microstructure": bool(getattr(meta, "uses_microstructure", False)),
            "uses_market_structure": bool(getattr(meta, "uses_market_structure", False)),
            "description": getattr(meta, "description", None),
        })
    return response
