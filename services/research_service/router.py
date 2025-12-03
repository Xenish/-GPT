"""
Research Service API Router.

Defines all research-related API endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter

from services.research_service.jobs_api import router as jobs_router
from services.research_service.health import router as health_router
from services.research_service.scenarios_api import router as scenarios_router
from services.research_service.strategies_api import router as strategies_router

# Main research router
research_router = APIRouter()

# Include sub-routers
research_router.include_router(jobs_router, prefix="/strategy-search/jobs", tags=["strategy-search"])
research_router.include_router(scenarios_router, prefix="/scenarios", tags=["scenarios"])
research_router.include_router(strategies_router, prefix="/strategies", tags=["strategies"])
research_router.include_router(health_router, tags=["health"])
