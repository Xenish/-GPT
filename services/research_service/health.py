"""
Health check endpoints for research service.
"""
from __future__ import annotations

from fastapi import APIRouter

from services.research_service.concurrency import get_job_limiter

router = APIRouter()


@router.get("/ping")
async def ping():
    """Simple ping endpoint."""
    return {"status": "pong"}


@router.get("/status")
async def status():
    """Detailed status endpoint with job limiter info."""
    try:
        limiter = get_job_limiter()
        job_info = {
            "active_jobs": limiter.active_jobs,
            "max_jobs": limiter.max_jobs,
            "available_slots": limiter.available_slots,
        }
    except RuntimeError:
        job_info = {"error": "Job limiter not initialized"}

    return {
        "service": "research",
        "status": "operational",
        "features": {
            "strategy_search": "enabled",
            "scenarios": "enabled",
            "backtests": "enabled",
        },
        "job_limiter": job_info,
    }
