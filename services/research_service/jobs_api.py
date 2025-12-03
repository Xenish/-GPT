"""
Strategy Search Jobs API.

Endpoints for creating and managing strategy search jobs.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================

class StrategySearchJobRequest(BaseModel):
    """Request model for creating a strategy search job."""
    job_name: str = Field(..., description="Human-readable job name")
    strategy_name: str = Field(..., description="Strategy to optimize")
    symbol: str = Field(..., description="Trading symbol (e.g., AIAUSDT)")
    timeframe: str = Field(..., description="Timeframe (e.g., 15m, 1h)")
    mode: str = Field("random", description="Search mode: 'random' or 'grid'")
    n_samples: int = Field(50, description="Number of samples for random search", ge=1, le=1000)
    fixed_params: Optional[Dict[str, Any]] = Field(None, description="Fixed parameters (not searched)")
    search_space_override: Optional[Dict[str, Any]] = Field(None, description="Override param space")
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class StrategySearchJobResponse(BaseModel):
    """Response model for job creation."""
    job_id: str
    status: str
    message: str
    results_path: Optional[str] = None


class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    status: str
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    n_results: int = 0
    results_available: bool = False


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/", response_model=StrategySearchJobResponse)
async def create_strategy_search_job(request: StrategySearchJobRequest):
    """
    Create and execute a strategy search job.

    NOTE: This is a synchronous implementation (blocks until complete).
    Future versions will support async/background job execution.
    """
    from datetime import datetime
    from finantradealgo.research.strategy_search.jobs import StrategySearchJob, create_job_id
    from finantradealgo.research.strategy_search.search_engine import run_random_search
    from finantradealgo.system.config_loader import load_config
    from services.research_service.concurrency import get_job_limiter

    # Get job limiter
    try:
        limiter = get_job_limiter()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Check if slots available
    if limiter.available_slots <= 0:
        raise HTTPException(
            status_code=429,
            detail=f"Job limit reached. {limiter.active_jobs}/{limiter.max_jobs} jobs running."
        )

    # Load system config
    try:
        sys_cfg = load_config("research")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Config load failed: {str(e)}")

    # Create job
    job_id = create_job_id(
        strategy=request.strategy_name,
        symbol=request.symbol,
        timeframe=request.timeframe,
    )

    job = StrategySearchJob(
        job_id=job_id,
        strategy=request.strategy_name,
        symbol=request.symbol,
        timeframe=request.timeframe,
        search_type=request.mode,
        n_samples=request.n_samples,
        profile="research",
        created_at=datetime.utcnow(),
        seed=request.random_seed,
        mode="research",
    )

    # Run job (with concurrency control)
    async with limiter.acquire():
        try:
            job_dir = run_random_search(
                job=job,
                sys_cfg=sys_cfg,
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Job execution failed: {str(e)}"
            )

    return StrategySearchJobResponse(
        job_id=job.job_id,
        status="completed",
        message="Job completed successfully",
        results_path=str(job_dir / "results.csv"),
    )


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get status and results of a strategy search job.
    """
    from pathlib import Path
    import json

    # Look for job in outputs
    job_dir = Path("outputs/strategy_search") / job_id

    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    # Load meta.json
    meta_path = job_dir / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Job metadata not found")

    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load metadata: {str(e)}")

    # Check if results exist
    results_parquet = job_dir / "results.parquet"
    results_csv = job_dir / "results.csv"
    results_available = results_parquet.exists() or results_csv.exists()

    return JobStatusResponse(
        job_id=job_id,
        status="completed" if results_available else "unknown",
        created_at=meta.get("created_at"),
        completed_at=None,  # Not tracked yet
        n_results=meta.get("n_results", 0),
        results_available=results_available,
    )


@router.get("/", response_model=List[JobStatusResponse])
async def list_jobs(limit: int = 10, status: Optional[str] = None):
    """
    List recent strategy search jobs.

    Args:
        limit: Maximum number of jobs to return
        status: Filter by status (pending, running, completed, failed)
    """
    from pathlib import Path
    import json

    base_dir = Path("outputs/strategy_search")
    if not base_dir.exists():
        return []

    # Get all job directories
    job_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    job_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)  # Most recent first

    jobs = []
    for job_dir in job_dirs[:limit]:
        meta_path = job_dir / "meta.json"
        if not meta_path.exists():
            continue

        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)

            results_available = (job_dir / "results.parquet").exists()

            jobs.append(JobStatusResponse(
                job_id=job_dir.name,
                status="completed" if results_available else "unknown",
                created_at=meta.get("created_at"),
                completed_at=None,
                n_results=meta.get("n_results", 0),
                results_available=results_available,
            ))
        except Exception:
            continue

    return jobs
