"""
Strategy Search Jobs API.

Endpoints for creating and managing strategy search jobs.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import finantradealgo.research.strategy_search.search_engine as se
from finantradealgo.research.strategy_search.analysis import load_results, top_n_by_metric
from finantradealgo.research.reporting.base import ReportFormat
from finantradealgo.research.strategy_search.jobs import (
    StrategySearchJob,
    StrategySearchJobConfig,
    create_job_id,
)
from finantradealgo.research.reporting.strategy_search import StrategySearchReportGenerator
from finantradealgo.strategies.strategy_engine import get_strategy_meta
from finantradealgo.system.config_loader import load_config
from services.research_service.concurrency import get_job_limiter

router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================


class StrategySearchJobRequest(BaseModel):
    """Request model for creating a strategy search job."""
    strategy: str = Field(..., description="Strategy to optimize (e.g., rule)")
    symbol: str = Field(..., description="Trading symbol (e.g., AIAUSDT)")
    timeframe: str = Field(..., description="Timeframe (e.g., 15m, 1h)")
    search_type: str = Field("random", description="Search mode: 'random' or 'grid'")
    n_samples: int = Field(50, description="Number of samples for search", ge=1, le=1000)
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    notes: Optional[str] = Field(None, description="Optional notes about the job")


class StrategySearchJobResponse(BaseModel):
    """Response model for job creation."""
    job_id: str
    status: str
    message: str
    results_path: Optional[str] = None
    output_dir: Optional[str] = None


class JobStatusResponse(BaseModel):
    """Response model for job status/summary."""
    job_id: str
    status: str
    strategy: Optional[str] = None
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    n_samples: Optional[int] = None
    created_at: Optional[str] = None
    n_results: int = 0
    best_sharpe: Optional[float] = None
    best_cum_return: Optional[float] = None
    top_candidate_params: Optional[Dict[str, Any]] = None
    results_available: bool = False


# ============================================================================
# Helpers
# ============================================================================


def _persist_results(job: StrategySearchJob, results: List[Dict[str, Any]], sys_cfg: Dict[str, Any]) -> Path:
    """Persist results and metadata similar to run_random_search."""
    job_dir = se.BASE_OUTPUT_DIR / job.job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    se._validate_results(df)

    results_parquet = job_dir / "results.parquet"
    results_csv = job_dir / "results.csv"
    df.to_parquet(results_parquet, index=False)
    df.to_csv(results_csv, index=False)

    git_sha = se._get_git_sha()
    n_success = (df["status"] == "ok").sum() if "status" in df.columns else len(df)
    n_errors = (df["status"] == "error").sum() if "status" in df.columns else 0

    data_cfg = sys_cfg.get("data_cfg") if sys_cfg else None
    data_snapshot = {}
    if data_cfg:
        data_snapshot = {
            "ohlcv_dir": getattr(data_cfg, "ohlcv_dir", "unknown"),
            "lookback_days": getattr(data_cfg, "lookback_days", {}),
        }

    meta_dict = job.to_dict()
    meta_dict["git_sha"] = git_sha
    meta_dict["results_path_parquet"] = str(results_parquet.relative_to(job_dir))
    meta_dict["results_path_csv"] = str(results_csv.relative_to(job_dir))
    meta_dict["n_results"] = len(df)
    meta_dict["n_success"] = int(n_success)
    meta_dict["n_errors"] = int(n_errors)
    meta_dict["data_snapshot"] = data_snapshot
    meta_dict["profile"] = getattr(job, "profile", "research")

    meta_path = job_dir / "meta.json"
    meta_path.write_text(json.dumps(meta_dict, indent=2), encoding="utf-8")

    return job_dir


def _compute_summary(job_dir: Path) -> JobStatusResponse:
    """Build a summary response from persisted results."""
    try:
        df, meta = load_results(job_dir, include_meta=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    results_available = not df.empty
    best_sharpe = float(df["sharpe"].max()) if "sharpe" in df.columns and not df["sharpe"].isna().all() else None
    best_cum_return = float(df["cum_return"].max()) if "cum_return" in df.columns and not df["cum_return"].isna().all() else None

    top_candidate_params = None
    if "sharpe" in df.columns and not df.empty:
        top_df = top_n_by_metric(df, metric="sharpe", n=1, ascending=False)
        if not top_df.empty:
            top_candidate_params = top_df.iloc[0].get("params")

    return JobStatusResponse(
        job_id=str(job_dir.name),
        status="completed" if results_available else "unknown",
        strategy=meta.get("strategy"),
        symbol=meta.get("symbol"),
        timeframe=meta.get("timeframe"),
        n_samples=meta.get("n_samples"),
        created_at=meta.get("created_at"),
        n_results=meta.get("n_results", len(df)),
        best_sharpe=best_sharpe,
        best_cum_return=best_cum_return,
        top_candidate_params=top_candidate_params,
        results_available=results_available,
    )


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
    # Load system config
    try:
        sys_cfg = load_config("research")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Config load failed: {str(e)}")
    cfg_profile = sys_cfg.get("profile", sys_cfg.get("mode", "unknown"))
    if cfg_profile != "research":
        raise HTTPException(
            status_code=400,
            detail="Strategy search is allowed only with the 'research' profile.",
        )

    # Get job limiter (initialize lazily if needed for tests)
    try:
        limiter = get_job_limiter()
    except RuntimeError:
        from services.research_service.concurrency import initialize_job_limiter

        initialize_job_limiter(sys_cfg["research_cfg"])
        limiter = get_job_limiter()

    # Check if slots available
    if limiter.available_slots <= 0:
        raise HTTPException(
            status_code=429,
            detail=f"Job limit reached. {limiter.active_jobs}/{limiter.max_jobs} jobs running."
        )

    # Optional output dir override for testing
    output_override = os.environ.get("STRATEGY_SEARCH_OUTPUT_DIR")
    if output_override:
        se.BASE_OUTPUT_DIR = Path(output_override)

    # Validate strategy meta and param space
    try:
        meta = get_strategy_meta(request.strategy)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not meta.is_searchable or meta.param_space is None:
        raise HTTPException(
            status_code=400,
            detail=f"Strategy '{request.strategy}' is not searchable (no ParamSpace defined)."
        )

    # Build job via config
    job_cfg = StrategySearchJobConfig(
        job_name=f"{request.strategy}_{request.symbol}_{request.timeframe}",
        strategy_name=request.strategy,
        symbol=request.symbol,
        timeframe=request.timeframe,
        search_type=request.search_type,
        n_samples=request.n_samples,
        random_seed=request.seed,
        notes=request.notes,
    )
    job = job_cfg.to_job(profile="research")

    # Run job (with concurrency control)
    async with limiter.acquire():
        try:
            dry_run = os.environ.get("STRATEGY_SEARCH_DRYRUN") == "1"
            if dry_run:
                dummy_row = {col: None for col in se.REQUIRED_RESULT_COLUMNS}
                dummy_row["params"] = {"_dummy": 0}
                dummy_row["status"] = "ok"
                dummy_row["error_message"] = None
                dummy_row["cum_return"] = 0.0
                dummy_row["sharpe"] = 0.0
                dummy_row["max_drawdown"] = 0.0
                dummy_row["win_rate"] = 0.0
                dummy_row["trade_count"] = 0
                job_dir = _persist_results(job, [dummy_row], sys_cfg)
            elif request.search_type == "random":
                job_dir = se.run_random_search(
                    job=job,
                    param_space=meta.param_space,
                    sys_cfg=sys_cfg,
                )
            elif request.search_type == "grid":
                grid_results = se.grid_search(
                    strategy_name=job.strategy,
                    sys_cfg=sys_cfg,
                    param_space=meta.param_space,
                    grid_points=None,
                )
                job_dir = _persist_results(job, grid_results, sys_cfg)
            else:  # pragma: no cover - defensive
                raise HTTPException(status_code=400, detail=f"Unsupported search_type: {request.search_type}")
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
        output_dir=str(job_dir),
    )


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get status and results summary of a strategy search job.
    """
    job_dir = se.BASE_OUTPUT_DIR / job_id

    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    return _compute_summary(job_dir)


@router.get("/{job_id}/report")
async def get_job_report(job_id: str, format: str = "html"):
    """
    Generate and return a strategy search report for a job.
    """
    output_override = os.environ.get("STRATEGY_SEARCH_OUTPUT_DIR")
    if output_override:
        se.BASE_OUTPUT_DIR = Path(output_override)

    job_dir = se.BASE_OUTPUT_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    gen = StrategySearchReportGenerator()
    try:
        report = gen.generate(job_dir=job_dir, job_id=job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if format == "markdown":
        content = report.to_markdown()
        report.save(job_dir / "report.md", ReportFormat.MARKDOWN)
        return {"job_id": job_id, "format": "markdown", "content": content}
    if format == "json":
        return {"job_id": job_id, "format": "json", "content": report.to_dict()}

    # Default to HTML
    content = report.to_html()
    report.save(job_dir / "report.html", ReportFormat.HTML)
    return {"job_id": job_id, "format": "html", "content": content}


@router.get("/", response_model=List[JobStatusResponse])
async def list_jobs(limit: int = 10, status: Optional[str] = None):
    """
    List recent strategy search jobs.

    Args:
        limit: Maximum number of jobs to return
        status: Filter by status (pending, running, completed, failed)
    """
    base_dir = se.BASE_OUTPUT_DIR
    if not base_dir.exists():
        return []

    job_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    job_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)  # Most recent first

    jobs = []
    for job_dir in job_dirs[:limit]:
        try:
            summary = _compute_summary(job_dir)
            jobs.append(summary)
        except HTTPException:
            continue

    return jobs
