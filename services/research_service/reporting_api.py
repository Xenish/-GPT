"""
Reporting API Endpoints.

Endpoints for generating research reports from backtest results.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse, HTMLResponse

from finantradealgo.research.reporting import BacktestReportGenerator, ReportFormat

router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================

class GenerateStrategySearchReportRequest(BaseModel):
    """Request to generate strategy search report."""
    job_id: str = Field(..., description="Job ID of the strategy search")
    top_n: int = Field(10, description="Number of top performers to highlight")
    format: str = Field("html", description="Report format: 'html', 'markdown', or 'json'")


class GenerateEnsembleReportRequest(BaseModel):
    """Request to generate ensemble backtest report."""
    ensemble_type: str = Field(..., description="Type of ensemble: 'weighted' or 'bandit'")
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe")
    component_names: Optional[List[str]] = Field(None, description="Component strategy names")
    format: str = Field("html", description="Report format: 'html', 'markdown', or 'json'")


class ReportInfo(BaseModel):
    """Information about a generated report."""
    report_id: str
    title: str
    created_at: str
    file_path: str
    format: str
    size_bytes: int


class GenerateReportResponse(BaseModel):
    """Response from report generation."""
    success: bool
    report_id: str
    file_path: str
    format: str
    message: str


class ListReportsResponse(BaseModel):
    """Response listing available reports."""
    reports: List[ReportInfo]
    total_count: int


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/strategy-search")
async def generate_strategy_search_report(request: GenerateStrategySearchReportRequest):
    """
    Generate report for strategy parameter search job.

    This endpoint:
    1. Loads strategy search results from job directory
    2. Generates comprehensive report with top performers, distributions, parameter analysis
    3. Saves report to file
    4. Returns report location

    Args:
        request: Report generation request

    Returns:
        Report generation response with file path

    Raises:
        HTTPException: If job not found or report generation fails
    """
    from finantradealgo.research.reporting import (
        StrategySearchReportGenerator,
        ReportFormat,
    )

    try:
        # Locate job directory
        job_dir = Path("outputs") / "strategy_search" / request.job_id

        if not job_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Job directory not found: {request.job_id}"
            )

        # Determine report format
        format_map = {
            "html": ReportFormat.HTML,
            "markdown": ReportFormat.MARKDOWN,
            "json": ReportFormat.JSON,
        }

        report_format = format_map.get(request.format.lower(), ReportFormat.HTML)

        generator = StrategySearchReportGenerator()
        report = generator.generate(
            job_dir=job_dir,
            job_id=request.job_id,
            top_n=request.top_n,
        )

        fmt = request.format.lower()
        report_dir = Path("reports") / "strategy_search"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{request.job_id}.{fmt if fmt != 'markdown' else 'md'}"

        if fmt == "markdown":
            content = report.to_markdown()
            report.save(report_path, ReportFormat.MARKDOWN)
            return {"job_id": request.job_id, "format": "markdown", "content": content}
        if fmt == "json":
            report.save(report_path, ReportFormat.JSON)
            return {"job_id": request.job_id, "format": "json", "content": report.to_dict()}

        content = report.to_html()
        report.save(report_path, ReportFormat.HTML)
        return {"job_id": request.job_id, "format": "html", "content": content}

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed: {str(e)}"
        )


@router.post("/ensemble", response_model=GenerateReportResponse)
async def generate_ensemble_report(request: GenerateEnsembleReportRequest):
    """
    Generate report for ensemble backtest.

    This endpoint:
    1. Uses ensemble backtest result data (must be provided externally or from session)
    2. Generates comprehensive ensemble report
    3. Saves report to file
    4. Returns report location

    NOTE: This endpoint currently expects ensemble backtest data to be available
    in memory or from a previous API call. For full functionality, integrate with
    ensemble backtest storage/session management.

    Args:
        request: Report generation request

    Returns:
        Report generation response with file path

    Raises:
        HTTPException: If data not found or report generation fails
    """
    raise HTTPException(
        status_code=501,
        detail=(
            "Ensemble report generation requires integration with ensemble backtest storage. "
            "Currently, ensemble reports should be generated directly via Python SDK after backtest. "
            "See playbook: ensemble_development.md"
        )
    )


@router.get("/backtests/{job_id}/report")
async def generate_backtest_report(job_id: str, format: str = "html"):
    """
    Generate and return a backtest report for a job.

    Uses metrics.json, equity_curve.csv, and trades.csv under the job directory.
    """
    base_dir = Path(os.environ.get("BACKTEST_REPORT_BASE_DIR", Path("outputs") / "backtests"))
    job_dir = base_dir / job_id

    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found under {base_dir}")

    metrics_path = job_dir / "metrics.json"
    equity_path = job_dir / "equity_curve.csv"
    trades_path = job_dir / "trades.csv"

    for required in [metrics_path, equity_path, trades_path]:
        if not required.exists():
            raise HTTPException(status_code=404, detail=f"Missing required file: {required}")

    gen = BacktestReportGenerator()
    try:
        report = gen.generate(
            metrics=metrics_path,
            equity_curve_path=equity_path,
            trades_path=trades_path,
            job_id=job_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc))

    fmt = format.lower()
    if fmt in ("markdown", "md"):
        content = report.to_markdown()
        report.save(job_dir / "report.md", ReportFormat.MARKDOWN)
        return {"job_id": job_id, "format": "markdown", "content": content}
    if fmt == "json":
        return {"job_id": job_id, "format": "json", "content": report.to_dict()}

    # Default to HTML
    content = report.to_html()
    report.save(job_dir / "report.html", ReportFormat.HTML)
    return {"job_id": job_id, "format": "html", "content": content}


@router.get("/list", response_model=ListReportsResponse)
async def list_reports(
    report_type: Optional[str] = None,
    limit: int = 50,
):
    """
    List available generated reports.

    Args:
        report_type: Filter by report type ('strategy_search', 'ensemble', etc.)
        limit: Maximum number of reports to return

    Returns:
        List of available reports with metadata
    """
    import os
    from datetime import datetime

    try:
        reports_dir = Path("reports")

        if not reports_dir.exists():
            return ListReportsResponse(reports=[], total_count=0)

        # Collect all report files
        report_files = []

        if report_type:
            # Filter by type
            search_dir = reports_dir / report_type
            if search_dir.exists():
                report_files.extend(search_dir.glob("*.*"))
        else:
            # All reports
            for subdir in reports_dir.iterdir():
                if subdir.is_dir():
                    report_files.extend(subdir.glob("*.*"))

        # Sort by modification time (newest first)
        report_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Limit results
        report_files = report_files[:limit]

        # Build report info
        reports = []
        for report_path in report_files:
            stat = report_path.stat()

            # Determine format from extension
            suffix = report_path.suffix.lstrip(".")
            if suffix not in ["html", "md", "json"]:
                continue  # Skip non-report files

            # Extract report ID from filename
            report_id = report_path.stem

            # Try to extract title from file (simple heuristic)
            title = report_id.replace("_", " ").title()

            reports.append(ReportInfo(
                report_id=report_id,
                title=title,
                created_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                file_path=str(report_path),
                format=suffix,
                size_bytes=stat.st_size,
            ))

        return ListReportsResponse(
            reports=reports,
            total_count=len(reports),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list reports: {str(e)}"
        )


@router.get("/view/{report_type}/{report_id}")
async def view_report(
    report_type: str,
    report_id: str,
    format: str = "html",
):
    """
    View/download a generated report.

    Args:
        report_type: Type of report ('strategy_search', 'ensemble', etc.)
        report_id: Report ID
        format: Report format ('html', 'markdown', 'json')

    Returns:
        Report file (HTML rendered in browser, or file download)
    """
    try:
        # Construct file path
        report_dir = Path("reports") / report_type
        report_filename = f"{report_id}.{format}"
        report_path = report_dir / report_filename

        if not report_path.exists():
            # Try alternate extensions
            for ext in ["html", "md", "json"]:
                alternate_path = report_dir / f"{report_id}.{ext}"
                if alternate_path.exists():
                    report_path = alternate_path
                    break
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Report not found: {report_type}/{report_id}"
                )

        # Return file based on format
        if report_path.suffix == ".html":
            # Render HTML directly in browser
            with report_path.open("r", encoding="utf-8") as f:
                html_content = f.read()
            return HTMLResponse(content=html_content)
        else:
            # Download file
            return FileResponse(
                path=str(report_path),
                filename=report_path.name,
                media_type="application/octet-stream",
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to view report: {str(e)}"
        )


@router.delete("/{report_type}/{report_id}")
async def delete_report(
    report_type: str,
    report_id: str,
):
    """
    Delete a generated report.

    Args:
        report_type: Type of report ('strategy_search', 'ensemble', etc.)
        report_id: Report ID

    Returns:
        Success message
    """
    try:
        report_dir = Path("reports") / report_type

        if not report_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Report type directory not found: {report_type}"
            )

        # Find all files matching report_id
        deleted_files = []
        for file_path in report_dir.glob(f"{report_id}.*"):
            file_path.unlink()
            deleted_files.append(str(file_path))

        if not deleted_files:
            raise HTTPException(
                status_code=404,
                detail=f"Report not found: {report_type}/{report_id}"
            )

        return {
            "success": True,
            "message": f"Deleted {len(deleted_files)} file(s)",
            "deleted_files": deleted_files,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete report: {str(e)}"
        )
