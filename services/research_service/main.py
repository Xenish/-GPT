"""
Research Service FastAPI Application.

This service provides REST endpoints for strategy research operations.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from finantradealgo.system.config_loader import load_system_config
from finantradealgo.system.config_validation import validate_research_config
from services.research_service.router import research_router
from services.research_service.concurrency import initialize_job_limiter
from services.research_service.ensemble_api import router as ensemble_router
from services.research_service.reporting_api import router as reporting_router
from services.research_service.performance_api import router as performance_router
from services.research_service.visualization_api import router as visualization_router
from services.research_service.walkforward_api import router as walkforward_router
from services.research_service.montecarlo_api import router as montecarlo_router
from services.research_service.livetesting_api import router as livetesting_router
from services.research_service.portfolio_api import router as portfolio_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Runs validation checks on startup and cleanup on shutdown.
    """
    # Startup: Validate research config
    print("=" * 60)
    print("Research Service Starting...")
    print("=" * 60)

    try:
        cfg = load_system_config()
        validate_research_config(cfg)

        # Initialize job limiter
        initialize_job_limiter(cfg['research_cfg'])

        print(f"[OK] Config validation passed")
        print(f"  Mode: {cfg.get('mode')}")
        print(f"  Exchange Type: {cfg.get('exchange', {}).get('type')}")
        print(f"  Strategy Universe: {cfg['research_cfg'].strategy_universe}")
        print(f"  Max Parallel Jobs: {cfg['research_cfg'].max_parallel_jobs}")
        print(f"[OK] Job limiter initialized")
    except Exception as e:
        print(f"[FAIL] Startup failed: {e}")
        raise

    print("=" * 60)
    print("Research Service Ready")
    print("=" * 60)

    yield

    # Shutdown
    print("Research Service Shutting Down...")


# Create FastAPI app
app = FastAPI(
    title="FinanTradeAlgo Research Service",
    description=(
        "Research API for strategy search, backtesting, and scenario analysis. "
        "This service is isolated from live trading and only operates on historical data."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(research_router, prefix="/api/research", tags=["research"])
app.include_router(ensemble_router, prefix="/api/research/ensemble", tags=["ensemble"])
app.include_router(reporting_router, prefix="/api/research/reports", tags=["reports"])
app.include_router(performance_router, prefix="/api/research/performance", tags=["performance"])
app.include_router(visualization_router, prefix="/api/research/visualizations", tags=["visualizations"])
app.include_router(walkforward_router, prefix="/api/research/walkforward", tags=["walkforward"])
app.include_router(montecarlo_router, prefix="/api/research/montecarlo", tags=["montecarlo"])
app.include_router(livetesting_router, prefix="/api/research/livetesting", tags=["livetesting"])
app.include_router(portfolio_router, prefix="/api/research/portfolio", tags=["portfolio"])


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "FinanTradeAlgo Research Service",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
