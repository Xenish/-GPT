"""
Research Service FastAPI Application.

This service provides REST endpoints for strategy research operations.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import time
from typing import AsyncGenerator, Any, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from finantradealgo.monitoring import (
    MonitoringConfig,
    get_metrics_collector,
    initialize_tracing,
    instrument_fastapi_app,
    trace_span,
)
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

# Monitoring setup
monitoring_config = MonitoringConfig(
    enabled=True,
    prometheus_enabled=True,
    otel_enabled=True,
    service_name="finantrade_research_service",
    environment="local",
)
metrics_collector = get_metrics_collector(monitoring_config)
tracer_provider = initialize_tracing(monitoring_config) if monitoring_config.otel_enabled else None


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

if tracer_provider:
    instrument_fastapi_app(app, tracer_provider)

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


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Capture request metrics for Prometheus."""
    start = time.perf_counter()
    response: Response | None = None
    try:
        response = await call_next(request)
        return response
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        status_code = response.status_code if response else 500
        endpoint = request.scope.get("path", request.url.path)
        metrics_collector.record_api_request(
            endpoint=endpoint,
            method=request.method,
            status_code=status_code,
            duration_ms=duration_ms,
        )


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


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


# === Monitoring helper wrappers (call from strategy/backtest/live trading code) ===
def record_strategy_signal_event(
    strategy_name: str,
    symbol: str,
    event_type: str,
    latency_ms: float,
    success: bool = True,
    **labels: Any,
) -> None:
    """Call from strategy execution path to emit strategy signal metrics."""
    metrics_collector.record_strategy_signal(
        strategy_name=strategy_name,
        symbol=symbol,
        event_type=event_type,
        latency_ms=latency_ms,
        success=success,
        **labels,
    )


def record_backtest_run_event(
    name: str,
    duration_seconds: float,
    memory_mb: float | None = None,
    success: bool = True,
    **labels: Any,
) -> None:
    """Call after backtest completion to emit duration and memory metrics."""
    metrics_collector.record_backtest_run(
        name=name,
        duration_seconds=duration_seconds,
        memory_mb=memory_mb,
        success=success,
        **labels,
    )


def record_live_trade_event(
    symbol: str,
    side: str,
    size: float,
    fill_rate: float,
    slippage_bps: float,
    **labels: Any,
) -> None:
    """Call from live trading execution to emit fills and slippage metrics."""
    metrics_collector.record_live_trade(
        symbol=symbol,
        side=side,
        size=size,
        fill_rate=fill_rate,
        slippage_bps=slippage_bps,
        **labels,
    )


@trace_span("backtest.run")
def run_backtest_with_trace(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """
    Example helper to execute a backtest inside a trace span.
    Wrap your backtest runner call with this to nest spans under API requests.
    """
    return fn(*args, **kwargs)
