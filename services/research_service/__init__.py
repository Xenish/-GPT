"""
Research Service - Strategy Search & Backtest API.

This service provides REST API endpoints for:
- Strategy search jobs (random/grid parameter optimization)
- Scenario-based backtesting
- Research job management

IMPORTANT: This service is isolated from live trading and NEVER
accesses real exchange APIs. It only uses backtest/mock execution.
"""
from __future__ import annotations

__version__ = "0.1.0"
