"""Interfaces and configuration helpers for monitoring."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class MetricName(str, Enum):
    """Canonical metric identifiers for FinanTradeAlgo instrumentation."""

    STRATEGY_EXECUTION = "strategy_execution"
    BACKTEST_PERFORMANCE = "backtest_performance"
    API_REQUESTS = "api_requests"
    LIVE_TRADING = "live_trading"


@dataclass(frozen=True)
class MonitoringConfig:
    """Flags and metadata used when configuring monitoring backends."""

    enabled: bool = True
    prometheus_enabled: bool = False
    otel_enabled: bool = False
    otlp_endpoint: str | None = None
    otlp_insecure: bool = True
    service_name: str | None = None
    environment: str | None = None
    labels: Mapping[str, Any] | None = None


class MetricsCollector(ABC):
    """Base interface for emitting monitoring events."""

    def __init__(self, config: MonitoringConfig | None = None) -> None:
        self.config = config or MonitoringConfig()

    @abstractmethod
    def record_strategy_signal(
        self,
        strategy_name: str,
        symbol: str,
        event_type: str,
        latency_ms: float,
        success: bool = True,
        **labels: Any,
    ) -> None:
        """Record a strategy execution signal."""

    @abstractmethod
    def record_backtest_run(
        self,
        name: str,
        duration_seconds: float,
        memory_mb: float | None = None,
        success: bool = True,
        **labels: Any,
    ) -> None:
        """Record a backtest execution summary."""

    @abstractmethod
    def record_api_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration_ms: float,
        **labels: Any,
    ) -> None:
        """Record an API request lifecycle."""

    @abstractmethod
    def record_live_trade(
        self,
        symbol: str,
        side: str,
        size: float,
        fill_rate: float,
        slippage_bps: float,
        **labels: Any,
    ) -> None:
        """Record live trading execution metrics."""


class NullMetricsCollector(MetricsCollector):
    """In-memory placeholder that safely ignores all recorded metrics."""

    def record_strategy_signal(  # type: ignore[override]
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        return None

    def record_backtest_run(  # type: ignore[override]
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        return None

    def record_api_request(  # type: ignore[override]
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        return None

    def record_live_trade(  # type: ignore[override]
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        return None
