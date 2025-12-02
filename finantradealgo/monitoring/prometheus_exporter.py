"""Prometheus-backed metrics collector implementation."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from prometheus_client import Counter, Gauge, Histogram

from .metrics_collector import MetricsCollector, MonitoringConfig, NullMetricsCollector

# Default buckets tuned for millisecond latency metrics and larger runtime values.
_LATENCY_BUCKETS_MS: tuple[float, ...] = (
    1,
    5,
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
    2500,
    5000,
)
_RUNTIME_BUCKETS_SECONDS: tuple[float, ...] = (
    0.5,
    1,
    2,
    5,
    10,
    30,
    60,
    120,
    300,
    600,
)


class PrometheusMetricsCollector(MetricsCollector):
    """Prometheus-backed implementation of the metrics collector."""

    def __init__(self, config: MonitoringConfig | None = None) -> None:
        super().__init__(config=config)
        self._default_labels = self._stringify_labels(self.config.labels or {})
        self._default_label_names = tuple(self._default_labels.keys())

        self.strategy_signals = Counter(
            "finantrade_strategy_signals_total",
            "Total number of strategy signals emitted.",
            ("strategy_name", "symbol", "signal_type", "status", *self._default_label_names),
        )
        self.strategy_signal_latency = Histogram(
            "finantrade_strategy_signal_latency_ms",
            "Latency of strategy signal processing in milliseconds.",
            ("strategy_name", "symbol", "signal_type", "status", *self._default_label_names),
            buckets=_LATENCY_BUCKETS_MS,
        )

        self.backtest_runtime = Histogram(
            "finantrade_backtest_runtime_seconds",
            "Duration of backtest runs in seconds.",
            ("name", "status", *self._default_label_names),
            buckets=_RUNTIME_BUCKETS_SECONDS,
        )
        self.backtest_memory_mb = Gauge(
            "finantrade_backtest_memory_mb",
            "Memory usage of backtest runs in megabytes.",
            ("name", *self._default_label_names),
        )

        self.api_requests = Counter(
            "finantrade_api_requests_total",
            "Total API requests handled.",
            ("endpoint", "method", "status_code", *self._default_label_names),
        )
        self.api_latency = Histogram(
            "finantrade_api_latency_ms",
            "API request latency in milliseconds.",
            ("endpoint", "method", "status_code", *self._default_label_names),
            buckets=_LATENCY_BUCKETS_MS,
        )

        self.live_trades = Counter(
            "finantrade_live_trades_total",
            "Total live trading executions.",
            ("symbol", "side", "status", *self._default_label_names),
        )
        self.live_trade_slippage = Histogram(
            "finantrade_live_trade_slippage_bps",
            "Observed slippage in basis points for live trades.",
            ("symbol", "side", "status", *self._default_label_names),
            buckets=(0.1, 0.5, 1, 2, 5, 10, 20, 50, 100),
        )
        self.live_position_size = Gauge(
            "finantrade_live_position_size",
            "Latest observed position size per symbol and side.",
            ("symbol", "side", *self._default_label_names),
        )

    def record_strategy_signal(
        self,
        strategy_name: str,
        symbol: str,
        event_type: str,
        latency_ms: float,
        success: bool = True,
        **labels: Any,
    ) -> None:
        if not self.config.enabled:
            return

        status = "success" if success else "failure"
        label_values = self._merge_labels(
            {
                "strategy_name": strategy_name,
                "symbol": symbol,
                "signal_type": event_type,
                "status": status,
            },
            labels,
        )
        self.strategy_signals.labels(**label_values).inc()
        self.strategy_signal_latency.labels(**label_values).observe(float(latency_ms))

    def record_backtest_run(
        self,
        name: str,
        duration_seconds: float,
        memory_mb: float | None = None,
        success: bool = True,
        **labels: Any,
    ) -> None:
        if not self.config.enabled:
            return

        status = "success" if success else "failure"
        merged = self._merge_labels(
            {
                "name": name,
                "status": status,
            },
            labels,
        )
        self.backtest_runtime.labels(**merged).observe(float(duration_seconds))
        if memory_mb is not None:
            gauge_labels = self._merge_labels({"name": name}, labels)
            self.backtest_memory_mb.labels(**gauge_labels).set(float(memory_mb))

    def record_api_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration_ms: float,
        **labels: Any,
    ) -> None:
        if not self.config.enabled:
            return

        label_values = self._merge_labels(
            {
                "endpoint": endpoint,
                "method": method.upper(),
                "status_code": str(status_code),
            },
            labels,
        )
        self.api_requests.labels(**label_values).inc()
        self.api_latency.labels(**label_values).observe(float(duration_ms))

    def record_live_trade(
        self,
        symbol: str,
        side: str,
        size: float,
        fill_rate: float,
        slippage_bps: float,
        **labels: Any,
    ) -> None:
        if not self.config.enabled:
            return

        status = "filled" if fill_rate >= 1.0 else "partial"
        label_values = self._merge_labels(
            {
                "symbol": symbol,
                "side": side,
                "status": status,
            },
            labels,
        )
        self.live_trades.labels(**label_values).inc()
        self.live_trade_slippage.labels(**label_values).observe(float(slippage_bps))

        position_labels = self._merge_labels(
            {
                "symbol": symbol,
                "side": side,
            },
            labels,
        )
        self.live_position_size.labels(**position_labels).set(float(size))

    def _merge_labels(
        self,
        base: Dict[str, Any],
        extra: Dict[str, Any] | None = None,
    ) -> Dict[str, str]:
        merged: Dict[str, Any] = dict(self._default_labels)
        merged.update(base)
        if extra:
            merged.update(extra)
        return self._stringify_labels(merged)

    @staticmethod
    def _stringify_labels(values: Mapping[str, Any]) -> Dict[str, str]:
        return {key: str(value) for key, value in values.items()}


_PROM_COLLECTOR: PrometheusMetricsCollector | None = None


def get_metrics_collector(config: MonitoringConfig) -> MetricsCollector:
    """Choose the active metrics collector based on monitoring flags."""

    if config.enabled and config.prometheus_enabled:
        global _PROM_COLLECTOR
        if _PROM_COLLECTOR is None:
            _PROM_COLLECTOR = PrometheusMetricsCollector(config=config)
        return _PROM_COLLECTOR

    return NullMetricsCollector(config=config)
