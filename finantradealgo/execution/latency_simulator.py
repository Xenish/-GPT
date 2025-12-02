from __future__ import annotations

"""
Latency simulator for execution modeling.

Models a simple latency stack: network + exchange processing + queueing. Each
component is sampled from a normal distribution around configurable means and
standard deviations, then scaled by liquidity regime multipliers if provided.
No real sleeping is performed; this purely shifts timestamps for simulation.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
import random
from typing import Any

from finantradealgo.execution import ExecutionContext


@dataclass
class LatencyModelConfig:
    base_network_ms: float = 20.0
    network_jitter_ms: float = 10.0
    base_exchange_ms: float = 5.0
    exchange_jitter_ms: float = 3.0
    queue_delay_ms: float = 0.0
    queue_jitter_ms: float = 0.0
    liquidity_regime_multiplier: dict[str, float] | None = None
    min_latency_ms: float = 1.0
    max_latency_ms: float = 5000.0
    metadata: dict[str, Any] | None = None


class LatencySimulator:
    def __init__(
        self,
        model_config: LatencyModelConfig | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.model_config = model_config or LatencyModelConfig()
        self._rng = rng or random.Random()

    def sample_latency_ms(self, ctx: ExecutionContext) -> float:
        """
        Sample total latency (ms) from network + exchange + queue components,
        adjusted by liquidity regime multipliers and clamped to sane bounds.
        """
        cfg = self.model_config
        net = self._sample_component(cfg.base_network_ms, cfg.network_jitter_ms)
        exch = self._sample_component(cfg.base_exchange_ms, cfg.exchange_jitter_ms)
        queue = self._sample_component(cfg.queue_delay_ms, cfg.queue_jitter_ms)

        total = net + exch + queue

        multiplier = 1.0
        if cfg.liquidity_regime_multiplier and ctx.liquidity_regime:
            multiplier = cfg.liquidity_regime_multiplier.get(ctx.liquidity_regime, 1.0)
        total *= multiplier

        total = max(cfg.min_latency_ms, total)
        total = min(cfg.max_latency_ms, total)
        return total

    def apply_latency(self, ctx: ExecutionContext, base_timestamp: datetime) -> datetime:
        """
        Apply sampled latency to the provided base timestamp, returning the
        effective execution timestamp.
        """
        latency_ms = self.sample_latency_ms(ctx)
        return base_timestamp + timedelta(milliseconds=latency_ms)

    def _sample_component(self, mean_ms: float, jitter_ms: float) -> float:
        if jitter_ms <= 0:
            return max(0.0, mean_ms)
        return max(0.0, self._rng.normalvariate(mean_ms, jitter_ms))


__all__ = ["LatencyModelConfig", "LatencySimulator"]
