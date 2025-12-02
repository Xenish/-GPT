from __future__ import annotations

"""
Exchange health monitoring scaffolding with simple scoring and failover selection.

Integration notes:
- Data ingestion can poll health and switch to the next ranked exchange if the primary is DOWN/DEGRADED.
- Execution/routing can prefer exchanges with highest scores, subject to liquidity/fees.
- Data quality validation can feed into ExchangeHealth.data_quality_score for richer scoring.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping
import time

from finantradealgo.exchanges import ExchangeAdapter, ExchangeHealth, ExchangeId, ExchangeStatus


@dataclass
class HealthHistory:
    exchange: ExchangeId
    samples: list[ExchangeHealth] = field(default_factory=list)
    max_samples: int = 100

    def add(self, health: ExchangeHealth) -> None:
        self.samples.append(health)
        if len(self.samples) > self.max_samples:
            self.samples.pop(0)

    def latest(self) -> ExchangeHealth | None:
        return self.samples[-1] if self.samples else None

    def uptime_ratio(self, window_sec: float | None = None) -> float | None:
        if not self.samples:
            return None
        # Approximate uptime: fraction of samples not DOWN.
        healthy_count = sum(1 for s in self.samples if s.status != ExchangeStatus.DOWN)
        return healthy_count / len(self.samples)


class ExchangeHealthMonitor:
    def __init__(
        self,
        adapters: Mapping[ExchangeId, ExchangeAdapter],
    ) -> None:
        self.adapters = dict(adapters)
        self.history: dict[ExchangeId, HealthHistory] = {
            ex_id: HealthHistory(exchange=ex_id) for ex_id in self.adapters.keys()
        }

    def poll_once(self) -> dict[ExchangeId, ExchangeHealth]:
        results: dict[ExchangeId, ExchangeHealth] = {}
        now = time.time()

        for ex_id, adapter in self.adapters.items():
            health = adapter.get_health()
            if health.last_heartbeat_ts is None:
                health.last_heartbeat_ts = now
            results[ex_id] = health
            self.history[ex_id].add(health)

        return results

    def compute_scores(self) -> dict[ExchangeId, float]:
        scores: dict[ExchangeId, float] = {}
        for ex_id, history in self.history.items():
            latest = history.latest()
            if latest is None:
                scores[ex_id] = 0.0
                continue

            score = 1.0

            # Status penalties.
            if latest.status == ExchangeStatus.DEGRADED:
                score -= 0.2
            elif latest.status == ExchangeStatus.DOWN:
                score -= 0.6

            # Latency penalty (simple heuristic).
            if latest.avg_latency_ms is not None:
                if latest.avg_latency_ms > 1000:
                    score -= 0.2
                elif latest.avg_latency_ms > 500:
                    score -= 0.1

            # Data delay penalty.
            if latest.data_delay_ms is not None:
                if latest.data_delay_ms > 2000:
                    score -= 0.2
                elif latest.data_delay_ms > 500:
                    score -= 0.1

            # Data quality score (0-1) if provided.
            if latest.data_quality_score is not None:
                score *= max(0.0, min(1.0, latest.data_quality_score))

            # Uptime ratio over history.
            uptime = history.uptime_ratio()
            if uptime is not None:
                score *= uptime

            scores[ex_id] = max(0.0, min(1.0, score))

        return scores

    def ranked_exchanges(self) -> list[tuple[ExchangeId, float]]:
        scores = self.compute_scores()
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    def choose_preferred_exchanges(
        self,
        min_score: float = 0.5,
        max_exchanges: int | None = None,
    ) -> list[ExchangeId]:
        ranked = self.ranked_exchanges()
        filtered = [ex for ex, score in ranked if score >= min_score]
        if max_exchanges is not None:
            return filtered[:max_exchanges]
        return filtered

    def is_exchange_healthy(self, exchange: ExchangeId, *, threshold: float = 0.5) -> bool:
        scores = self.compute_scores()
        return scores.get(exchange, 0.0) >= threshold

