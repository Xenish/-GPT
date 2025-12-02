"""Strategy Discovery & Mining base interfaces and shared data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

TStrategyCandidate = TypeVar("TStrategyCandidate", bound="StrategyCandidate")


@dataclass(slots=True)
class StrategyCandidate:
    """Lightweight representation of a candidate trading strategy."""

    candidate_id: str
    name: str | None = None
    description: str | None = None
    tags: set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    fitness: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def primary_score(self) -> float | None:
        """Return the primary score used for ranking, defaulting to fitness."""
        return self.fitness if self.fitness is not None else self.metrics.get("fitness")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the candidate into primitive Python types."""
        return {
            "candidate_id": self.candidate_id,
            "name": self.name,
            "description": self.description,
            "tags": sorted(self.tags),
            "created_at": self.created_at.isoformat(),
            "fitness": self.fitness,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class DiscoveryConfig:
    """Common configuration for discovery runs."""

    random_seed: int | None = None
    max_candidates: int = 100
    max_runtime_seconds: int | None = None
    max_generations: int | None = None
    evaluation_parallelism: int | None = None
    scoring_metric: str | None = None
    data_range: tuple[datetime | None, datetime | None] | None = None
    allow_partial_results: bool = True
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration for logging or persistence."""
        return {
            "random_seed": self.random_seed,
            "max_candidates": self.max_candidates,
            "max_runtime_seconds": self.max_runtime_seconds,
            "max_generations": self.max_generations,
            "evaluation_parallelism": self.evaluation_parallelism,
            "scoring_metric": self.scoring_metric,
            "data_range": tuple(
                dt.isoformat() if dt is not None else None for dt in self.data_range
            )
            if self.data_range
            else None,
            "allow_partial_results": self.allow_partial_results,
            "notes": self.notes,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class DiscoveryResult(Generic[TStrategyCandidate]):
    """Container for discovered strategies and run metadata."""

    method: str
    candidates: list[TStrategyCandidate] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    data_range: tuple[datetime | None, datetime | None] | None = None
    config: DiscoveryConfig | None = None
    total_candidates_evaluated: int = 0
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def best_candidate(self) -> TStrategyCandidate | None:
        """Return the highest scoring candidate, if any."""
        if not self.candidates:
            return None

        def _score(candidate: StrategyCandidate) -> float:
            score = candidate.primary_score()
            return score if score is not None else float("-inf")

        return max(self.candidates, key=_score)

    def duration_seconds(self) -> float | None:
        """Elapsed runtime in seconds if the run has finished."""
        if not self.completed_at:
            return None
        return (self.completed_at - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Serialize result for storage or reporting."""
        return {
            "method": self.method,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "data_range": tuple(
                dt.isoformat() if dt is not None else None for dt in self.data_range
            )
            if self.data_range
            else None,
            "config": self.config.to_dict() if self.config else None,
            "total_candidates_evaluated": self.total_candidates_evaluated,
            "notes": self.notes,
            "metadata": self.metadata,
        }


__all__ = [
    "DiscoveryConfig",
    "DiscoveryResult",
    "StrategyCandidate",
    "TStrategyCandidate",
]
