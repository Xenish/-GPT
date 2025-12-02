from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Mapping, Sequence

# Strategy identifier, e.g. "trend_follow_v2"
StrategyId = str


@dataclass
class StrategyVersion:
    id: StrategyId
    version: str  # semantic version or git hash
    git_ref: str | None = None  # commit hash / tag
    path: str | None = None  # module or file path
    created_at: Any | None = None
    created_by: str | None = None
    dependencies: dict[str, str] | None = None  # package -> version
    metadata: dict[str, Any] | None = None


class DeploymentEnvironment(Enum):
    RESEARCH = auto()
    PAPER = auto()
    LIVE = auto()


class DeploymentStatus(Enum):
    PENDING = auto()
    DEPLOYED = auto()
    ROLLED_BACK = auto()
    FAILED = auto()
    DEPRECATED = auto()


@dataclass
class StrategyDeploymentConfig:
    strategy: StrategyVersion
    environment: DeploymentEnvironment
    capital_fraction: float = 1.0  # 0-1 of available capital
    max_leverage: float | None = None
    risk_limits: dict[str, Any] | None = None  # e.g. daily loss, max drawdown
    auto_rollout: bool = False  # allow promotion to higher capital/env
    metadata: dict[str, Any] | None = None


@dataclass
class DeploymentRecord:
    config: StrategyDeploymentConfig
    status: DeploymentStatus
    started_at: Any | None = None
    completed_at: Any | None = None
    rollback_to: StrategyVersion | None = None
    notes: str | None = None
    metrics_snapshot: dict[str, float] | None = None
