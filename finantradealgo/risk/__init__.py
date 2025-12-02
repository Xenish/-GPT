from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Protocol, TYPE_CHECKING, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


@runtime_checkable
class ReturnSeriesLike(Protocol):
    """Represents a 1D series of returns (numpy array, pandas Series, list, etc.)."""

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterable[float | int]: ...

    def __getitem__(self, idx: int) -> float | int: ...


@dataclass
class RiskMetricConfig:
    confidence_level: float = 0.95
    horizon_days: int = 1
    use_log_returns: bool = True
    annualization_factor: float | None = 252.0
    metadata: dict[str, Any] | None = None


class VaRMethod(Enum):
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"


@dataclass
class VaRResult:
    method: VaRMethod
    confidence_level: float
    horizon_days: int
    var_value: float
    cvar_value: float | None
    sample_size: int
    metadata: dict[str, Any] | None


class StressScenarioType(Enum):
    HISTORICAL = "historical"
    HYPOTHETICAL = "hypothetical"


@dataclass
class StressScenario:
    name: str
    scenario_type: StressScenarioType
    description: str
    shock_returns: np.ndarray | None
    shocks: dict[str, float] | None
    metadata: dict[str, Any] | None


@dataclass
class StressTestResult:
    scenario: StressScenario
    pnl_series: list[float] | None
    equity_curve: list[float] | None
    max_drawdown: float | None
    var_like_loss: float | None
    summary_metrics: dict[str, float]
    metadata: dict[str, Any] | None


@dataclass
class TailRiskMetrics:
    confidence_level: float
    var: float
    cvar: float
    expected_max_drawdown: float | None
    tail_index: float | None
    omega_ratio: float | None
    information_ratio: float | None
    extra: dict[str, Any] | None
