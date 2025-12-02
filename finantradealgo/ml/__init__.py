from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Protocol, Sequence


class EstimatorLike(Protocol):
    def fit(self, X, y, *args, **kwargs) -> "EstimatorLike":
        ...

    def predict(self, X) -> Any:
        ...

    def predict_proba(self, X) -> Any:
        ...


class TaskType(Enum):
    BINARY_CLASSIFICATION = auto()
    MULTICLASS_CLASSIFICATION = auto()
    REGRESSION = auto()


@dataclass
class ModelSpec:
    name: str
    task_type: TaskType
    estimator_factory: Callable[[], EstimatorLike]
    weight: float = 1.0
    params: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class BlendMethod(Enum):
    UNIFORM = auto()
    STATIC_WEIGHTS = auto()
    PERFORMANCE_WEIGHTED = auto()
    DYNAMIC = auto()


@dataclass
class EnsembleConfig:
    task_type: TaskType
    models: list[ModelSpec]
    blend_method: BlendMethod = BlendMethod.UNIFORM
    metric_name: str = "sharpe"
    dynamic_update: bool = False
    metadata: dict[str, Any] | None = None
