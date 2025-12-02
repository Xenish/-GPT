"""Feature importance and interaction mining utilities."""

from __future__ import annotations

import dataclasses
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from . import DiscoveryResult, StrategyCandidate


@dataclass(slots=True)
class FeatureMiningConfig:
    """Configuration for feature importance analysis."""

    methods: list[str] = field(default_factory=lambda: ["permutation"])
    metric_name: str = "accuracy"
    n_samples: int | None = None
    random_seed: int | None = None


@dataclass(slots=True)
class FeatureImportanceResult:
    """Aggregated feature importance scores and interactions."""

    importances: dict[str, float]
    per_method: dict[str, dict[str, float]] = field(default_factory=dict)
    interactions: dict[tuple[str, str], float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "importances": self.importances,
            "per_method": self.per_method,
            "interactions": {f"{a}|{b}": v for (a, b), v in self.interactions.items()},
            "metadata": self.metadata,
        }


class FeatureMiner:
    """Compute feature importance and simple interactions for estimators/signals."""

    def __init__(
        self,
        model: Any,
        X: Any,
        y: Any,
        config: FeatureMiningConfig,
        scorer: Callable[[Any, Any, Any], float] | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        """
        Args:
            model: Trained estimator supporting predict / predict_proba or a project-specific interface.
            X: Feature matrix (numpy array, pandas DataFrame, or compatible).
            y: Targets/labels.
            config: Mining configuration.
            scorer: Optional callable (model, X, y) -> metric. Defaults to accuracy for classification
                    or negative MSE for regression if not provided.
            feature_names: Optional explicit feature names; inferred from DataFrame columns if absent.
        """
        self.model = model
        self.X = X
        self.y = y
        self.config = config
        self.feature_names = (
            list(X.columns) if feature_names is None and hasattr(X, "columns") else list(feature_names or [])
        )
        self._rng = random.Random(config.random_seed)
        self.scorer = scorer or self._default_scorer

    # --- Public API -----------------------------------------------------

    def compute_shap_importance(self) -> FeatureImportanceResult:
        """Compute feature importance using SHAP values if available."""
        try:
            import shap  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "SHAP is not available in the environment; install shap to enable this method."
            ) from exc

        explainer = shap.Explainer(self.model, self.X)
        shap_values = explainer(self.X, check_additivity=False)
        values = np.abs(shap_values.values).mean(axis=0)
        if values.ndim > 1:
            values = values.mean(axis=0)
        importances = self._normalize_importances(values)
        per_method = {"shap": dict(zip(self._feature_names(len(values)), importances))}
        return FeatureImportanceResult(importances=per_method["shap"], per_method=per_method)

    def compute_permutation_importance(self) -> FeatureImportanceResult:
        """Compute permutation importance by measuring metric degradation per feature."""
        baseline = self.scorer(self.model, self.X, self.y)
        X_perm = self._to_numpy(self.X)
        n_features = X_perm.shape[1]
        scores: dict[str, float] = {}
        for idx in range(n_features):
            X_copy = X_perm.copy()
            self._rng.shuffle(list(X_copy[:, idx]))
            permuted_score = self.scorer(self.model, X_copy, self.y)
            scores[self._feature_names(n_features)[idx]] = baseline - permuted_score

        normalized = self._normalize_importances(np.array(list(scores.values())))
        scores = dict(zip(self._feature_names(n_features), normalized))
        return FeatureImportanceResult(importances=scores, per_method={"permutation": scores})

    def detect_interactions(self) -> dict[tuple[str, str], float]:
        """Naive pairwise interaction measure using joint permutation drops."""
        baseline = self.scorer(self.model, self.X, self.y)
        X_np = self._to_numpy(self.X)
        n_features = X_np.shape[1]
        interactions: dict[tuple[str, str], float] = {}
        names = self._feature_names(n_features)
        for i in range(n_features):
            for j in range(i + 1, n_features):
                X_copy = X_np.copy()
                self._rng.shuffle(list(X_copy[:, i]))
                self._rng.shuffle(list(X_copy[:, j]))
                score_drop = baseline - self.scorer(self.model, X_copy, self.y)
                interactions[(names[i], names[j])] = score_drop
        return interactions

    def run_all(self) -> FeatureImportanceResult:
        """Run all configured methods and aggregate results."""
        combined_importances: dict[str, float] = {}
        per_method: dict[str, dict[str, float]] = {}

        if "shap" in self.config.methods:
            shap_res = self.compute_shap_importance()
            per_method["shap"] = shap_res.importances
            combined_importances = self._combine(combined_importances, shap_res.importances)

        if "permutation" in self.config.methods:
            perm_res = self.compute_permutation_importance()
            per_method["permutation"] = perm_res.importances
            combined_importances = self._combine(combined_importances, perm_res.importances)

        interactions = self.detect_interactions()
        return FeatureImportanceResult(
            importances=combined_importances,
            per_method=per_method,
            interactions=interactions,
            metadata={"methods": self.config.methods, "metric": self.config.metric_name},
        )

    def to_candidates(self, result: FeatureImportanceResult) -> list[StrategyCandidate]:
        """Optional mapping of importance results to StrategyCandidate objects."""
        candidates: list[StrategyCandidate] = []
        for idx, (name, score) in enumerate(sorted(result.importances.items(), key=lambda kv: kv[1], reverse=True)):
            candidates.append(
                StrategyCandidate(
                    candidate_id=f"feat_{idx}",
                    name=f"Feature:{name}",
                    description=f"Strategy using feature {name}",
                    tags={"feature"},
                    metrics={"importance": score},
                    metadata={"feature_name": name, "per_method": result.per_method.get(name)},
                )
            )
        return candidates

    # --- Internal helpers ----------------------------------------------

    def _default_scorer(self, model: Any, X: Any, y: Any) -> float:
        """Heuristic scorer: accuracy for classification, negative MSE for regression."""
        preds = self._predict(model, X)
        if self._is_classification(y, preds):
            correct = (preds == y).mean()
            return float(correct)
        errors = (preds - y) ** 2
        return float(-errors.mean())

    def _predict(self, model: Any, X: Any) -> Any:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            return np.argmax(proba, axis=1)
        if hasattr(model, "predict"):
            return model.predict(X)
        raise ValueError("Model must implement predict or predict_proba.")

    def _normalize_importances(self, values: np.ndarray) -> np.ndarray:
        values = np.maximum(values, 0)
        total = values.sum()
        if total == 0:
            return values
        return values / total

    def _feature_names(self, n_features: int) -> list[str]:
        if self.feature_names:
            return list(self.feature_names)
        return [f"f{i}" for i in range(n_features)]

    def _to_numpy(self, X: Any) -> np.ndarray:
        if hasattr(X, "to_numpy"):
            return X.to_numpy()
        return np.asarray(X)

    def _combine(self, base: dict[str, float], new: dict[str, float]) -> dict[str, float]:
        combined = dict(base)
        for k, v in new.items():
            combined[k] = combined.get(k, 0.0) + v
        return combined

    def _is_classification(self, y: Any, preds: Any) -> bool:
        try:
            return set(np.unique(y)) <= {0, 1} or len(np.unique(y)) < 20
        except Exception:
            return False


__all__ = [
    "FeatureImportanceResult",
    "FeatureMiner",
    "FeatureMiningConfig",
]
