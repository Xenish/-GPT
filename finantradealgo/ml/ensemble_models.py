from __future__ import annotations

"""Model blending utilities (uniform, weighted, performance-based, dynamic).

Design notes
------------
- Metrics should be provided from validation/backtest runs via `update_performance`.
- This module does not run backtests; it consumes metrics to adjust weights.
- Integrates with existing training in `model.py`: train base models there, then
  pass fitted estimators (or let `fit` build them from `ModelSpec.estimator_factory`).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from finantradealgo.ml import BlendMethod, EnsembleConfig, EstimatorLike, ModelSpec, TaskType


@dataclass
class ModelPerformance:
    name: str
    metrics: dict[str, float]
    timestamp: Any | None = None
    metadata: dict[str, Any] | None = None


class BlendedEnsemble:
    def __init__(
        self,
        config: EnsembleConfig,
        models: dict[str, EstimatorLike] | None = None,
    ) -> None:
        """
        If `models` is provided, it should map model_spec.name -> fitted estimator.
        Otherwise, `fit` will instantiate estimators from ModelSpec.estimator_factory.
        """
        self.config: EnsembleConfig = config
        self.model_specs: List[ModelSpec] = list(config.models)
        self.models: Dict[str, EstimatorLike] = models or {}
        self.weights: Dict[str, float] = {}
        self.performance_history: Dict[str, List[ModelPerformance]] = {
            spec.name: [] for spec in self.model_specs
        }

    def fit(self, X, y, sample_weight=None) -> "BlendedEnsemble":
        if not self.models:
            for spec in self.model_specs:
                estimator = spec.estimator_factory()
                if spec.params:
                    if hasattr(estimator, "set_params"):
                        estimator = estimator.set_params(**spec.params)
                    else:
                        for key, value in spec.params.items():
                            setattr(estimator, key, value)
                estimator.fit(X, y, sample_weight=sample_weight)
                self.models[spec.name] = estimator

        # Initialize weights
        if self.config.blend_method == BlendMethod.UNIFORM:
            base_weights = {spec.name: 1.0 for spec in self.model_specs}
        elif self.config.blend_method == BlendMethod.STATIC_WEIGHTS:
            base_weights = {spec.name: float(spec.weight) for spec in self.model_specs}
        else:
            # PERFORMANCE_WEIGHTED or DYNAMIC start with static/uniform before metrics arrive
            base_weights = {spec.name: float(spec.weight) for spec in self.model_specs}
            if all(w == 0.0 for w in base_weights.values()):
                base_weights = {spec.name: 1.0 for spec in self.model_specs}

        total = sum(base_weights.values()) or 1.0
        self.weights = {name: w / total for name, w in base_weights.items()}
        return self

    def update_performance(self, perf_list: Sequence[ModelPerformance]) -> None:
        metric_key = self.config.metric_name
        latest_scores: Dict[str, float] = {}

        for perf in perf_list:
            if perf.name not in self.performance_history:
                # Unknown model; skip but keep history consistent.
                continue
            self.performance_history[perf.name].append(perf)
            score = perf.metrics.get(metric_key)
            if score is not None:
                latest_scores[perf.name] = float(score)

        if self.config.blend_method not in {BlendMethod.PERFORMANCE_WEIGHTED, BlendMethod.DYNAMIC}:
            return

        # Compute weights from provided metric
        scores = []
        names = []
        for spec in self.model_specs:
            val = latest_scores.get(spec.name)
            if val is None:
                continue
            # Handle negative metrics gracefully by flooring at zero
            scores.append(max(val, 0.0))
            names.append(spec.name)

        if not scores or sum(scores) == 0.0:
            # Fallback to uniform if no usable metrics
            new_weights = {spec.name: 1.0 / len(self.model_specs) for spec in self.model_specs}
        else:
            score_arr = np.asarray(scores, dtype=float)
            weight_arr = score_arr / score_arr.sum()
            new_weights = {name: float(w) for name, w in zip(names, weight_arr, strict=False)}
            # Any model without a score gets zero weight
            for spec in self.model_specs:
                new_weights.setdefault(spec.name, 0.0)

        if self.config.blend_method == BlendMethod.DYNAMIC:
            # Exponential moving average smoothing to avoid abrupt shifts
            alpha = 0.7
            smoothed: Dict[str, float] = {}
            for name in self.weights.keys() | new_weights.keys():
                prev = self.weights.get(name, 0.0)
                curr = new_weights.get(name, 0.0)
                smoothed[name] = alpha * curr + (1 - alpha) * prev
            total = sum(smoothed.values()) or 1.0
            self.weights = {n: v / total for n, v in smoothed.items()}
        else:
            total = sum(new_weights.values()) or 1.0
            self.weights = {n: v / total for n, v in new_weights.items()}

    def _compute_blend_weights(self) -> np.ndarray:
        """
        Return weights in the order of self.model_specs (normalized).
        """
        weights = np.array([self.weights.get(spec.name, 0.0) for spec in self.model_specs], dtype=float)
        total = weights.sum()
        if total <= 0:
            return np.full_like(weights, 1.0 / len(self.model_specs))
        return weights / total

    def predict_proba(self, X) -> np.ndarray:
        if self.config.task_type not in {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}:
            raise ValueError("predict_proba is only available for classification tasks.")

        weights = self._compute_blend_weights()
        probas = None
        for weight, spec in zip(weights, self.model_specs, strict=False):
            model = self.models[spec.name]
            if not hasattr(model, "predict_proba"):
                raise AttributeError(f"Model '{spec.name}' does not implement predict_proba.")
            model_proba = model.predict_proba(X)
            if probas is None:
                probas = np.zeros_like(model_proba, dtype=float)
            probas += weight * model_proba
        return probas

    def predict(self, X) -> np.ndarray:
        weights = self._compute_blend_weights()

        if self.config.task_type == TaskType.REGRESSION:
            preds = None
            for weight, spec in zip(weights, self.model_specs, strict=False):
                model = self.models[spec.name]
                model_pred = model.predict(X)
                if preds is None:
                    preds = np.zeros_like(model_pred, dtype=float)
                preds += weight * model_pred
            return preds

        # Classification: blend probabilities when available
        if any(hasattr(self.models[spec.name], "predict_proba") for spec in self.model_specs):
            blended_proba = self.predict_proba(X)
            return np.argmax(blended_proba, axis=1)

        # Fallback: weighted votes on class predictions
        class_votes = {}
        for weight, spec in zip(weights, self.model_specs, strict=False):
            model = self.models[spec.name]
            preds = model.predict(X)
            for label in np.unique(preds):
                mask = preds == label
                class_votes.setdefault(label, np.zeros_like(preds, dtype=float))
                class_votes[label][mask] += weight

        # Pick class with highest vote per sample
        stacked = np.vstack([v for v in class_votes.values()])
        best_idx = np.argmax(stacked, axis=0)
        classes = list(class_votes.keys())
        return np.array([classes[i] for i in best_idx])
