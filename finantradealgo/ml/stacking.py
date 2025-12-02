from __future__ import annotations

"""Classic stacking ensemble with OOF meta-features.

Design notes
------------
- Level 1 base models generate out-of-fold (OOF) predictions that train a level 2 meta-learner.
- Metrics/backtests live elsewhere; this module only builds/consumes model predictions.
- Integrates with existing `ModelSpec` definitions so configs can be shared with other pipelines.
- Limitations: assumes row-aligned X, y and does not yet support multi-output tasks.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from finantradealgo.ml import EstimatorLike, ModelSpec, TaskType


@dataclass
class StackingConfig:
    task_type: TaskType
    base_models: list[ModelSpec]
    meta_model: ModelSpec
    n_folds: int = 5
    shuffle: bool = True
    random_state: int | None = 42
    use_proba: bool = True  # classification: use probabilities as meta features
    metadata: dict[str, Any] | None = None


class StackingEnsemble:
    def __init__(self, config: StackingConfig) -> None:
        self.config = config
        self.base_models_: Dict[str, EstimatorLike] = {}
        self.meta_model_: EstimatorLike | None = None

    def _build_estimator(self, spec: ModelSpec) -> EstimatorLike:
        est = spec.estimator_factory()
        if spec.params:
            if hasattr(est, "set_params"):
                est = est.set_params(**spec.params)
            else:
                for key, value in spec.params.items():
                    setattr(est, key, value)
        return est

    def fit(self, X, y) -> "StackingEnsemble":
        n_samples = len(X)
        if self.config.task_type in {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}:
            cv = StratifiedKFold(
                n_splits=self.config.n_folds,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state,
            )
        else:
            cv = KFold(
                n_splits=self.config.n_folds,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state,
            )

        meta_features: List[np.ndarray] = []

        for spec in self.config.base_models:
            base_est = self._build_estimator(spec)

            if self.config.task_type == TaskType.REGRESSION:
                oof = np.zeros((n_samples, 1), dtype=float)
            else:
                if self.config.use_proba and hasattr(base_est, "predict_proba"):
                    # infer class count from y
                    classes = np.unique(y)
                    n_classes = len(classes)
                    oof = np.zeros((n_samples, n_classes), dtype=float)
                else:
                    oof = np.zeros((n_samples, 1), dtype=float)

            for train_idx, val_idx in cv.split(X, y):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val = X[val_idx]
                base_est.fit(X_train, y_train)

                if self.config.task_type == TaskType.REGRESSION:
                    preds = base_est.predict(X_val)
                    oof[val_idx, 0] = preds
                else:
                    if self.config.use_proba and hasattr(base_est, "predict_proba"):
                        proba = base_est.predict_proba(X_val)
                        # align to inferred class ordering
                        oof[val_idx, :] = proba
                    else:
                        preds = base_est.predict(X_val)
                        oof[val_idx, 0] = preds

            # Refit on full data for inference
            base_est_full = self._build_estimator(spec)
            base_est_full.fit(X, y)
            self.base_models_[spec.name] = base_est_full
            meta_features.append(oof)

        # Concatenate OOF predictions column-wise to form meta-features
        Z = np.concatenate(meta_features, axis=1)

        meta_est = self._build_estimator(self.config.meta_model)
        meta_est.fit(Z, y)
        self.meta_model_ = meta_est
        return self

    def _stack_meta_features(self, X) -> np.ndarray:
        feats: List[np.ndarray] = []
        for spec in self.config.base_models:
            model = self.base_models_[spec.name]
            if self.config.task_type == TaskType.REGRESSION:
                preds = model.predict(X)
                feats.append(preds.reshape(-1, 1))
            else:
                if self.config.use_proba and hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)
                    feats.append(proba)
                else:
                    preds = model.predict(X)
                    feats.append(preds.reshape(-1, 1))
        return np.concatenate(feats, axis=1)

    def predict(self, X) -> np.ndarray:
        if self.meta_model_ is None:
            raise RuntimeError("Meta-model is not fitted. Call fit first.")
        Z = self._stack_meta_features(X)
        return self.meta_model_.predict(Z)

    def predict_proba(self, X) -> np.ndarray:
        if self.config.task_type not in {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}:
            raise ValueError("predict_proba is only available for classification tasks.")
        if self.meta_model_ is None:
            raise RuntimeError("Meta-model is not fitted. Call fit first.")
        if not hasattr(self.meta_model_, "predict_proba"):
            raise AttributeError("Meta-model does not implement predict_proba.")

        Z = self._stack_meta_features(X)
        return self.meta_model_.predict_proba(Z)
