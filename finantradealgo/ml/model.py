from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


@dataclass
class SklearnModelConfig:
    model_type: str = "gradient_boosting"
    params: Dict[str, Any] = field(default_factory=dict)
    random_state: int = 42


class SklearnLongModel:
    def __init__(self, config: SklearnModelConfig | None = None):
        self.config = config or SklearnModelConfig()
        self.clf = self._build_classifier()
        self.is_fitted = False

    def _build_classifier(self):
        model_type = self.config.model_type.lower()
        params = dict(self.config.params or {})

        if model_type in ("gradient_boosting", "gbm", "gb"):
            defaults = {
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 3,
                "random_state": self.config.random_state,
            }
            defaults.update(params)
            return GradientBoostingClassifier(**defaults)

        if model_type in ("logreg", "logistic_regression"):
            defaults = {
                "C": 1.0,
                "max_iter": 200,
                "class_weight": "balanced",
                "solver": "lbfgs",
            }
            defaults.update(params)
            return LogisticRegression(**defaults)

        if model_type in ("random_forest", "rf"):
            defaults = {
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_leaf": 2,
                "class_weight": "balanced_subsample",
                "random_state": self.config.random_state,
                "n_jobs": -1,
            }
            defaults.update(params)
            return RandomForestClassifier(**defaults)

        if model_type in ("xgboost", "xgb"):
            if XGBClassifier is None:
                raise ImportError(
                    "xgboost is not installed. Install it or choose another model_type."
                )
            defaults = {
                "n_estimators": 300,
                "max_depth": 3,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "random_state": self.config.random_state,
                "n_jobs": -1,
                "verbosity": 0,
            }
            defaults.update(params)
            return XGBClassifier(**defaults)

        raise ValueError(f"Unknown model_type: {self.config.model_type}")

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
    ) -> None:
        self.clf.fit(X_train, y_train)
        self.is_fitted = True

    def predict_proba(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        proba = self.clf.predict_proba(X)
        return proba

    def evaluate(
        self,
        X_test: pd.DataFrame | np.ndarray,
        y_test: pd.Series | np.ndarray,
    ) -> dict:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        proba = self.predict_proba(X_test)
        if proba.shape[1] < 2:
            raise ValueError("predict_proba should return at least 2 columns.")
        pos_proba = proba[:, 1]
        y_pred = (pos_proba >= 0.5).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        }

        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, pos_proba))
        except ValueError:
            metrics["roc_auc"] = float("nan")

        return metrics

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"config": self.config, "model": self.clf}, path)

    @classmethod
    def load(cls, path: str | Path) -> "SklearnLongModel":
        payload = joblib.load(path)
        config = payload["config"]
        clf = payload["model"]

        obj = cls(config)
        obj.clf = clf
        obj.is_fitted = True
        return obj
