from __future__ import annotations

import json
import os
import sys
import hashlib
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .labels import LabelConfig

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


@dataclass
class SklearnModelConfig:
    """
    Configuration for sklearn-based models.

    Attributes:
        model_type: Identifier for estimator ("gradient_boosting", "random_forest", "logreg", "xgboost")
        params: Estimator hyperparameters
        random_state: Seed propagated to estimators for determinism
    """

    model_type: str = "gradient_boosting"
    params: Dict[str, Any] = field(default_factory=dict)
    random_state: int = 42

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "SklearnModelConfig":
        data = data or {}
        model_type = data.get("type", data.get("model_type", cls.model_type))
        random_state = data.get("random_state", cls.random_state)
        params = data.get("params")
        if params is None:
            params = {
                key: value
                for key, value in data.items()
                if key not in {"type", "model_type", "random_state"}
            }
        return cls(model_type=model_type, params=params, random_state=random_state)


@dataclass
class ModelMetadata:
    model_id: str
    symbol: str
    timeframe: str
    model_type: str
    created_at: str
    feature_preset: str
    feature_cols: List[str]
    label_config: Dict[str, Any]
    train_start: str
    train_end: str
    metrics: Dict[str, float]
    model_path: str
    metrics_path: str
    meta_path: str
    pipeline_version: str = "unknown"
    random_state: Optional[int] = None
    python_version: str = ""
    sklearn_version: str = ""
    pandas_version: str = ""
    feature_importances: Optional[Dict[str, float]] = None
    feature_schema_hash: Optional[str] = None
    seed: Optional[int] = None
    config_snapshot: Optional[Dict[str, Any]] = None


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

        if model_type in ("random_forest", "rf", "randomforest"):
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


def set_global_seed(seed: int) -> None:
    """
    Set Python and numpy RNG seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)


def compute_feature_schema_hash(feature_cols: List[str]) -> str:
    """
    Compute a deterministic hash for a set of feature columns.
    """
    normalized = ",".join(sorted(feature_cols))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def save_sklearn_model(
    model: Any,
    *,
    symbol: str,
    timeframe: str,
    model_cfg: SklearnModelConfig,
    label_cfg: LabelConfig,
    feature_preset: str,
    feature_cols: List[str],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    metrics: Dict[str, float],
    base_dir: str = "outputs/ml_models",
    pipeline_version: str = "unknown",
    seed: Optional[int] = None,
    config_snapshot: Optional[Dict[str, Any]] = None,
) -> ModelMetadata:
    os.makedirs(base_dir, exist_ok=True)

    ts_dt = datetime.now(UTC)
    ts_id = ts_dt.strftime("%Y%m%d_%H%M%S")
    created_iso = ts_dt.isoformat()
    model_id = f"{symbol}_{timeframe}_{model_cfg.model_type}_{ts_id}"

    run_dir = os.path.join(base_dir, model_id)
    os.makedirs(run_dir, exist_ok=True)

    model_path = os.path.join(run_dir, "model.joblib")
    metrics_path = os.path.join(run_dir, "metrics.csv")
    meta_path = os.path.join(run_dir, "meta.json")

    joblib.dump(model, model_path)

    if metrics:
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    feature_importances: Optional[Dict[str, float]] = None
    if feature_cols and hasattr(model, "feature_importances_"):
        raw = getattr(model, "feature_importances_", None)
        if raw is not None:
            arr = np.asarray(raw, dtype=float)
            if arr.size:
                total = float(arr.sum()) or 1.0
                feature_importances = {
                    name: float(val) / total
                    for name, val in zip(feature_cols, arr)
                }

    feature_schema_hash = compute_feature_schema_hash(feature_cols)

    meta = ModelMetadata(
        model_id=model_id,
        symbol=symbol,
        timeframe=timeframe,
        model_type=model_cfg.model_type,
        created_at=created_iso,
        pipeline_version=pipeline_version,
        feature_preset=feature_preset,
        feature_cols=list(feature_cols),
        label_config=asdict(label_cfg),
        train_start=str(pd.to_datetime(train_start)),
        train_end=str(pd.to_datetime(train_end)),
        metrics=metrics,
        random_state=getattr(model_cfg, "random_state", None),
        seed=seed,
        python_version=sys.version,
        sklearn_version=sklearn.__version__,
        pandas_version=pd.__version__,
        model_path=model_path,
        metrics_path=metrics_path,
        meta_path=meta_path,
        feature_importances=feature_importances,
        feature_schema_hash=feature_schema_hash,
        config_snapshot=config_snapshot,
    )

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, ensure_ascii=False, indent=2)

    return meta


def load_sklearn_model(model_dir: str) -> Tuple[Any, ModelMetadata]:
    meta_path = os.path.join(model_dir, "meta.json")
    model_path = os.path.join(model_dir, "model.joblib")

    if not os.path.exists(meta_path) or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifacts missing under {model_dir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_dict = json.load(f)
    meta = ModelMetadata(**meta_dict)

    model = joblib.load(model_path)
    return model, meta
