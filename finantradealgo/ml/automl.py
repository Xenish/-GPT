from __future__ import annotations

"""Minimal Optuna-based AutoML scaffolding for model selection and tuning.

Design notes
------------
- Separates search space definition (`suggest_model_and_params`) from the runner.
- `AutoMLRunner` depends only on data arrays and an `eval_fn` supplied by the caller.
- Future hooks: blend/stack top trials, feature search, neural models, etc.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict

import numpy as np
import optuna

from finantradealgo.ml import EnsembleConfig, ModelSpec, TaskType

# Optional imports for ensemble integration in future iterations
try:  # pragma: no cover - optional dependency use
    from finantradealgo.ml.ensemble_models import BlendedEnsemble
except Exception:  # pragma: no cover
    BlendedEnsemble = None  # type: ignore

try:  # pragma: no cover - optional dependency use
    from finantradealgo.ml.stacking import StackingEnsemble, StackingConfig
except Exception:  # pragma: no cover
    StackingEnsemble = None  # type: ignore
    StackingConfig = None  # type: ignore


@dataclass
class AutoMLConfig:
    task_type: TaskType
    max_trials: int = 50
    timeout_seconds: int | None = None
    metric_name: str = "sharpe"
    direction: str = "maximize"  # or "minimize"
    enable_blending: bool = False
    enable_stacking: bool = False
    random_state: int | None = 42
    metadata: dict[str, Any] | None = None


@dataclass
class AutoMLResult:
    best_score: float
    best_params: dict[str, Any]
    best_model_spec: ModelSpec | None
    best_ensemble_config: EnsembleConfig | None
    best_estimator: Any | None
    study: optuna.Study | None = None


def _build_model_spec_from_params(model_name: str, params: Dict[str, Any], task_type: TaskType) -> ModelSpec:
    """Reconstruct a ModelSpec given a chosen model_name and parameter dict (excluding model_name)."""
    if model_name == "random_forest":
        def factory():
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

            if task_type == TaskType.REGRESSION:
                return RandomForestRegressor()
            return RandomForestClassifier()

        return ModelSpec(name="random_forest", task_type=task_type, estimator_factory=factory, params=params)

    if model_name == "logistic_regression":
        def factory():
            if task_type == TaskType.REGRESSION:
                from sklearn.linear_model import Ridge

                return Ridge()
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression()

        return ModelSpec(name="logistic_regression", task_type=task_type, estimator_factory=factory, params=params)

    if model_name == "xgboost":
        def factory():
            try:
                from xgboost import XGBRegressor, XGBClassifier
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("xgboost is not installed") from exc

            if task_type == TaskType.REGRESSION:
                return XGBRegressor(objective="reg:squarederror", eval_metric="rmse", verbosity=0)
            return XGBClassifier(objective="binary:logistic", eval_metric="logloss", use_label_encoder=False, verbosity=0)

        return ModelSpec(name="xgboost", task_type=task_type, estimator_factory=factory, params=params)

    if model_name == "lightgbm":
        def factory():
            try:
                from lightgbm import LGBMRegressor, LGBMClassifier
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("lightgbm is not installed") from exc

            if task_type == TaskType.REGRESSION:
                return LGBMRegressor()
            return LGBMClassifier()

        return ModelSpec(name="lightgbm", task_type=task_type, estimator_factory=factory, params=params)

    def fallback_factory():
        from sklearn.linear_model import LinearRegression

        return LinearRegression()

    return ModelSpec(name="linear_fallback", task_type=task_type, estimator_factory=fallback_factory, params=params)


def suggest_model_and_params(
    trial: optuna.Trial,
    task_type: TaskType,
) -> ModelSpec:
    """
    Suggest a model type (e.g., RF, XGBoost, LightGBM, linear) and tuned hyperparameters.
    Returns a ModelSpec with estimator_factory and params filled.
    """
    model_name = trial.suggest_categorical(
        "model_name", ["random_forest", "logistic_regression", "xgboost", "lightgbm"]
    )

    params: Dict[str, Any] = {}

    if model_name == "random_forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 12, step=2, log=False),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "n_jobs": -1,
        }
    elif model_name == "logistic_regression":
        if task_type == TaskType.REGRESSION:
            params = {
                "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
                "max_iter": trial.suggest_int("max_iter", 100, 500, step=50),
            }
        else:
            params = {
                "C": trial.suggest_float("C", 0.01, 10.0, log=True),
                "max_iter": trial.suggest_int("max_iter", 100, 500, step=50),
            }
    elif model_name == "xgboost":
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "n_jobs": -1,
        }
    elif model_name == "lightgbm":
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 700, step=50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "n_jobs": -1,
        }

    return _build_model_spec_from_params(model_name, params, task_type)


class AutoMLRunner:
    """
    Orchestrates an Optuna study. Caller supplies eval_fn(model, X_val, y_val) -> float.
    This stays decoupled from any backtesting engine.
    """

    def __init__(self, config: AutoMLConfig, eval_fn: Callable[[Any, np.ndarray, np.ndarray], float]) -> None:
        self.config = config
        self.eval_fn = eval_fn

    def _apply_params(self, estimator: Any, params: Dict[str, Any]) -> Any:
        if not params:
            return estimator
        if hasattr(estimator, "set_params"):
            return estimator.set_params(**params)
        for key, value in params.items():
            setattr(estimator, key, value)
        return estimator

    def run(self, X_train, y_train, X_val, y_val) -> AutoMLResult:
        sampler = optuna.samplers.TPESampler(seed=self.config.random_state)
        study = optuna.create_study(direction=self.config.direction, sampler=sampler)

        def objective(trial: optuna.Trial) -> float:
            model_spec = suggest_model_and_params(trial, self.config.task_type)

            estimator = model_spec.estimator_factory()
            estimator = self._apply_params(estimator, model_spec.params or {})
            estimator.fit(X_train, y_train)

            score = self.eval_fn(estimator, X_val, y_val)
            trial.set_user_attr(
                "model_info",
                {"model_name": model_spec.name, "params": model_spec.params or {}},
            )
            return float(score)

        study.optimize(objective, n_trials=self.config.max_trials, timeout=self.config.timeout_seconds)

        best_trial = study.best_trial
        best_score = float(best_trial.value)
        best_params = dict(best_trial.params)
        model_info = best_trial.user_attrs.get("model_info", {})
        model_name = model_info.get("model_name", best_params.get("model_name", "random_forest"))
        model_params = {k: v for k, v in best_params.items() if k != "model_name"}
        # Prefer stored params if present (they omit model_name already)
        if "params" in model_info:
            model_params = model_info["params"]
        best_spec = _build_model_spec_from_params(model_name, model_params, self.config.task_type)

        # Refit best model on combined train+val for final materialization
        combined_X = np.concatenate([X_train, X_val], axis=0)
        combined_y = np.concatenate([y_train, y_val], axis=0)

        best_estimator = None
        if best_spec is not None:
            best_estimator = best_spec.estimator_factory()
            best_estimator = self._apply_params(best_estimator, best_spec.params or {})
            best_estimator.fit(combined_X, combined_y)

        result = AutoMLResult(
            best_score=best_score,
            best_params=best_params,
            best_model_spec=best_spec,
            best_ensemble_config=None,  # Future: assemble top trials into a blended/stacked ensemble
            best_estimator=best_estimator,
            study=study,
        )
        return result

# Future hooks:
# - Combine top-N trials into a BlendedEnsemble using EnsembleConfig/ModelSpec.
# - Explore stacking by wrapping base trials into StackingConfig.
# - Feature engineering / column subset search.
# - Neural models / NAS integration.
