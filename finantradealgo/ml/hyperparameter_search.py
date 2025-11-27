from __future__ import annotations

import itertools
from typing import Any, Dict, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


def run_rf_grid_search(
    X,
    y,
    param_grid: Dict[str, List[Any]],
    n_splits: int = 3,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    """
    Perform a basic RandomForest grid search with stratified K-fold cross validation.
    """

    X = np.asarray(X)
    y = np.asarray(y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    keys = list(param_grid.keys())
    results: List[Dict[str, Any]] = []

    for values in itertools.product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, values))

        clf_kwargs = {
            "n_estimators": 100,
            "random_state": random_state,
            "n_jobs": -1,
        }
        clf_kwargs.update(params)
        clf = RandomForestClassifier(**clf_kwargs)

        acc_scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
        f1_scores = cross_val_score(clf, X, y, cv=skf, scoring="f1", n_jobs=-1)

        results.append(
            {
                "params": params,
                "mean_accuracy": float(acc_scores.mean()),
                "std_accuracy": float(acc_scores.std()),
                "mean_f1": float(f1_scores.mean()),
                "std_f1": float(f1_scores.std()),
                "n_splits": n_splits,
            }
        )

    return results


__all__ = ["run_rf_grid_search"]
