from __future__ import annotations

from sklearn.datasets import make_classification

from finantradealgo.ml.hyperparameter_search import run_rf_grid_search


def test_run_rf_grid_search_combinations():
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        random_state=123,
    )

    param_grid = {
        "n_estimators": [10, 20],
        "max_depth": [None, 3],
    }

    results = run_rf_grid_search(X, y, param_grid, n_splits=3)

    assert len(results) == 4
    for r in results:
        assert "params" in r
        assert "mean_accuracy" in r
        assert 0.0 <= r["mean_accuracy"] <= 1.0
        assert r["n_splits"] == 3
