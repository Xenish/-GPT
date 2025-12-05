import numpy as np
import pandas as pd

from finantradealgo.ml.model import SklearnModelConfig, SklearnLongModel


def test_sklearn_long_model_fit_predict_proba():
    # Dummy binary classification data
    X = pd.DataFrame({"x1": [0, 1, 0, 1], "x2": [1, 0, 1, 0]})
    y = np.array([0, 1, 0, 1])

    cfg = SklearnModelConfig(model_type="random_forest", random_state=123, params={"n_estimators": 10})
    model = SklearnLongModel(cfg)

    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape[0] == X.shape[0]
    assert proba.shape[1] >= 2
    assert proba.min() >= 0.0 and proba.max() <= 1.0


def test_sklearn_model_metadata_generation(tmp_path):
    X = pd.DataFrame({"x": [0, 1, 0, 1]})
    y = np.array([0, 1, 0, 1])
    cfg = SklearnModelConfig(model_type="random_forest", random_state=123, params={"n_estimators": 5})
    model = SklearnLongModel(cfg)
    model.fit(X, y)
    meta = model.evaluate(X, y)  # metrics dict
    assert "accuracy" in meta
