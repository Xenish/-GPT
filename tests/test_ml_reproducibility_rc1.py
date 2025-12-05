from __future__ import annotations

import numpy as np
import pandas as pd

from finantradealgo.ml.model import SklearnLongModel, SklearnModelConfig, set_global_seed


def test_ml_reproducibility_rc1():
    # Deterministic seed should yield identical probabilities across runs
    seed = 123
    X = np.random.randn(100, 3)
    y = (X[:, 0] > 0).astype(int)

    set_global_seed(seed)
    model_cfg = SklearnModelConfig(model_type="random_forest", params={"n_estimators": 50}, random_state=seed)
    model1 = SklearnLongModel(model_cfg)
    model1.fit(X, y)
    proba1 = model1.predict_proba(X)[:, 1]

    set_global_seed(seed)
    model2 = SklearnLongModel(model_cfg)
    model2.fit(X, y)
    proba2 = model2.predict_proba(X)[:, 1]

    np.testing.assert_allclose(proba1, proba2, atol=1e-8)
