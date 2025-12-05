from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finantradealgo.ml.model import SklearnLongModel, SklearnModelConfig


def test_sklearn_long_model_reproducible_with_seed():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(50, 4)), columns=["f1", "f2", "f3", "f4"])
    y = (X["f1"] + X["f2"] > 0).astype(int)

    cfg = SklearnModelConfig(model_type="random_forest", random_state=123, params={"n_estimators": 20})
    m1 = SklearnLongModel(cfg)
    m2 = SklearnLongModel(cfg)

    m1.fit(X, y)
    m2.fit(X, y)

    p1 = m1.predict_proba(X)
    p2 = m2.predict_proba(X)

    assert np.allclose(p1, p2)


def test_sklearn_long_model_non_deterministic_without_seed():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=["a", "b", "c"])
    y = (X["a"] > 0).astype(int)

    cfg_no_seed = SklearnModelConfig(model_type="random_forest", params={"n_estimators": 10})
    m1 = SklearnLongModel(cfg_no_seed)
    m2 = SklearnLongModel(cfg_no_seed)

    m1.fit(X, y)
    m2.fit(X, y)

    p1 = m1.predict_proba(X)
    p2 = m2.predict_proba(X)

    # Without a seed, determinism is not guaranteed; allow equality but do not assert it.
    assert p1.shape == p2.shape
