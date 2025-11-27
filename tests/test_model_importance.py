from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from finantradealgo.ml.labels import LabelConfig
from finantradealgo.ml.model import save_sklearn_model, SklearnModelConfig


def test_model_importance_saved_and_loaded(tmp_path):
    X, y = make_classification(
        n_samples=120,
        n_features=4,
        n_informative=2,
        random_state=42,
    )
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X, y)

    feature_cols = [f"f{i}" for i in range(X.shape[1])]
    base_dir = tmp_path / "ml_models"
    meta = save_sklearn_model(
        model=rf,
        symbol="AIAUSDT",
        timeframe="15m",
        model_cfg=SklearnModelConfig(model_type="random_forest", random_state=42),
        label_cfg=LabelConfig(),
        feature_preset="extended",
        feature_cols=feature_cols,
        train_start=pd.Timestamp("2024-01-01"),
        train_end=pd.Timestamp("2024-01-02"),
        metrics={"accuracy": 0.9},
        base_dir=str(base_dir),
        pipeline_version="test",
    )

    assert meta.feature_importances is not None
    assert set(meta.feature_importances.keys()) == set(feature_cols)
    total = sum(meta.feature_importances.values())
    assert abs(total - 1.0) < 1e-6

    meta_path = Path(meta.meta_path)
    assert meta_path.is_file()
    with meta_path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert "feature_importances" in loaded
    assert set(loaded["feature_importances"].keys()) == set(feature_cols)
