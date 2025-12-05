import json
from pathlib import Path

import numpy as np
import pandas as pd

from finantradealgo.ml.model import SklearnLongModel, SklearnModelConfig, save_sklearn_model
from finantradealgo.ml.model_registry import (
    get_latest_model,
    list_models,
    load_model_by_id,
    register_model,
)
from finantradealgo.ml.labels import LabelConfig


def test_model_registry_roundtrip(tmp_path: Path):
    # Train a tiny model
    X = pd.DataFrame({"x": [0, 1, 0, 1]})
    y = np.array([0, 1, 0, 1])
    cfg = SklearnModelConfig(model_type="random_forest", random_state=123, params={"n_estimators": 5})
    model = SklearnLongModel(cfg)
    model.fit(X, y)
    meta = save_sklearn_model(
        model=model.clf,
        symbol="BTCUSDT",
        timeframe="15m",
        model_cfg=cfg,
        label_cfg=LabelConfig(),
        feature_preset="test",
        feature_cols=list(X.columns),
        train_start=pd.Timestamp("2025-01-01"),
        train_end=pd.Timestamp("2025-01-02"),
        metrics={"accuracy": 1.0},
        base_dir=str(tmp_path),
    )

    register_model(meta, base_dir=str(tmp_path), status="success", max_models=3)

    entries = list_models(str(tmp_path), symbol="BTCUSDT", timeframe="15m")
    assert len(entries) == 1
    latest = get_latest_model(str(tmp_path), symbol="BTCUSDT", timeframe="15m")
    assert latest is not None
    assert latest.model_id == meta.model_id

    loaded_model, loaded_meta = load_model_by_id(str(tmp_path), meta.model_id)
    assert loaded_meta.model_id == meta.model_id
