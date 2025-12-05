from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finantradealgo.backtester.runners import _inject_ml_proba_from_registry
from finantradealgo.ml.model import ModelMetadata, SklearnLongModel, SklearnModelConfig, compute_feature_schema_hash


class DummyRegistryEntry:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.symbol = "TESTUSDT"
        self.timeframe = "15m"
        self.model_type = "rf"
        self.created_at = "2025-01-01T00:00:00"
        self.path = ""
        self.status = "success"
        self.cum_return = None
        self.sharpe = None


def test_feature_schema_guard(monkeypatch, tmp_path):
    # Prepare simple model and metadata with schema hash
    X = np.random.randn(20, 3)
    y = (X[:, 0] > 0).astype(int)
    model_cfg = SklearnModelConfig(model_type="random_forest", params={"n_estimators": 5}, random_state=7)
    model = SklearnLongModel(model_cfg)
    model.fit(X, y)

    feature_cols = ["f1", "f2", "f3"]
    meta = ModelMetadata(
        model_id="dummy",
        symbol="TESTUSDT",
        timeframe="15m",
        model_type="random_forest",
        created_at="2025-01-01T00:00:00",
        feature_preset="test",
        feature_cols=feature_cols,
        label_config={},
        train_start="2025-01-01",
        train_end="2025-01-02",
        metrics={},
        model_path="",
        metrics_path="",
        meta_path="",
        pipeline_version="v1",
        feature_importances=None,
        random_state=model_cfg.random_state,
        python_version="",
        sklearn_version="",
        pandas_version="",
        feature_schema_hash=compute_feature_schema_hash(feature_cols),
        seed=model_cfg.random_state,
        config_snapshot=None,
    )

    # Monkeypatch registry lookups
    monkeypatch.setattr(
        "finantradealgo.backtester.runners.get_latest_model",
        lambda base_dir, symbol, timeframe, model_type=None: DummyRegistryEntry("dummy"),
    )
    monkeypatch.setattr(
        "finantradealgo.backtester.runners.load_model_by_id",
        lambda base_dir, model_id: (model, meta),
    )

    df_features = pd.DataFrame(
        {"f1": np.random.randn(10), "f2": np.random.randn(10), "f3": np.random.randn(10)}
    )

    # Should work with matching schema
    out = _inject_ml_proba_from_registry(
        df_features=df_features,
        cfg={"ml": {"persistence": {"model_dir": str(tmp_path)}}},
        symbol="TESTUSDT",
        timeframe="15m",
    )
    assert "ml_long_proba" in out.columns

    # Remove a column to force schema mismatch
    df_bad = df_features.drop(columns=["f3"])
    with pytest.raises(ValueError):
        _inject_ml_proba_from_registry(
            df_features=df_bad,
            cfg={"ml": {"persistence": {"model_dir": str(tmp_path)}}},
            symbol="TESTUSDT",
            timeframe="15m",
        )
