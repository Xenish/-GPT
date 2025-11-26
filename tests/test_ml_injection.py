from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from finantradealgo.backtester.runners import _inject_ml_proba_from_registry
from finantradealgo.ml.model import ModelMetadata
from finantradealgo.ml.model_registry import ModelRegistryEntry, register_model


def _prepare_model(tmp_path: Path) -> tuple[Path, ModelRegistryEntry]:
    registry_dir = tmp_path / "registry"
    model_id = "dummy_model"
    model_dir = registry_dir / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    df_features = pd.DataFrame(
        {
            "feat1": np.random.randn(20),
            "feat2": np.random.randn(20),
            "feat3": np.random.randn(20),
        }
    )
    X = df_features[["feat1", "feat2", "feat3"]]
    y = (X["feat1"] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    model_path = model_dir / "model.joblib"
    metrics_path = model_dir / "metrics.csv"
    meta_path = model_dir / "meta.json"

    import joblib

    joblib.dump(model, model_path)
    metrics_path.write_text("metric,value\nroc_auc,0.5\n", encoding="utf-8")

    meta = ModelMetadata(
        model_id=model_id,
        symbol="TESTUSDT",
        timeframe="15m",
        model_type="RandomForest",
        created_at="2025-01-01T00:00:00",
        feature_preset="extended",
        feature_cols=["feat1", "feat2", "feat3"],
        label_config={},
        train_start="2025-01-01",
        train_end="2025-01-02",
        metrics={"roc_auc": 0.5},
        model_path=str(model_path),
        metrics_path=str(metrics_path),
        meta_path=str(meta_path),
        pipeline_version="v1",
        python_version="py",
        sklearn_version="sk",
        pandas_version="pd",
        random_state=42,
    )
    meta_path.write_text(json.dumps(asdict(meta)), encoding="utf-8")

    register_model(meta, str(registry_dir), status="success", max_models=5)
    entry = ModelRegistryEntry(
        model_id=model_id,
        symbol="TESTUSDT",
        timeframe="15m",
        model_type="RandomForest",
        created_at=meta.created_at,
        path=str(model_dir),
        status="success",
    )
    return registry_dir, entry


def _base_cfg(model_dir: Path) -> dict:
    return {
        "ml": {
            "model": {"type": "RandomForest"},
            "backtest": {"proba_column": "ml_proba_long"},
            "persistence": {"model_dir": str(model_dir)},
        }
    }


def test_inject_ml_proba_happy_path(tmp_path):
    model_dir, _entry = _prepare_model(tmp_path)
    df_features = pd.DataFrame(
        {
            "feat1": np.random.randn(20),
            "feat2": np.random.randn(20),
            "feat3": np.random.randn(20),
        }
    )

    out = _inject_ml_proba_from_registry(
        df_features=df_features,
        cfg=_base_cfg(model_dir),
        symbol="TESTUSDT",
        timeframe="15m",
    )

    assert "ml_proba_long" in out.columns
    assert not out["ml_proba_long"].isna().all()


def test_inject_ml_proba_missing_feature_raises(tmp_path):
    model_dir, _entry = _prepare_model(tmp_path)
    df_features = pd.DataFrame(
        {
            "feat1": np.random.randn(20),
            "feat2": np.random.randn(20),
        }
    )

    with pytest.raises(ValueError) as excinfo:
        _inject_ml_proba_from_registry(
            df_features=df_features,
            cfg=_base_cfg(model_dir),
            symbol="TESTUSDT",
            timeframe="15m",
        )

    assert "Feature mismatch" in str(excinfo.value)
