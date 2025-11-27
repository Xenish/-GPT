from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from finantradealgo.api import server as server_module
from finantradealgo.api.server import create_app


def _setup_model_dir(tmp_path: Path) -> tuple[Path, str]:
    base_dir = tmp_path / "ml_models"
    base_dir.mkdir(parents=True, exist_ok=True)
    model_id = "AIAUSDT_15m_rf_20250101"
    model_dir = base_dir / model_id
    model_dir.mkdir()
    (model_dir / "model.joblib").write_text("dummy", encoding="utf-8")
    meta = {
        "model_id": model_id,
        "symbol": "AIAUSDT",
        "timeframe": "15m",
        "model_type": "rf_classifier",
        "created_at": "2025-01-01T00:00:00",
        "feature_preset": "extended",
        "feature_cols": ["feat_a", "feat_b"],
        "label_config": {},
        "train_start": "2025-01-01",
        "train_end": "2025-01-02",
        "metrics": {"accuracy": 0.8},
        "model_path": str(model_dir / "model.joblib"),
        "metrics_path": str(model_dir / "metrics.csv"),
        "meta_path": str(model_dir / "meta.json"),
        "pipeline_version": "v1",
        "random_state": 42,
        "python_version": "3.10",
        "sklearn_version": "1.2",
        "pandas_version": "2.1",
        "feature_importances": {"feat_a": 0.7, "feat_b": 0.3},
    }
    (model_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    registry = pd.DataFrame(
        [
            {
                "model_id": model_id,
                "symbol": "AIAUSDT",
                "timeframe": "15m",
                "model_type": "rf_classifier",
                "created_at": "2025-01-01T00:00:00",
                "path": str(model_dir),
                "cum_return": 0.2,
                "sharpe": 1.1,
                "status": "success",
            }
        ]
    )
    registry.to_csv(base_dir / "registry_index.csv", index=False)
    return base_dir, model_id


def test_list_ml_models(tmp_path, monkeypatch):
    base_dir, model_id = _setup_model_dir(tmp_path)
    monkeypatch.setattr(
        server_module,
        "load_system_config",
        lambda: {"ml": {"persistence": {"model_dir": str(base_dir)}}},
    )
    client = TestClient(create_app())
    resp = client.get("/api/ml/models/AIAUSDT/15m")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    item = data[0]
    assert item["model_id"] == model_id
    assert item["metrics"]["cum_return"] == 0.2


def test_get_feature_importance_returns_meta(tmp_path, monkeypatch):
    base_dir, model_id = _setup_model_dir(tmp_path)
    monkeypatch.setattr(
        server_module,
        "load_system_config",
        lambda: {"ml": {"persistence": {"model_dir": str(base_dir)}}},
    )
    client = TestClient(create_app())
    resp = client.get(f"/api/ml/models/{model_id}/importance")
    assert resp.status_code == 200
    data = resp.json()
    assert "feat_a" in data
    assert data["feat_a"] == 0.7
