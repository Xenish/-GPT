from __future__ import annotations

from pathlib import Path

from finantradealgo.ml.model import ModelMetadata
from finantradealgo.ml.model_registry import (
    get_latest_model,
    list_models,
    register_model,
)


def _make_meta(base_dir: Path, model_id: str, created_at: str, symbol: str = "BTCUSDT") -> ModelMetadata:
    run_dir = base_dir / model_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model.joblib").write_text("stub")
    (run_dir / "metrics.csv").write_text("metric,value")
    (run_dir / "meta.json").write_text("{}")
    return ModelMetadata(
        model_id=model_id,
        symbol=symbol,
        timeframe="15m",
        model_type="RandomForest",
        created_at=created_at,
        feature_preset="extended",
        feature_cols=["foo"],
        label_config={},
        train_start="2025-01-01",
        train_end="2025-01-02",
        metrics={"cum_return": 0.1, "sharpe": 0.2},
        model_path=str(run_dir / "model.joblib"),
        metrics_path=str(run_dir / "metrics.csv"),
        meta_path=str(run_dir / "meta.json"),
        pipeline_version="v1.0.0",
        python_version="py",
        sklearn_version="sk",
        pandas_version="pd",
    )


def test_register_and_list_models(tmp_path):
    base_dir = tmp_path / "registry"
    meta1 = _make_meta(base_dir, "model_a", "2025-01-01T00:00:00")
    meta2 = _make_meta(base_dir, "model_b", "2025-01-02T00:00:00")

    register_model(meta1, str(base_dir), status="success", max_models=5)
    register_model(meta2, str(base_dir), status="failed", max_models=5)

    entries = list_models(str(base_dir), symbol="BTCUSDT", timeframe="15m")
    assert len(entries) == 2

    latest = get_latest_model(str(base_dir), symbol="BTCUSDT", timeframe="15m", model_type="RandomForest")
    assert latest is not None
    assert latest.model_id == "model_a"


def test_auto_prune_old_models(tmp_path):
    base_dir = tmp_path / "registry"
    meta_ids = [
        ("model_old", "2025-01-01T00:00:00"),
        ("model_mid", "2025-01-02T00:00:00"),
        ("model_new", "2025-01-03T00:00:00"),
    ]
    for model_id, created_at in meta_ids:
        meta = _make_meta(base_dir, model_id, created_at)
        register_model(meta, str(base_dir), status="success", max_models=2)

    entries = list_models(str(base_dir), symbol="BTCUSDT", timeframe="15m")
    ids = {entry.model_id for entry in entries}
    assert "model_old" not in ids
    assert (base_dir / "model_old").exists() is False
