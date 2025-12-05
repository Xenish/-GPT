from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finantradealgo.live_trading.factories import create_live_engine
from finantradealgo.live_trading.live_engine import LiveEngine
from finantradealgo.ml.labels import LabelConfig
from finantradealgo.ml.model import SklearnLongModel, SklearnModelConfig, save_sklearn_model
from finantradealgo.ml.model_registry import register_model
from finantradealgo.system.config_loader import LiveConfig, load_config

pytestmark = pytest.mark.integration


def test_live_engine_replay_with_ml_strategy(monkeypatch, tmp_path):
    cfg = load_config("research")
    cfg_local = dict(cfg)
    cfg_local["strategy"] = {"default": "ml"}
    cfg_local["live"] = {
        "mode": "replay",
        "data_source": "replay",
        "symbol": cfg.get("symbol", "BTCUSDT"),
        "symbols": [cfg.get("symbol", "BTCUSDT")],
        "timeframe": cfg.get("timeframe", "15m"),
        "replay": {"bars_limit": 20},
    }
    # force LiveConfig regeneration so the patched live block is respected
    cfg_local["live_cfg"] = LiveConfig.from_dict(
        cfg_local["live"],
        default_symbol=cfg_local["live"]["symbol"],
        default_timeframe=cfg_local["live"]["timeframe"],
    )

    ts = pd.date_range("2024-01-01", periods=40, freq="15min")
    close = np.linspace(100, 102, len(ts))
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": close - 0.1,
            "high": close + 0.1,
            "low": close - 0.2,
            "close": close,
            "f1": np.linspace(0, 1, len(ts)),
            "f2": np.linspace(1, 0, len(ts)),
        }
    )

    # Train tiny model
    X = df[["f1", "f2"]].to_numpy()
    y = (X[:, 0] > 0.5).astype(int)
    model_cfg = SklearnModelConfig(model_type="random_forest", params={"n_estimators": 5}, random_state=42)
    model = SklearnLongModel(model_cfg)
    model.fit(X, y)

    model_dir = tmp_path / "models"
    dummy_label_cfg = LabelConfig()
    meta = save_sklearn_model(
        model=model.clf,
        symbol=cfg_local["live"]["symbol"],
        timeframe=cfg_local["live"]["timeframe"],
        model_cfg=model_cfg,
        label_cfg=dummy_label_cfg,
        feature_preset="test",
        feature_cols=["f1", "f2"],
        train_start=pd.Timestamp(ts[0]),
        train_end=pd.Timestamp(ts[-1]),
        metrics={},
        base_dir=str(model_dir),
        pipeline_version="vtest",
        seed=model_cfg.random_state,
        config_snapshot={"symbol": cfg_local["live"]["symbol"], "timeframe": cfg_local["live"]["timeframe"]},
    )
    register_model(meta, base_dir=str(model_dir))

    cfg_local.setdefault("ml", {}).setdefault("persistence", {})["model_dir"] = str(model_dir)
    cfg_local.setdefault("ml", {}).setdefault("registry", {})["selected_id"] = meta.model_id

    # Patch feature pipeline to return our dummy frame directly (patch factories import)
    def fake_pipeline(sys_cfg, symbol=None, timeframe=None):
        return df.copy(), {
            "symbol": symbol or cfg_local.get("symbol"),
            "timeframe": timeframe or cfg_local.get("timeframe"),
            "feature_cols": ["f1", "f2"],
            "feature_preset": "test",
            "pipeline_version": "vtest",
        }

    monkeypatch.setattr(
        "finantradealgo.live_trading.factories.build_feature_pipeline_from_system_config",
        fake_pipeline,
    )
    cfg_local["features"] = {"feature_preset": "test", "use_base": False, "use_ta": False}
    cfg_local["rule"] = {}

    engine, strat_name = create_live_engine(cfg_local, run_id="test_ml_live")
    assert isinstance(engine, LiveEngine)
    assert strat_name == "ml"

    engine.run(max_iterations=10)
    assert engine.iteration > 0
    assert not engine.kill_switch_triggered_flag
