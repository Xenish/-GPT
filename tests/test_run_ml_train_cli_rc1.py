from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def test_run_ml_train_cli_rc1(monkeypatch, tmp_path, capsys):
    """
    Smoke-test the ML training CLI with research profile and config loader wiring.
    """
    import scripts.run_ml_train as cli
    from finantradealgo.system.config_loader import MLConfig

    # Stub config to avoid touching real files/registry
    def fake_load_config(profile):
        assert profile == "research"
        ml_cfg = MLConfig.from_dict({"model_dir": str(tmp_path / "models")})
        return {
            "profile": "research",
            "symbol": "TESTUSDT",
            "timeframe": "15m",
            "features": {"feature_preset": "test"},
            "ml_cfg": ml_cfg,
            "ml": {
                "label": {"horizon": 1, "pos_threshold": 0.0, "fee_slippage": 0.0},
                "model": {"type": "logreg", "random_state": 7},
                "persistence": {
                    "save_model": True,
                    "use_registry": False,
                    "model_dir": str(tmp_path / "models"),
                },
            },
            "risk": {},
        }

    def fake_pipeline(cfg, symbol=None, timeframe=None):
        ts = pd.date_range("2024-01-01", periods=40, freq="15min")
        close = 100 + np.sin(np.linspace(0, 4 * np.pi, 40))
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "close": close,
                "f1": np.random.randn(40),
                "f2": np.random.randn(40),
            }
        )
        meta = {
            "symbol": symbol or cfg.get("symbol", "TESTUSDT"),
            "timeframe": timeframe or cfg.get("timeframe", "15m"),
            "feature_cols": ["f1", "f2"],
            "feature_preset": "test",
            "pipeline_version": "vtest",
        }
        return df, meta

    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli, "build_feature_pipeline_from_system_config", fake_pipeline)
    monkeypatch.setattr(sys, "argv", ["run_ml_train.py", "--output-dir", str(tmp_path / "models")])

    cli.main()

    # Model artifacts exist under output dir
    artifacts = list(Path(tmp_path / "models").glob("*/*"))
    assert artifacts, "Expected trained model artifacts to be written"
