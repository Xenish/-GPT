from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def test_run_ml_backtest_cli_rc1(monkeypatch, tmp_path):
    """
    Smoke-test ML backtest CLI wiring with research profile and standard proba column.
    """
    import scripts.run_ml_backtest as cli
    from finantradealgo.system.config_loader import MLConfig

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
                "model": {"type": "logreg", "random_state": 11},
                "backtest": {"warmup_bars": 0, "proba_column": ml_cfg.proba_column},
                "persistence": {
                    "save_model": False,
                    "use_registry": False,
                    "model_dir": str(tmp_path / "models"),
                },
                "registry": {},
            },
            "risk": {},
        }

    def fake_pipeline(cfg, symbol=None, timeframe=None):
        ts = pd.date_range("2024-01-01", periods=30, freq="15min")
        close = 100 + np.sin(np.linspace(0, 3 * np.pi, 30))
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "close": close,
                "f1": np.linspace(0, 1, 30),
                "f2": np.linspace(1, 0, 30),
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

    def fake_generate_report(result, df, config):
        equity_curve = pd.Series(np.linspace(1000, 1010, len(df)))
        trades = pd.DataFrame({"pnl": [1.0], "timestamp": [df["timestamp"].iloc[0]]})
        return {
            "equity_metrics": {
                "initial_cash": 1000.0,
                "final_equity": 1010.0,
                "cum_return": 0.01,
                "max_drawdown": 0.001,
                "sharpe": 1.0,
            },
            "trade_stats": {
                "trade_count": len(trades),
                "win_rate": 1.0,
                "avg_pnl": 1.0,
                "avg_win": 1.0,
                "avg_loss": 0.0,
                "profit_factor": 1.0,
                "median_hold_time": 1.0,
            },
            "risk_stats": {},
            "equity_curve": equity_curve,
            "trades": trades,
        }

    def fake_dirs():
        bt_dir = tmp_path / "backtests"
        tr_dir = tmp_path / "trades"
        bt_dir.mkdir(parents=True, exist_ok=True)
        tr_dir.mkdir(parents=True, exist_ok=True)
        return bt_dir, tr_dir

    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli, "build_feature_pipeline_from_system_config", fake_pipeline)
    monkeypatch.setattr(cli, "generate_report", fake_generate_report)
    monkeypatch.setattr(cli, "_ensure_output_dirs", fake_dirs)
    monkeypatch.setattr(sys, "argv", ["run_ml_backtest.py", "--symbol", "TESTUSDT"])

    cli.main()

    bt_files = list((tmp_path / "backtests").glob("*.csv"))
    trade_files = list((tmp_path / "trades").glob("*.csv"))
    assert bt_files and trade_files, "Backtest outputs should be written to temp directories"
