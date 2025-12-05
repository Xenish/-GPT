from __future__ import annotations

from pathlib import Path

import pandas as pd

import scripts.run_test_risk_overlays as script


def test_run_test_risk_overlays_outputs_csv(tmp_path, monkeypatch):
    # Dummy report and meta to avoid heavy backtests
    dummy_report = {
        "equity_metrics": {"final_equity": 1050.0, "cum_return": 0.05, "max_drawdown": -0.02},
        "trade_stats": {"trade_count": 10, "win_rate": 0.6},
        "risk_stats": {"blocked_entries": {"2024-01-01": 2}},
    }
    dummy_meta = {
        "symbol": "TESTUSDT",
        "timeframe": "15m",
        "feature_preset": "test",
        "pipeline_version": "vtest",
    }

    def fake_run_rule_backtest(sys_cfg, symbol, timeframe):
        return dummy_report, None, dummy_meta

    def fake_load_config(profile):
        return {"profile": "research", "symbol": "TESTUSDT", "timeframe": "15m"}

    monkeypatch.setattr(script, "run_rule_backtest", fake_run_rule_backtest)
    monkeypatch.setattr(script, "load_config", fake_load_config)

    out_dir = tmp_path / "outputs" / "backtests"
    monkeypatch.chdir(tmp_path)

    script.main(symbol="TESTUSDT", timeframe="15m")

    files = list(out_dir.glob("risk_overlay_sweep_TESTUSDT_15m.csv"))
    assert files, "Expected risk overlay sweep output CSV"
    df = pd.read_csv(files[0])
    required_cols = [
        "max_daily_loss_pct",
        "final_equity",
        "cum_return",
        "max_drawdown",
        "trade_count",
        "win_rate",
        "blocked_entries",
    ]
    for col in required_cols:
        assert col in df.columns
