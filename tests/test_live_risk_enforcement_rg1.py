from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finantradealgo.live_trading.factories import create_live_engine
from finantradealgo.live_trading.live_engine import LiveEngine
from finantradealgo.system.config_loader import load_config, RiskLimitsConfig
from finantradealgo.risk.risk_engine import RiskConfig


@pytest.mark.integration
def test_live_risk_enforcement_blocks_large_position(monkeypatch):
    cfg = load_config("research")
    cfg_local = dict(cfg)
    cfg_local["strategy"] = {"default": "rule"}
    cfg_local["live"] = {
        "mode": "replay",
        "data_source": "replay",
        "symbol": cfg.get("symbol", "BTCUSDT"),
        "symbols": [cfg.get("symbol", "BTCUSDT")],
        "timeframe": cfg.get("timeframe", "15m"),
        "replay": {"bars_limit": 5},
    }
    # Force tiny notional limit
    cfg_local["risk"] = {"max_notional_per_symbol": 1.0, "capital_risk_pct_per_trade": 0.5}

    ts = pd.date_range("2024-01-01", periods=5, freq="15min")
    close = np.linspace(100, 101, len(ts))
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "rule_long_entry": [1, 0, 0, 0, 0],
            "rule_long_exit": [0, 0, 0, 0, 0],
        }
    )

    # Patch feature pipeline to supply dummy rule signals
    def fake_pipeline(sys_cfg, symbol=None, timeframe=None):
        return df.copy(), {"symbol": symbol, "timeframe": timeframe, "feature_cols": []}

    monkeypatch.setattr(
        "finantradealgo.live_trading.factories.build_feature_pipeline_from_system_config",
        fake_pipeline,
    )

    engine, strat_name = create_live_engine(cfg_local, run_id="risk_test")
    assert isinstance(engine, LiveEngine)
    assert strat_name == "rule"

    engine.run(max_iterations=5)
    portfolio = engine.execution_client.get_portfolio()
    position = engine.execution_client.get_open_positions()
    has_position = position and position[0].get("qty", 0) > 0
    if has_position:
        assert position[0]["qty"] * position[0]["entry_price"] <= 1.5
