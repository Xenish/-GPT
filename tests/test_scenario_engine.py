from __future__ import annotations

import pandas as pd

from finantradealgo.backtester.scenario_engine import (
    Scenario,
    ScenarioConfig,
    ScenarioEngine,
    run_scenarios,
)
from finantradealgo.system.config_loader import load_config


def _build_dummy_df() -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=20, freq="15min")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100 + i for i in range(len(ts))],
            "high": [101 + i for i in range(len(ts))],
            "low": [99 + i for i in range(len(ts))],
            "close": [100 + i for i in range(len(ts))],
            "rule_long_entry": [1 if i == 2 else 0 for i in range(len(ts))],
            "rule_long_exit": [1 if i == 10 else 0 for i in range(len(ts))],
            "atr_14": [1.0] * len(ts),
            "atr_14_pct": [0.01] * len(ts),
            "hv_20": [0.02] * len(ts),
            "bb_width_20": [0.02] * len(ts),
            "ms_trend_score": [0.0] * len(ts),
            "ema_20": [100 + i for i in range(len(ts))],
            "ema_50": [99 + i for i in range(len(ts))],
            "rsi_14": [55] * len(ts),
            "htf1h_trend_score": [0.2] * len(ts),
            "ms_swing_high": [105] * len(ts),
            "ms_swing_low": [95] * len(ts),
            "ms_fvg_flag": [1] * len(ts),
        }
    )


def test_run_scenarios_returns_dataframe():
    cfg = load_config("research")
    df = _build_dummy_df()
    engine = ScenarioEngine(cfg)

    scenarios = [
        ScenarioConfig(
            name="rule_base",
            strategy_name="rule",
            strategy_params={"use_ms_trend_filter": False},
        ),
        ScenarioConfig(
            name="trend_follow",
            strategy_name="trend_continuation",
            strategy_params={"warmup_bars": 0},
        ),
    ]

    result = engine.run_scenarios(scenarios, df)
    assert isinstance(result, pd.DataFrame)
    assert set(result["scenario_name"]) == {"rule_base", "trend_follow"}
    assert (result["trade_count"] >= 0).all()
    assert pd.api.types.is_numeric_dtype(result["cum_return"])


def test_run_scenarios_produces_rows():
    cfg = load_config("research")
    scenarios = [
        Scenario(
            symbol="AIAUSDT",
            timeframe="15m",
            strategy="rule",
            params={"tp_atr_mult": 2.0},
        ),
        Scenario(
            symbol="AIAUSDT",
            timeframe="15m",
            strategy="rule",
            params={"tp_atr_mult": 3.0},
        ),
    ]

    df = run_scenarios(cfg, scenarios)
    assert len(df) == 2
    assert {"scenario_id", "symbol", "timeframe", "strategy"}.issubset(df.columns)
    assert df["scenario_id"].nunique() == 2
    assert (df["trade_count"] >= 0).all()
