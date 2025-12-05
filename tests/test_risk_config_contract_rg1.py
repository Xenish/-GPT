from __future__ import annotations

import pytest

from finantradealgo.system.config_loader import (
    RiskLimitsConfig,
    KillSwitchConfig,
    ExchangeRiskConfig,
    load_config,
)


def test_risk_config_research_and_live_types():
    cfg_research = load_config("research")
    cfg_live = load_config("live")

    for cfg in (cfg_research, cfg_live):
        risk_limits = cfg.get("risk_limits_cfg")
        kill_cfg = cfg.get("kill_switch_cfg")
        exch_risk = cfg.get("exchange_risk_cfg")
        assert isinstance(risk_limits, RiskLimitsConfig)
        assert isinstance(kill_cfg, KillSwitchConfig)
        assert isinstance(exch_risk, ExchangeRiskConfig)
        assert risk_limits.max_open_trades > 0
        assert risk_limits.risk_per_trade_pct > 0
        assert kill_cfg.evaluation_interval_bars > 0


def test_risk_limits_invalid_raises():
    with pytest.raises(ValueError):
        RiskLimitsConfig.from_dict({"risk_per_trade_pct": -0.1})
    with pytest.raises(ValueError):
        RiskLimitsConfig.from_dict({"max_open_trades": 0})


def test_kill_switch_invalid_raises():
    with pytest.raises(ValueError):
        KillSwitchConfig.from_dict({"evaluation_interval_bars": 0})
