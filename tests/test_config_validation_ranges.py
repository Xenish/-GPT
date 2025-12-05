from __future__ import annotations

from copy import deepcopy

import pytest

from finantradealgo.system.config_loader import load_config
from finantradealgo.system.config_validation import ConfigValidationError, validate_config


def test_valid_configs_pass():
    validate_config(load_config("research"))
    validate_config(load_config("live"))


@pytest.mark.parametrize(
    "field, value",
    [
        ("risk_limits_cfg.risk_per_trade_pct", 2.0),
        ("risk_limits_cfg.leverage_ceiling", 500.0),
        ("exchange_risk_cfg.max_leverage", 1000.0),
    ],
)
def test_research_numeric_out_of_range(field, value):
    cfg = load_config("research")
    cfg_mut = deepcopy(cfg)
    target = cfg_mut
    # simple path resolver
    for part in field.split(".")[:-1]:
        target = target[part]
    last = field.split(".")[-1]
    if hasattr(target, last):
        setattr(target, last, value)
    else:
        target[last] = value
    with pytest.raises(ConfigValidationError):
        validate_config(cfg_mut)


def test_live_numeric_out_of_range():
    cfg = load_config("live")
    cfg_bad = deepcopy(cfg)
    cfg_bad["live_cfg"].max_position_notional = -1
    with pytest.raises(ConfigValidationError):
        validate_config(cfg_bad)

    cfg_bad2 = deepcopy(cfg)
    cfg_bad2["kill_switch_cfg"].max_equity_drawdown_pct = 150
    with pytest.raises(ConfigValidationError):
        validate_config(cfg_bad2)

    cfg_bad3 = deepcopy(cfg)
    cfg_bad3["kill_switch_cfg"].daily_realized_pnl_limit = 10  # should be negative loss threshold
    with pytest.raises(ConfigValidationError):
        validate_config(cfg_bad3)
