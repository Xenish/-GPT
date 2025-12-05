from __future__ import annotations

from copy import deepcopy

import pytest

from finantradealgo.system.config_loader import load_config
from finantradealgo.system.config_validation import ConfigValidationError, validate_config


def test_research_requires_data_fields():
    cfg = load_config("research")

    cfg_missing_symbols = deepcopy(cfg)
    cfg_missing_symbols["data"]["symbols"] = []
    with pytest.raises(ConfigValidationError):
        validate_config(cfg_missing_symbols)

    cfg_missing_timeframes = deepcopy(cfg)
    cfg_missing_timeframes["data"]["timeframes"] = []
    with pytest.raises(ConfigValidationError):
        validate_config(cfg_missing_timeframes)

    cfg_missing_lookback = deepcopy(cfg)
    cfg_missing_lookback["data"]["lookback_days"] = {}
    with pytest.raises(ConfigValidationError):
        validate_config(cfg_missing_lookback)


def test_live_requires_data_fields_and_max_daily_loss():
    cfg = load_config("live")

    cfg_missing_symbols = deepcopy(cfg)
    cfg_missing_symbols["data"]["symbols"] = []
    with pytest.raises(ConfigValidationError):
        validate_config(cfg_missing_symbols)

    cfg_missing_timeframes = deepcopy(cfg)
    cfg_missing_timeframes["data"]["timeframes"] = []
    with pytest.raises(ConfigValidationError):
        validate_config(cfg_missing_timeframes)

    cfg_missing_lookback = deepcopy(cfg)
    cfg_missing_lookback["data"]["lookback_days"] = {}
    with pytest.raises(ConfigValidationError):
        validate_config(cfg_missing_lookback)

    cfg_no_daily_loss = deepcopy(cfg)
    cfg_no_daily_loss["live_cfg"].max_daily_loss = 0
    with pytest.raises(ConfigValidationError):
        validate_config(cfg_no_daily_loss)
