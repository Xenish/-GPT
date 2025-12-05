from __future__ import annotations

from copy import deepcopy

import pytest

from finantradealgo.system.config_loader import load_config
from finantradealgo.system.config_validation import (
    ConfigValidationError,
    validate_config,
)


def test_validate_config_research_passes():
    cfg = load_config("research")
    validate_config(cfg)


def test_validate_config_live_passes():
    cfg = load_config("live")
    validate_config(cfg)


def test_research_rejects_live_exchange():
    cfg = load_config("research")
    cfg_bad = deepcopy(cfg)
    cfg_bad["exchange"]["type"] = "live"
    with pytest.raises(ConfigValidationError):
        validate_config(cfg_bad)


def test_live_requires_kill_switch_and_single_symbol():
    cfg = load_config("live")

    # Missing kill_switch should fail
    cfg_no_kill = deepcopy(cfg)
    cfg_no_kill.pop("kill_switch", None)
    with pytest.raises(ConfigValidationError):
        validate_config(cfg_no_kill)

    # Multiple live symbols should fail
    cfg_multi_symbols = deepcopy(cfg)
    cfg_multi_symbols["live_cfg"].symbols = ["AIAUSDT", "BTCUSDT"]
    with pytest.raises(ConfigValidationError):
        validate_config(cfg_multi_symbols)
