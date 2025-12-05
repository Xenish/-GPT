from __future__ import annotations

from copy import deepcopy

import pytest

from finantradealgo.system.config_loader import load_config
from finantradealgo.system.config_validation import ConfigValidationError, validate_config


def test_live_requires_single_symbol():
    cfg = load_config("live")
    cfg_bad = deepcopy(cfg)
    cfg_bad["live_cfg"].symbols = ["AIAUSDT", "BTCUSDT"]
    with pytest.raises(ConfigValidationError):
        validate_config(cfg_bad)


def test_live_requires_api_envs():
    cfg = load_config("live")
    cfg_bad = deepcopy(cfg)
    cfg_bad["exchange"]["api_key_env"] = ""
    cfg_bad["exchange"]["secret_key_env"] = ""
    cfg_bad["exchange"]["api_key"] = ""
    cfg_bad["exchange"]["secret_key"] = ""
    assert cfg_bad["exchange"]["api_key_env"] == ""
    assert cfg_bad["exchange"]["secret_key_env"] == ""
    assert cfg_bad["exchange"]["api_key"] == ""
    assert cfg_bad["exchange"]["secret_key"] == ""
    api_key_env = cfg_bad["exchange"].get("api_key_env") or cfg_bad["exchange"].get("api_key") or ""
    secret_key_env = cfg_bad["exchange"].get("secret_key_env") or cfg_bad["exchange"].get("secret_key") or ""
    assert (not api_key_env) or (not secret_key_env)
    # also blank out dataclass copy if present
    if "exchange_cfg" in cfg_bad:
        cfg_bad["exchange_cfg"].api_key_env = ""
        cfg_bad["exchange_cfg"].secret_key_env = ""
        cfg_bad["exchange_cfg"].api_key = ""
        cfg_bad["exchange_cfg"].secret_key = ""
    with pytest.raises(ConfigValidationError):
        validate_config(cfg_bad)


def test_live_requires_kill_switch_enabled():
    cfg = load_config("live")
    cfg_bad = deepcopy(cfg)
    cfg_bad["kill_switch"]["enabled"] = False
    with pytest.raises(ConfigValidationError):
        validate_config(cfg_bad)
