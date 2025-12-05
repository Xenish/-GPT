import pytest

from finantradealgo.system.config_loader import load_config


def test_load_live_profile_basic_constraints():
    cfg = load_config("live")
    assert cfg["profile"] == "live"
    assert cfg["exchange"].get("type", "").lower() == "live"
    live_cfg = cfg["live_cfg"]
    assert live_cfg.mode in ("paper", "exchange")
    assert len(live_cfg.symbols) == 1
    assert live_cfg.symbol == live_cfg.symbols[0]
