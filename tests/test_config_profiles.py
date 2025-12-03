from finantradealgo.system.config_loader import load_config
import pytest


def test_load_config_profiles():
    cfg_research = load_config("research")
    cfg_live = load_config("live")

    assert cfg_research["exchange"].get("type") == "backtest"
    assert cfg_live["exchange"].get("type") == "live"


def test_load_config_invalid_profile():
    with pytest.raises(ValueError):
        load_config("foobar")  # type: ignore[arg-type]
