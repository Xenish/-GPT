import pytest

from finantradealgo.system.config_loader import load_system_config


def test_load_research_profile_has_core_sections():
    cfg = load_system_config("config/system.research.yml")

    for section in ("exchange", "data", "strategy", "risk"):
        assert section in cfg, f"{section} missing from loaded config"

    assert cfg["_config_meta"]["is_profile"] is True
    assert cfg["live_cfg"].mode in ("replay", "live")
    assert cfg["exchange_cfg"].name
    assert cfg["data_cfg"].symbols or isinstance(cfg["data_cfg"].symbols, list)
    assert cfg["portfolio_cfg"].timeframe


def test_load_config_invalid_path_raises():
    with pytest.raises(FileNotFoundError):
        load_system_config("config/system.invalid_profile.yml")
