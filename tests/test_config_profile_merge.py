from __future__ import annotations

from finantradealgo.system.config_loader import load_config


def test_base_fields_propagate_to_profiles():
    cfg_research = load_config("research")
    cfg_live = load_config("live")

    # base -> shared lookback values should be preserved across profiles
    for tf in ("1m", "5m", "15m", "1h"):
        assert cfg_research["data"]["lookback_days"][tf] == cfg_live["data"]["lookback_days"][tf]
    # base -> features.feature_preset should survive merge
    assert cfg_research["features"]["feature_preset"] == "extended"
    assert cfg_live["features"]["feature_preset"] == "extended"
    # base -> default symbol/timeframe not dropped
    assert cfg_research["symbol"] == "AIAUSDT"
    assert cfg_live["symbol"] == "AIAUSDT"


def test_profile_overrides_remain_distinct():
    cfg_research = load_config("research")
    cfg_live = load_config("live")

    # research keeps broad universe; live runtime enforces single symbol/timeframe
    assert len(cfg_research["data"]["symbols"]) > 1
    assert cfg_live["live_cfg"].symbols == ["AIAUSDT"]

    # research has multiple timeframes; live narrowed
    assert len(cfg_research["data"]["timeframes"]) >= 4
    assert cfg_live["live_cfg"].timeframe == "15m"

    # exchange.type differs by profile
    assert cfg_research["exchange"]["type"] == "backtest"
    assert cfg_live["exchange"]["type"] == "live"
