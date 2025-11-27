from __future__ import annotations

from finantradealgo.strategies.param_space import (
    apply_strategy_params_to_cfg,
    sample_params,
)
from finantradealgo.strategies.strategy_engine import get_strategy_meta


def test_rule_strategy_param_space_exists_and_samples():
    meta = get_strategy_meta("rule")
    assert meta.param_space is not None, "Rule strategy should expose a param space."
    keys = set(meta.param_space.keys())
    assert {"ms_trend_min", "ms_trend_max", "tp_atr_mult", "sl_atr_mult", "use_ms_chop_filter"} <= keys
    params = sample_params(meta.param_space)
    assert "tp_atr_mult" in params
    assert isinstance(params["tp_atr_mult"], float)
    assert isinstance(params["use_ms_chop_filter"], bool)


def test_apply_strategy_params_to_cfg_rule():
    cfg = {
        "strategy": {
            "rule": {
                "use_ms_chop_filter": True,
                "tp_atr_mult": 1.5,
            }
        }
    }
    updated = apply_strategy_params_to_cfg(cfg, "rule", {"tp_atr_mult": 2.0})
    assert updated is not cfg
    assert updated["strategy"]["rule"]["tp_atr_mult"] == 2.0
    assert updated["strategy"]["rule"]["use_ms_chop_filter"] is True
    # Original cfg remains unchanged
    assert cfg["strategy"]["rule"]["tp_atr_mult"] == 1.5
